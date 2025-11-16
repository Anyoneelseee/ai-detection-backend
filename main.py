# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import pipeline
from Levenshtein import distance
import joblib
import logging
import os
import uvicorn
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://your-app.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 0 if torch.cuda.is_available() else -1

# === CODEBERT (PYTHON ONLY) ===
detector = None
try:
    detector = pipeline(
        "text-classification",
        model="models/codebert-detector/fine_tuned",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        truncation=True,        # ← FIX: truncate long code
        max_length=512          # ← FIX: max 512 tokens
    )
    logger.info("CodeBERT loaded (Python only)")
except Exception as e:
    logger.error(f"CodeBERT failed: {e}")

# === CLASSIC ML (C/C++/JAVA) ===
classic_classifier = None
vectorizer = None
model_path = "models/codebert-detector/classifier.joblib"
if os.path.exists(model_path):
    try:
        loaded = joblib.load(model_path)
        if isinstance(loaded, dict):
            classic_classifier = loaded.get('model') or loaded.get('classifier')
            vectorizer = loaded.get('vectorizer')
        else:
            classic_classifier = loaded  # assume it's the model

        if classic_classifier is None or vectorizer is None:
            raise ValueError("Invalid joblib structure")

        # Fit vectorizer if not already fitted
        if not hasattr(vectorizer, 'vocabulary_') or vectorizer.vocabulary_ is None:
            dummy_codes = [
                "#include <iostream>", "int main()", "public class Main",
                "printf(\"hello\")", "vector<int> arr;", "System.out.println",
                "for(int i=0; i<n; i++)", "return 0;", "import java.util.*;", "class Solution"
            ]
            vectorizer.fit(dummy_codes)
            logger.info("TF-IDF vectorizer fitted from joblib")

        logger.info("ML model loaded (C/C++/Java)")
    except Exception as e:
        logger.warning(f"Failed to load ML model: {e}")
        classic_classifier = None
        vectorizer = None

class CodeInput(BaseModel):
    code: str

def detect_language(code: str) -> str:
    code_lower = code.lower().strip()
    if not code_lower:
        return "unknown"
    if any(kw in code_lower for kw in ["def ", "import ", "print(", "class ", "from "]):
        return "python"
    if "#include" in code_lower and ("stdio.h" in code_lower or "stdlib.h" in code_lower):
        return "c"
    if "#include" in code_lower and ("iostream" in code_lower or "vector" in code_lower or "string" in code_lower):
        return "cpp"
    if any(kw in code_lower for kw in ["public class", "system.out", "void main", "string[]"]):
        return "java"
    return "unknown"

@app.post("/api/bulk-ai-detector")
async def bulk_ai_detector(inputs: List[CodeInput]):
    results = []
    for inp in inputs:
        code = inp.code.strip()
        if not code:
            results.append({"ai_percentage": None, "error": "Empty code", "cached": False})
            continue

        lang = detect_language(code)
        try:
            if lang == "python" and detector:
                # CodeBERT: safe with truncation
                r = detector(code, truncation=True, max_length=512)[0]
                score = r['score'] if r['label'] == 'LABEL_1' else 1 - r['score']
            elif lang in ["c", "cpp", "java"] and classic_classifier and vectorizer:
                # ML: use fitted vectorizer
                X = vectorizer.transform([code])
                prob = classic_classifier.predict_proba(X)
                score = prob[0][1] if prob.shape[1] > 1 else prob[0][0]
            else:
                results.append({"ai_percentage": None, "error": f"Unsupported: {lang}", "cached": False})
                continue

            results.append({
                "ai_percentage": round(score * 100, 2),
                "error": None,
                "cached": False
            })
        except Exception as e:
            logger.error(f"AI failed for {lang}: {e}")
            results.append({"ai_percentage": None, "error": "Detection failed", "cached": False})
    return results

@app.post("/similarity")
async def similarity(input: dict):
    codes = [c.strip() for c in input.get("codes", []) if c.strip()]
    if len(codes) < 2:
        raise HTTPException(400, "Need 2+ codes")
    sims = {}
    for i in range(len(codes)):
        for j in range(i+1, len(codes)):
            s = round((1 - distance(codes[i], codes[j]) / max(len(codes[i]), len(codes[j]), 1)) * 100, 2)
            sims[f"{i}-{j}"] = s
    return {"similarities": sims}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)