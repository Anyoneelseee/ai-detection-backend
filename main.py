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

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://my-project-five-plum.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = 0 if torch.cuda.is_available() else -1

# === CODEBERT (PYTHON) ===
detector = None
try:
    detector = pipeline(
        "text-classification",
        model="models/codebert-detector/fine_tuned",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        truncation=True,
        max_length=512
    )
    logger.info("CodeBERT loaded (Python only)")
except Exception as e:
    logger.error(f"CodeBERT load failed: {e}")

# === CLASSIC ML (C/C++/JAVA) — DISABLED UNTIL FIXED ===
classic_classifier = None
vectorizer = None
logger.warning("C/C++/Java detection DISABLED — joblib invalid")

# === INPUT MODEL ===
class CodeInput(BaseModel):
    code: str

# === LANGUAGE DETECTION ===
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

# === AI DETECTION ENDPOINT ===
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
                r = detector(code, truncation=True, max_length=512)[0]
                logger.info(f"Raw CodeBERT output: {r}")  # DEBUG

                # CRITICAL FIX: Check what LABEL_1 means
                # Test with AI code → if score low, swap this line
                score = r['score'] if r['label'] == 'LABEL_1' else 1 - r['score']
                # If AI code shows low %, use this instead:
                # score = r['score'] if r['label'] == 'LABEL_0' else 1 - r['score']

            elif lang in ["c", "cpp", "java"]:
                results.append({
                    "ai_percentage": None,
                    "error": "C/C++/Java not supported yet",
                    "cached": False
                })
                continue
            else:
                results.append({"ai_percentage": None, "error": f"Unsupported: {lang}", "cached": False})
                continue

            results.append({
                "ai_percentage": round(score * 100, 2),
                "error": None,
                "cached": False
            })
        except Exception as e:
            logger.error(f"AI detection failed: {e}")
            results.append({"ai_percentage": None, "error": "Detection failed", "cached": False})
    return results

# === SIMILARITY ENDPOINT ===
@app.post("/similarity")
async def similarity(input: dict):
    codes = [c.strip() for c in input.get("codes", []) if c.strip()]
    if len(codes) < 2:
        raise HTTPException(400, "Need 2+ code snippets")
    sims = {}
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            d = distance(codes[i], codes[j])
            norm = max(len(codes[i]), len(codes[j]), 1)
            sim = round((1 - d / norm) * 100, 2)
            sims[f"{i}-{j}"] = sim
    return {"similarities": sims}

# === HEALTH CHECK ===
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "codebert_loaded": detector is not None,
        "ml_disabled": True,
        "model_path": os.path.exists("models/codebert-detector/fine_tuned")
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=False)