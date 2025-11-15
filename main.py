# api.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import pipeline
from Levenshtein import distance
import logging
import os
import uvicorn
from typing import List

app = FastAPI(title="AI Code Detector v2", version="2.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://my-project-five-plum.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# LOAD YOUR 99.77% MODEL
try:
    device = 0 if torch.cuda.is_available() else -1
    detector = pipeline(
        "text-classification",
        model="models/codebert-detector/fine_tuned",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32
    )
    logger.info("AI Detector v2 loaded: 99.77% accuracy")
except Exception as e:
    logger.error(f"Model load failed: {e}")
    raise

class CodeInput(BaseModel):
    code: str

class SimilarityInput(BaseModel):
    codes: List[str]

def compute_levenshtein_similarity(code1: str, code2: str) -> float:
    dist = distance(code1, code2)
    max_len = max(len(code1), len(code2), 1)
    return round((1 - dist / max_len) * 100, 2)

@app.get("/")
async def root():
    return {"message": "AI Code Detector v2", "accuracy": "99.77%"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/detect")
async def detect_ai_code(input: CodeInput):
    if not input.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")
    try:
        result = detector(input.code)[0]
        ai_score = result['score'] if result['label'] == 'LABEL_1' else 1 - result['score']
        return {"ai_percentage": round(ai_score * 100, 2)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Detection failed")

@app.post("/similarity")
async def detect_code_similarity(input: SimilarityInput):
    valid_codes = [c for c in input.codes if c.strip()]
    if len(valid_codes) < 2:
        raise HTTPException(status_code=400, detail="Need 2+ non-empty codes")
    similarities = {}
    for i in range(len(valid_codes)):
        for j in range(i + 1, len(valid_codes)):
            sim = compute_levenshtein_similarity(valid_codes[i], valid_codes[j])
            similarities[f"{i}-{j}"] = sim
    return {"similarities": similarities}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, workers=1)