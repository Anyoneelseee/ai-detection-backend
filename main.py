from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from Levenshtein import distance
import ast
import logging
import os
import uvicorn
import traceback
from typing import List
import subprocess  # For git lfs pull
import sys

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------------------------------------
# CORS SETUP
# -----------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://my-project-five-plum.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------
# PULL LFS MODEL AT STARTUP (Render has git, but not git-lfs)
# -----------------------------------------------------
MODEL_DIR = "./codeberta"
MODEL_FILE = os.path.join(MODEL_DIR, "pytorch_model.bin")
LFS_MIN_SIZE = 100_000_000  # 100 MB

def pull_lfs_model():
    if not os.path.exists(MODEL_FILE):
        logger.warning("Model file missing. Attempting to pull via Git LFS...")
    elif os.path.getsize(MODEL_FILE) < LFS_MIN_SIZE:
        logger.warning(f"Model file too small ({os.path.getsize(MODEL_FILE)} bytes). Likely LFS pointer. Pulling...")
    else:
        logger.info("Model file already downloaded and full size.")
        return

    # Try to pull using git lfs
    try:
        logger.info("Running: git lfs pull --include codeberta/pytorch_model.bin")
        result = subprocess.run(
            ["git", "lfs", "pull", "--include", "codeberta/pytorch_model.bin"],
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes
        )
        logger.info("Git LFS pull successful.")
        logger.debug(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Git LFS pull failed: {e.stderr}")
        logger.info("Continuing without model — will crash on load.")
    except FileNotFoundError:
        logger.error("git command not found. Cannot pull LFS.")
        logger.info("Falling back to offline mode.")
    except subprocess.TimeoutExpired:
        logger.error("Git LFS pull timed out after 5 minutes.")
        logger.info("Continuing — model may be partial.")

# Run LFS pull at startup
pull_lfs_model()

# -----------------------------------------------------
# LOAD MODEL + TOKENIZER + SCALER
# -----------------------------------------------------
try:
    logger.info("Loading tokenizer and model from ./codeberta...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModel.from_pretrained(MODEL_DIR)

    logger.info("Loading classifier from classifier.joblib...")
    model_bundle = joblib.load("classifier.joblib")
    classifier = model_bundle["model"]
    scaler = model_bundle["scaler"]
    threshold = model_bundle.get("threshold", 0.7)

    logger.info("Loading reference code...")
    with open("reference_code.txt", "r", encoding="utf-8") as f:
        reference_code = f.read()

    logger.info("All resources loaded successfully (CodeBERTa-small, ~336 MB)")

except Exception as e:
    logger.error(f"Failed to load resources: {str(e)}")
    logger.error(traceback.format_exc())
    raise

# -----------------------------------------------------
# DATA MODELS
# -----------------------------------------------------
class CodeInput(BaseModel):
    code: str

class SimilarityInput(BaseModel):
    codes: List[str]

# -----------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------
def compute_levenshtein_similarity(code1, code2):
    dist = distance(code1, code2)
    max_len = max(len(code1), len(code2))
    return 1 - (dist / max_len) if max_len > 0 else 1

def get_ast_max_depth(node, current_depth=0):
    if not isinstance(node, ast.AST):
        return current_depth
    max_child_depth = current_depth
    for field, value in ast.iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    child_depth = get_ast_max_depth(item, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)
        elif isinstance(value, ast.AST):
            child_depth = get_ast_max_depth(value, current_depth + 1)
            max_child_depth = max(max_child_depth, child_depth)
    return max_child_depth

def extract_ast_features(code):
    try:
        tree = ast.parse(code)
        num_nodes = sum(1 for _ in ast.walk(tree))
        max_depth = get_ast_max_depth(tree)
        func_defs = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
        return [num_nodes, max_depth, func_defs, loops]
    except SyntaxError:
        return [0, 0, 0, 0]

def extract_combined_features(texts, reference_code):
    device = torch.device("cpu")
    model.to(device)
    model_features = []
    for text in texts:
        code = "[CODE] " + text[:256]
        inputs = tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        model_features.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())

    lev_features = [compute_levenshtein_similarity(text, reference_code) for text in texts]
    ast_features = [extract_ast_features(text) for text in texts]

    return np.hstack([
        np.array(model_features),
        np.array(lev_features).reshape(-1, 1),
        np.array(ast_features)
    ])

# -----------------------------------------------------
# ROUTES
# -----------------------------------------------------
@app.get("/")
async def root():
    return {"message": "AI Detection Backend (CodeBERTa-small)", "status": "healthy"}

@app.post("/detect")
async def detect_ai_code(input: CodeInput):
    try:
        if not input.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")

        features = extract_combined_features([input.code], reference_code)
        features = scaler.transform(features)

        probs = classifier.predict_proba(features)[0]
        is_ai = probs[1] > threshold
        ai_percentage = float(probs[1] * 100)

        label = "AI-generated" if is_ai else "Human-written"
        logger.info(f"Processed → {label} ({ai_percentage:.2f}%)")

        return {
            "ai_percentage": round(ai_percentage, 2),
            "label": label
        }

    except Exception as e:
        logger.error(f"Error in /detect: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Server error")

@app.post("/similarity")
async def detect_code_similarity(input: SimilarityInput):
    try:
        if len(input.codes) < 2:
            raise HTTPException(status_code=400, detail="Need 2+ codes")

        valid_codes = [c for c in input.codes if c.strip()]
        if len(valid_codes) < 2:
            raise HTTPException(status_code=400, detail="Need 2+ non-empty codes")

        similarities = {}
        for i in range(len(valid_codes)):
            for j in range(i + 1, len(valid_codes)):
                sim = compute_levenshtein_similarity(valid_codes[i], valid_codes[j]) * 100
                similarities[f"{i}-{j}"] = round(sim, 2)

        return {"similarities": similarities}

    except Exception as e:
        logger.error(f"Similarity error: {str(e)}")
        raise HTTPException(status_code=500, detail="Similarity error")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# -----------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)