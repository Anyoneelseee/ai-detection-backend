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

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://my-project-five-plum.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model, tokenizer, and reference code
try:
    # Use a smaller model to reduce memory usage
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
    codebert = AutoModel.from_pretrained("distilroberta-base")
    model = joblib.load("classifier.joblib")
    with open("reference_code.txt", "r", encoding="utf-8") as f:
        reference_code = f.read()
    logger.info("Model, tokenizer, and reference code loaded successfully")
except Exception as e:
    logger.error(f"Failed to load resources: {str(e)}")
    raise

class CodeInput(BaseModel):
    code: str

class SimilarityInput(BaseModel):
    codes: list[str]

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
    device = torch.device("cpu")  # Force CPU usage to reduce memory (CUDA not available on Render free tier)
    global codebert  # Use the globally loaded model
    codebert = codebert.to(device)
    codebert_features = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device)  # Reduced max_length
        with torch.no_grad():
            outputs = codebert(**inputs)
        codebert_features.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
    lev_features = [compute_levenshtein_similarity(text, reference_code) for text in texts]
    ast_features = [extract_ast_features(text) for text in texts]
    return np.hstack([
        np.array(codebert_features),
        np.array(lev_features).reshape(-1, 1),
        np.array(ast_features)
    ])

@app.post("/detect")
async def detect_ai_code(input: CodeInput):
    try:
        if not input.code.strip():
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        features = extract_combined_features([input.code], reference_code)
        prediction = model.predict_proba(features)[0]
        ai_percentage = float(prediction[1] * 100)
        logger.info(f"Processed code, AI percentage: {ai_percentage:.2f}%")
        return {"ai_percentage": round(ai_percentage, 2)}
    except ValueError as ve:
        logger.error(f"Value error in AI detection: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing code: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server error during AI detection: {str(e)}")

@app.post("/similarity")
async def detect_code_similarity(input: SimilarityInput):
    try:
        if not input.codes or len(input.codes) < 2:
            raise HTTPException(status_code=400, detail="At least two code samples are required")
        
        similarities = {}
        for i in range(len(input.codes)):
            for j in range(i + 1, len(input.codes)):
                code1 = input.codes[i].strip()
                code2 = input.codes[j].strip()
                if not code1 or not code2:
                    continue
                similarity = compute_levenshtein_similarity(code1, code2) * 100
                similarities[f"{i}-{j}"] = round(similarity, 2)
        
        logger.info(f"Computed similarities: {similarities}")
        return {"similarities": similarities}
    except Exception as e:
        logger.error(f"Error processing similarity: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing similarity: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)