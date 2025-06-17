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

app = FastAPI()
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG for more verbosity
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
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained("bert-base-uncased")
    classifier = joblib.load("classifier.joblib")
    with open("reference_code.txt", "r", encoding="utf-8") as f:
        reference_code = f.read()
    logger.info("Model, tokenizer, and reference code loaded successfully")
except Exception as e:
    logger.error(f"Failed to load resources: {str(e)}")
    raise

class CodeInput(BaseModel):
    code: str

class SimilarityInput(BaseModel):
    codes: List[str]

    # Add a custom validator to log the incoming data
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        logger.debug(f"Validating SimilarityInput: {value}")
        if isinstance(value, dict):
            codes = value.get("codes")
            if not isinstance(codes, list):
                logger.error(f"Invalid 'codes' field: expected list, got {type(codes)}")
                raise ValueError(f"'codes' must be a list, got {type(codes)}")
            for idx, code in enumerate(codes):
                if not isinstance(code, str):
                    logger.error(f"Invalid code at index {idx}: expected string, got {type(code)} with value {code}")
                    raise ValueError(f"Code at index {idx} must be a string, got {type(code)} with value {code}")
                if not code.strip():
                    logger.warning(f"Empty code string at index {idx}")
        return cls(**value)

def compute_levenshtein_similarity(code1, code2):
    logger.debug(f"Computing Levenshtein similarity between code1 (len={len(code1)}) and code2 (len={len(code2)})")
    dist = distance(code1, code2)
    max_len = max(len(code1), len(code2))
    similarity = 1 - (dist / max_len) if max_len > 0 else 1
    logger.debug(f"Levenshtein distance: {dist}, max_len: {max_len}, similarity: {similarity}")
    return similarity

def get_ast_max_depth(node, current_depth=0):
    logger.debug(f"Computing AST max depth at depth {current_depth}")
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
    logger.debug(f"Extracting AST features for code: {code[:50]}...")
    try:
        tree = ast.parse(code)
        num_nodes = sum(1 for _ in ast.walk(tree))
        max_depth = get_ast_max_depth(tree)
        func_defs = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        loops = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While)))
        logger.debug(f"AST features - num_nodes: {num_nodes}, max_depth: {max_depth}, func_defs: {func_defs}, loops: {loops}")
        return [num_nodes, max_depth, func_defs, loops]
    except SyntaxError as e:
        logger.warning(f"SyntaxError in AST parsing: {str(e)}")
        return [0, 0, 0, 0]

def extract_combined_features(texts, reference_code):
    device = torch.device("cpu")
    global model
    model = model.to(device)
    model_features = []
    logger.debug(f"Extracting combined features for {len(texts)} texts")
    for idx, text in enumerate(texts):
        logger.debug(f"Processing text {idx}: {text[:50]}...")
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding="max_length").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        model_features.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
    lev_features = [compute_levenshtein_similarity(text, reference_code) for text in texts]
    ast_features = [extract_ast_features(text) for text in texts]
    logger.debug(f"Combining features: model_features shape: {np.array(model_features).shape}, lev_features: {lev_features}, ast_features: {ast_features}")
    return np.hstack([
        np.array(model_features),
        np.array(lev_features).reshape(-1, 1),
        np.array(ast_features)
    ])

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to the AI Detection Backend", "status": "healthy"}

@app.post("/detect")
async def detect_ai_code(input: CodeInput):
    logger.info("Received /detect request")
    logger.debug(f"Input code: {input.code[:50]}...")
    try:
        if not input.code.strip():
            logger.error("Code is empty")
            raise HTTPException(status_code=400, detail="Code cannot be empty")
        features = extract_combined_features([input.code], reference_code)
        prediction = classifier.predict_proba(features)[0]
        ai_percentage = float(prediction[1] * 100)
        logger.info(f"Processed code, AI percentage: {ai_percentage:.2f}%")
        return {"ai_percentage": round(ai_percentage, 2)}
    except ValueError as ve:
        logger.error(f"Value error in AI detection: {str(ve)}")
        raise HTTPException(status_code=400, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        logger.error(f"Error processing code: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Server error during AI detection: {str(e)}")

@app.post("/similarity")
async def detect_code_similarity(input: SimilarityInput):
    logger.info("Received /similarity request")
    logger.debug(f"Input codes: {[code[:50] + '...' for code in input.codes]}")
    try:
        # Validate input
        if not input.codes or len(input.codes) < 2:
            logger.error(f"Insufficient codes: {len(input.codes)} provided, at least 2 required")
            raise HTTPException(status_code=400, detail="At least two code samples are required")

        # Check for empty or invalid codes
        valid_codes = []
        for idx, code in enumerate(input.codes):
            if not code.strip():
                logger.warning(f"Skipping empty code at index {idx}")
                continue
            valid_codes.append(code)
        if len(valid_codes) < 2:
            logger.error(f"After filtering, only {len(valid_codes)} valid codes remain, at least 2 required")
            raise HTTPException(status_code=400, detail="At least two non-empty code samples are required after filtering")

        # Compute similarities
        similarities = {}
        for i in range(len(valid_codes)):
            for j in range(i + 1, len(valid_codes)):
                code1 = valid_codes[i]
                code2 = valid_codes[j]
                logger.debug(f"Comparing codes at indices {i} and {j}")
                similarity = compute_levenshtein_similarity(code1, code2) * 100
                similarities[f"{i}-{j}"] = round(similarity, 2)

        logger.info(f"Computed similarities: {similarities}")
        return {"similarities": similarities}
    except HTTPException as he:
        logger.error(f"HTTP error: {str(he)}")
        raise
    except Exception as e:
        logger.error(f"Error processing similarity: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error processing similarity: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check endpoint accessed")
    return {"status": "healthy"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting Uvicorn on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)