#!/bin/bash
echo "Pulling LFS files..."
git lfs pull
echo "Starting server..."
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1