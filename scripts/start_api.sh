#!/bin/bash

# Start FastAPI server

echo "🚀 Starting Credit Default Prediction API..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Virtual environment not activated. Activating..."
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
fi

# Set PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH

# Start FastAPI server
echo "🔄 Starting API server on http://localhost:8000..."
python api/fastapi_main.py
