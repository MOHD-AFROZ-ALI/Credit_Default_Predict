#!/bin/bash

# Credit Default Prediction Training Script

echo "🚀 Starting Credit Default Prediction Training Pipeline..."

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

# Run training pipeline
echo "🔄 Running training pipeline..."
python src/credit_default/pipeline/training_pipeline.py

echo "✅ Training pipeline completed!"
