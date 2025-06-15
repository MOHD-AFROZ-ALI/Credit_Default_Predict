#!/bin/bash

# Start Streamlit Dashboard

echo "🚀 Starting Credit Default Prediction Dashboard..."

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

# Start Streamlit dashboard
echo "🔄 Starting dashboard on http://localhost:8501..."
streamlit run dashboard/streamlit_dashboard.py --server.port 8501 --server.address 0.0.0.0
