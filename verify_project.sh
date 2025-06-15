#!/bin/bash

# Project Verification Script
echo "🔍 Verifying Credit Default Prediction Project Structure..."

# Check key directories
directories=(
    "src/credit_default"
    "config"
    "api"
    "dashboard"
    "deployment"
    "scripts"
    "tests"
)

echo "📁 Checking directories..."
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        echo "  ✅ $dir"
    else
        echo "  ❌ $dir (missing)"
    fi
done

# Check key files
files=(
    "requirements.txt"
    "setup.py"
    "README.md"
    "src/credit_default/pipeline/training_pipeline.py"
    "src/credit_default/pipeline/prediction_pipeline.py"
    "api/fastapi_main.py"
    "dashboard/streamlit_dashboard.py"
    "deployment/Dockerfile"
    "deployment/docker-compose.yml"
)

echo ""
echo "📄 Checking key files..."
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (missing)"
    fi
done

echo ""
echo "🎯 Project verification complete!"
echo "📖 Next steps:"
echo "  1. Run: chmod +x scripts/setup.sh && ./scripts/setup.sh"
echo "  2. Train model: python src/credit_default/pipeline/training_pipeline.py"
echo "  3. Start API: python api/fastapi_main.py"
echo "  4. Start dashboard: streamlit run dashboard/streamlit_dashboard.py"
