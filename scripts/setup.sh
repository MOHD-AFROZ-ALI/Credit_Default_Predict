#!/bin/bash

# Credit Default Prediction Setup Script

echo "🚀 Setting up Credit Default Prediction Project..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing requirements..."
pip install -r requirements.txt

# Install project in development mode
echo "🔧 Installing project..."
pip install -e .

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p artifacts/{data_ingestion,data_validation,data_transformation,model_trainer,explainer}
mkdir -p logs
mkdir -p data/{raw,processed}

echo "✅ Setup completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate (Linux/Mac) or venv\Scripts\activate (Windows)"
echo "2. Run training pipeline: python src/credit_default/pipeline/training_pipeline.py"
echo "3. Start API server: python api/fastapi_main.py"
echo "4. Start dashboard: streamlit run dashboard/streamlit_dashboard.py"
echo ""
echo "📖 For more information, check the README.md file"
