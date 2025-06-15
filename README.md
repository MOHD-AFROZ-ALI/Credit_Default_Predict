# Credit Default Prediction - End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.101+-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)

> **End-to-End Credit Default Prediction with Explainable ML API & Dashboard**

A production-ready machine learning solution for credit risk assessment with comprehensive explainability features, designed for fintech companies and financial institutions.

## 🌟 Features

- **Advanced ML Pipeline**: Multiple algorithms with hyperparameter tuning
- **Explainable AI**: SHAP-based explanations for model decisions
- **Production API**: FastAPI backend with comprehensive endpoints
- **Interactive Dashboard**: Streamlit interface for risk analysis
- **Containerized Deployment**: Docker and Docker Compose ready
- **Feature Engineering**: 23+ engineered features from financial data
- **Real-time Predictions**: Single and batch prediction capabilities
- **Model Monitoring**: Performance tracking and validation

## 🏗️ Architecture

```
Credit Default Prediction System
├── Data Pipeline
│   ├── Data Ingestion (UCI Dataset)
│   ├── Data Validation (Schema + Drift Detection)
│   └── Data Transformation (Feature Engineering)
├── ML Pipeline
│   ├── Model Training (Multiple Algorithms)
│   ├── Hyperparameter Tuning (GridSearchCV)
│   └── Model Evaluation (Cross-validation)
├── Explainability Layer
│   ├── SHAP Global Explanations
│   ├── SHAP Local Explanations
│   └── Interactive Visualizations
├── API Layer
│   ├── FastAPI Backend
│   ├── Prediction Endpoints
│   └── Explanation Endpoints
├── Frontend Layer
│   ├── Streamlit Dashboard
│   ├── Interactive Risk Assessment
│   └── Batch Processing Interface
└── Deployment Layer
    ├── Docker Containerization
    ├── CI/CD Pipeline Support
    └── Cloud Deployment Ready
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Git
- 4GB+ RAM recommended
- Docker & Docker Compose (optional)

### 1. Clone & Setup

```bash
# Clone the repository
git clone <repository-url>
cd credit_default_prediction

# Run setup script
chmod +x scripts/setup.sh
./scripts/setup.sh
```

### 2. Train the Model

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate    # Windows

# Run training pipeline
python src/credit_default/pipeline/training_pipeline.py
# or
./scripts/train.sh
```

### 3. Start Services

#### Option 1: Using Docker Compose (Recommended)

```bash
cd deployment
docker-compose up -d
```

#### Option 2: Manual Startup

```bash
# Start API server (Terminal 1)
./scripts/start_api.sh

# Start Dashboard (Terminal 2)
./scripts/start_dashboard.sh
```

### 4. Access Applications

- **API Documentation**: http://localhost:8000/docs
- **Interactive Dashboard**: http://localhost:8501
- **MLflow Tracking**: http://localhost:5000 (if using Docker Compose)

## 📊 Model Performance

| Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-----------|----------|-----------|--------|----------|---------|
| XGBoost | 0.823 | 0.756 | 0.689 | 0.721 | 0.891 |
| Random Forest | 0.816 | 0.741 | 0.678 | 0.708 | 0.885 |
| Gradient Boosting | 0.819 | 0.748 | 0.672 | 0.708 | 0.887 |
| Logistic Regression | 0.801 | 0.695 | 0.634 | 0.663 | 0.856 |

*Best Model: **XGBoost** with hyperparameter tuning*

## 🧠 Explainable AI Features

### Global Explanations
- **Feature Importance**: Ranking of most predictive features
- **SHAP Summary Plots**: Overall impact of features across all predictions
- **Partial Dependence Plots**: Feature effect visualization

### Local Explanations
- **SHAP Force Plots**: Individual prediction breakdown
- **SHAP Waterfall Plots**: Step-by-step prediction explanation
- **Feature Contribution Analysis**: Positive/negative impact identification

## 📁 Project Structure

```
credit_default_prediction/
├── src/
│   └── credit_default/
│       ├── components/           # ML pipeline components
│       ├── configuration/        # Configuration management
│       ├── constants/           # Project constants
│       ├── entity/              # Data classes and entities
│       ├── exception/           # Custom exception handling
│       ├── logger/              # Logging configuration
│       ├── pipeline/            # Training and prediction pipelines
│       └── utils/               # Utility functions
├── config/
│   ├── schema.yaml              # Data schema configuration
│   └── model.yaml               # Model configuration
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/               # Processed data files
├── artifacts/                   # Pipeline artifacts
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   └── explainer/
├── api/
│   └── fastapi_main.py          # FastAPI application
├── dashboard/
│   └── streamlit_dashboard.py   # Streamlit application
├── deployment/
│   ├── Dockerfile               # Docker configuration
│   └── docker-compose.yml       # Multi-service orchestration
├── scripts/                     # Execution scripts
├── tests/                       # Unit and integration tests
├── logs/                        # Application logs
└── notebooks/                   # Jupyter notebooks for EDA
```

## 🔌 API Endpoints

### Prediction Endpoints

```bash
# Single prediction
POST /predict
Content-Type: application/json

{
  "LIMIT_BAL": 200000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 35,
  "PAY_0": 1,
  "PAY_2": 2,
  ...
}
```

### Explanation Endpoints

```bash
# Get prediction with explanation
POST /explain
Content-Type: application/json

# Same payload as /predict
# Returns SHAP explanations + visualizations
```

### Batch Processing

```bash
# Batch predictions
POST /batch-predict
Content-Type: multipart/form-data

# Upload CSV file with customer data
# Returns batch predictions + summary statistics
```

### Utility Endpoints

```bash
GET /health              # Health check
GET /model-info          # Model metadata
POST /sample-prediction  # Generate sample prediction
GET /feature-schema      # Get input schema
```

## 🎛️ Dashboard Features

### Single Customer Analysis
- **Risk Assessment Form**: Input customer details
- **Real-time Prediction**: Instant risk scoring
- **SHAP Explanations**: Feature contribution analysis
- **Risk Visualization**: Gauge charts and indicators

### Batch Processing
- **CSV Upload**: Process multiple customers
- **Summary Statistics**: Aggregate risk metrics
- **Risk Distribution**: Visual analytics
- **Export Results**: Download predictions

### Model Analytics
- **Performance Metrics**: Model evaluation scores
- **Feature Importance**: Global explanations
- **Model Information**: Architecture details

## 🐳 Deployment Options

### Local Development

```bash
# Quick start with Docker
docker-compose up -d

# Manual startup
source venv/bin/activate
python api/fastapi_main.py &
streamlit run dashboard/streamlit_dashboard.py
```

### Production Deployment

```bash
# Build production images
docker build -t credit-default-api .
docker build -t credit-default-dashboard .

# Deploy with orchestration
docker-compose -f docker-compose.prod.yml up -d
```

## 🔧 Configuration

### Model Configuration (`config/model.yaml`)
- Data source settings
- Feature engineering parameters
- Model hyperparameters
- Training configurations

### Schema Configuration (`config/schema.yaml`)
- Input feature definitions
- Data validation rules
- Quality thresholds

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=./ --cov-report=html

# Run specific test categories
python -m pytest tests/test_components.py -v
python -m pytest tests/test_pipeline.py -v
```

## 📈 Monitoring & Observability

### Application Monitoring
- Health checks and service monitoring
- Performance metrics tracking
- Error tracking and alerting

### Model Monitoring
- Prediction drift detection
- Model performance tracking
- Feature importance monitoring

## 🔒 Security & Compliance

### Data Security
- Input validation and sanitization
- Secure API endpoints
- Data encryption support

### Model Security
- Model versioning and artifact management
- Prediction audit trails
- Explainability for compliance

## 🤝 Contributing

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add unit tests for new features
- Update documentation for API changes
- Ensure Docker build passes

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UCI Machine Learning Repository** for the Credit Default Dataset
- **Open Source Community** for the excellent libraries and frameworks
- **SHAP** for explainable AI capabilities

## 📞 Support

For questions, issues, or contributions:

1. **Check existing issues**: [GitHub Issues](https://github.com/your-username/credit-default-prediction/issues)
2. **Create new issue**: Detailed bug reports or feature requests
3. **Documentation**: Check the `/docs` directory for detailed guides

## 🚀 Future Enhancements

### Technical Improvements
- Real-time model retraining pipeline
- A/B testing framework for model variants
- Advanced ensemble methods
- GPU acceleration support

### Business Features
- Multi-model comparison interface
- Custom risk threshold settings
- Regulatory compliance reporting
- Integration with external credit bureaus

---

**Built for Production-Ready ML Engineering**

*This project demonstrates enterprise-grade ML engineering practices suitable for fintech environments.*
