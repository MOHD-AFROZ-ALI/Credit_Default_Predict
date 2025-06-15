"""
Constants for Credit Default Prediction project
"""
import os
from pathlib import Path

# Root directories
ROOT_DIR = Path(__file__).parent.parent.parent.parent
CONFIG_DIR = ROOT_DIR / "config"
DATA_DIR = ROOT_DIR / "data"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOGS_DIR = ROOT_DIR / "logs"

# Configuration files
SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"
MODEL_CONFIG_FILE_PATH = CONFIG_DIR / "model.yaml"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Artifact paths
DATA_INGESTION_DIR = ARTIFACTS_DIR / "data_ingestion"
DATA_VALIDATION_DIR = ARTIFACTS_DIR / "data_validation"
DATA_TRANSFORMATION_DIR = ARTIFACTS_DIR / "data_transformation"
MODEL_TRAINER_DIR = ARTIFACTS_DIR / "model_trainer"
EXPLAINER_DIR = ARTIFACTS_DIR / "explainer"

# File names
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
RAW_FILE_NAME = "credit_default.csv"
PREPROCESSOR_FILE_NAME = "preprocessor.pkl"
MODEL_FILE_NAME = "model.pkl"
BEST_MODEL_FILE_NAME = "best_model.pkl"
METRICS_FILE_NAME = "metrics.yaml"
EXPLAINER_FILE_NAME = "shap_explainer.pkl"
FEATURE_NAMES_FILE_NAME = "feature_names.pkl"

# Data source
DATA_SOURCE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

# Model parameters
TARGET_COLUMN = "default.payment.next.month"
RANDOM_STATE = 42
TEST_SIZE = 0.2

# API parameters
API_HOST = "0.0.0.0"
API_PORT = 8000

# Dashboard parameters
DASHBOARD_HOST = "0.0.0.0"
DASHBOARD_PORT = 8501

# MLflow parameters
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "credit_default_prediction"

# Feature categories
NUMERICAL_FEATURES = [
    "LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", 
    "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2", 
    "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
]

CATEGORICAL_FEATURES = [
    "SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", 
    "PAY_4", "PAY_5", "PAY_6"
]

# Model algorithms
ALGORITHMS = {
    "xgboost": "XGBClassifier",
    "random_forest": "RandomForestClassifier", 
    "gradient_boosting": "GradientBoostingClassifier",
    "logistic_regression": "LogisticRegression"
}

# Evaluation metrics
EVALUATION_METRICS = [
    "accuracy", "precision", "recall", "f1_score", "roc_auc"
]

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "credit_default.log"
