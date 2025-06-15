"""
Entity classes for Credit Default Prediction
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path
    raw_data_path: Path
    train_data_path: Path
    test_data_path: Path
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True


@dataclass
class DataValidationConfig:
    """Configuration for data validation"""
    root_dir: Path
    status_file: Path
    unzip_data_dir: Path
    all_schema: Dict[str, Any]
    drift_report_file_path: Path


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    root_dir: Path
    data_path: Path
    preprocessor_obj_file_path: Path
    train_data_path: Path
    test_data_path: Path
    train_arr_path: Path
    test_arr_path: Path
    numerical_features: List[str]
    categorical_features: List[str]
    target_column: str
    create_ratio_features: bool = True
    create_payment_features: bool = True
    create_balance_features: bool = True


@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    model_name: str
    target_column: str
    model_path: Path
    best_model_path: Path
    metric_file_path: Path
    algorithms: Dict[str, Any]
    cv_folds: int = 5
    scoring: str = "roc_auc"
    random_state: int = 42


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    root_dir: Path
    test_data_path: Path
    model_path: Path
    all_params: Dict[str, Any]
    metric_file_name: Path
    target_column: str
    mlflow_uri: str


@dataclass
class ExplainerConfig:
    """Configuration for explainer"""
    root_dir: Path
    model_path: Path
    test_data_path: Path
    explainer_path: Path
    feature_names_path: Path
    sample_size: int = 1000


@dataclass
class PredictionConfig:
    """Configuration for prediction"""
    model_path: Path
    preprocessor_path: Path
    explainer_path: Path
    feature_names_path: Path


@dataclass
class TrainingPipelineConfig:
    """Configuration for the entire training pipeline"""
    artifacts_root: Path = Path("artifacts")


@dataclass
class PredictionPipelineConfig:
    """Configuration for prediction pipeline"""
    model_path: Path
    preprocessor_path: Path
    explainer_path: Path


@dataclass
class PredictionInput:
    """Input data for prediction"""
    LIMIT_BAL: int
    SEX: int
    EDUCATION: int
    MARRIAGE: int
    AGE: int
    PAY_0: int
    PAY_2: int
    PAY_3: int
    PAY_4: int
    PAY_5: int
    PAY_6: int
    BILL_AMT1: float
    BILL_AMT2: float
    BILL_AMT3: float
    BILL_AMT4: float
    BILL_AMT5: float
    BILL_AMT6: float
    PAY_AMT1: float
    PAY_AMT2: float
    PAY_AMT3: float
    PAY_AMT4: float
    PAY_AMT5: float
    PAY_AMT6: float


@dataclass
class PredictionOutput:
    """Output data for prediction"""
    prediction: int
    probability: float
    risk_score: float
    risk_category: str
    shap_explanation: Optional[Dict[str, Any]] = None


@dataclass
class ModelMetrics:
    """Model evaluation metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    confusion_matrix: List[List[int]]
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class BatchPredictionInput:
    """Input for batch prediction"""
    data_path: Path
    output_path: Path
    include_explanations: bool = False


@dataclass
class BatchPredictionOutput:
    """Output for batch prediction"""
    predictions_path: Path
    summary_stats: Dict[str, Any]
    total_processed: int
    success_count: int
    error_count: int
