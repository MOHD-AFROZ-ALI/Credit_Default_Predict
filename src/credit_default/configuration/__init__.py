"""
Configuration Manager for Credit Default Prediction
"""
import sys
from pathlib import Path
from typing import Dict, Any

from credit_default.constants import *
from credit_default.utils import read_yaml, create_directories
from credit_default.entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ExplainerConfig,
    TrainingPipelineConfig
)
from credit_default.exception import ConfigurationException
from credit_default.logger import logger


class ConfigurationManager:
    """Configuration manager for the entire pipeline"""

    def __init__(self, 
                 config_filepath: Path = CONFIG_DIR / "model.yaml",
                 schema_filepath: Path = CONFIG_DIR / "schema.yaml"):
        """
        Initialize configuration manager

        Args:
            config_filepath: Path to config file
            schema_filepath: Path to schema file
        """
        try:
            self.config = read_yaml(config_filepath)
            self.schema = read_yaml(schema_filepath)

            # Create artifacts directory
            create_directories([Path(self.config.get("artifacts_root", "artifacts"))])

        except Exception as e:
            raise ConfigurationException(f"Error initializing configuration: {e}", sys)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration"""
        try:
            config = self.config.data_ingestion

            create_directories([Path(config.root_dir) if isinstance(config.root_dir, str) else config.root_dir])

            data_ingestion_config = DataIngestionConfig(
                root_dir=Path(config.get("root_dir", "artifacts/data_ingestion")),
                source_url=self.config.data_source.url,
                local_data_file=Path(config.get("local_data_file", "data/raw/credit_default.xls")),
                unzip_dir=Path(config.get("unzip_dir", "data/raw")),
                raw_data_path=Path(config.get("raw_data_path", "data/raw/credit_default.csv")),
                train_data_path=Path(config.get("train_data_path", "artifacts/data_ingestion/train.csv")),
                test_data_path=Path(config.get("test_data_path", "artifacts/data_ingestion/test.csv")),
                test_size=config.get("test_size", 0.2),
                random_state=config.get("random_state", 42),
                stratify=config.get("stratify", True)
            )

            return data_ingestion_config

        except Exception as e:
            raise ConfigurationException(f"Error getting data ingestion config: {e}", sys)

    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration"""
        try:
            config = self.config.data_validation

            create_directories([Path(config.get("root_dir", "artifacts/data_validation"))])

            data_validation_config = DataValidationConfig(
                root_dir=Path(config.get("root_dir", "artifacts/data_validation")),
                status_file=Path(config.get("validation_status_file", "artifacts/data_validation/status.txt")),
                unzip_data_dir=Path("data/raw"),
                all_schema=self.schema,
                drift_report_file_path=Path(config.get("drift_report_file", "artifacts/data_validation/drift_report.yaml"))
            )

            return data_validation_config

        except Exception as e:
            raise ConfigurationException(f"Error getting data validation config: {e}", sys)

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration"""
        try:
            config = self.config.data_transformation

            create_directories([Path(config.get("root_dir", "artifacts/data_transformation"))])

            data_transformation_config = DataTransformationConfig(
                root_dir=Path(config.get("root_dir", "artifacts/data_transformation")),
                data_path=Path("data/raw/credit_default.csv"),
                preprocessor_obj_file_path=Path(config.get("preprocessor_path", "artifacts/data_transformation/preprocessor.pkl")),
                train_data_path=Path("artifacts/data_ingestion/train.csv"),
                test_data_path=Path("artifacts/data_ingestion/test.csv"),
                train_arr_path=Path(config.get("train_array_path", "artifacts/data_transformation/train.npy")),
                test_arr_path=Path(config.get("test_array_path", "artifacts/data_transformation/test.npy")),
                numerical_features=config.get("numerical_features", NUMERICAL_FEATURES),
                categorical_features=config.get("categorical_features", CATEGORICAL_FEATURES),
                target_column=TARGET_COLUMN,
                create_ratio_features=config.feature_engineering.get("create_ratio_features", True),
                create_payment_features=config.feature_engineering.get("create_payment_features", True),
                create_balance_features=config.feature_engineering.get("create_balance_features", True)
            )

            return data_transformation_config

        except Exception as e:
            raise ConfigurationException(f"Error getting data transformation config: {e}", sys)

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration"""
        try:
            config = self.config.model_trainer

            create_directories([Path(config.get("root_dir", "artifacts/model_trainer"))])

            model_trainer_config = ModelTrainerConfig(
                root_dir=Path(config.get("root_dir", "artifacts/model_trainer")),
                train_data_path=Path("artifacts/data_transformation/train.npy"),
                test_data_path=Path("artifacts/data_transformation/test.npy"),
                model_name="best_model",
                target_column=TARGET_COLUMN,
                model_path=Path(config.get("model_path", "artifacts/model_trainer/model.pkl")),
                best_model_path=Path(config.get("best_model_path", "artifacts/model_trainer/best_model.pkl")),
                metric_file_path=Path(config.get("metric_file_path", "artifacts/model_trainer/metrics.yaml")),
                algorithms=config.get("algorithms", {}),
                cv_folds=config.cross_validation.get("cv_folds", 5),
                scoring=config.cross_validation.get("scoring", "roc_auc"),
                random_state=RANDOM_STATE
            )

            return model_trainer_config

        except Exception as e:
            raise ConfigurationException(f"Error getting model trainer config: {e}", sys)

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """Get model evaluation configuration"""
        try:
            config = self.config.get("model_evaluation", {})

            create_directories([Path(config.get("root_dir", "artifacts/model_evaluation"))])

            model_evaluation_config = ModelEvaluationConfig(
                root_dir=Path(config.get("root_dir", "artifacts/model_evaluation")),
                test_data_path=Path("artifacts/data_transformation/test.npy"),
                model_path=Path("artifacts/model_trainer/best_model.pkl"),
                all_params=self.config.model_trainer.algorithms,
                metric_file_name=Path(config.get("metric_file_name", "artifacts/model_evaluation/metrics.yaml")),
                target_column=TARGET_COLUMN,
                mlflow_uri=self.config.mlflow.get("tracking_uri", "")
            )

            return model_evaluation_config

        except Exception as e:
            raise ConfigurationException(f"Error getting model evaluation config: {e}", sys)

    def get_explainer_config(self) -> ExplainerConfig:
        """Get explainer configuration"""
        try:
            config = self.config.explainer

            create_directories([Path(config.get("root_dir", "artifacts/explainer"))])

            explainer_config = ExplainerConfig(
                root_dir=Path(config.get("root_dir", "artifacts/explainer")),
                model_path=Path("artifacts/model_trainer/best_model.pkl"),
                test_data_path=Path("artifacts/data_transformation/test.npy"),
                explainer_path=Path(config.get("explainer_path", "artifacts/explainer/shap_explainer.pkl")),
                feature_names_path=Path(config.get("feature_names_path", "artifacts/explainer/feature_names.pkl")),
                sample_size=config.get("sample_size", 1000)
            )

            return explainer_config

        except Exception as e:
            raise ConfigurationException(f"Error getting explainer config: {e}", sys)
