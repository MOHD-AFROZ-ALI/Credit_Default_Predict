"""
Training Pipeline for Credit Default Prediction
"""
import sys
from pathlib import Path

from credit_default.configuration import ConfigurationManager
from credit_default.components.data_ingestion import DataIngestion
from credit_default.components.data_validation import DataValidation
from credit_default.components.data_transformation import DataTransformation
from credit_default.components.model_trainer import ModelTrainer
from credit_default.components.model_explainer import ModelExplainer
from credit_default.exception import CreditDefaultException
from credit_default.logger import logger


class TrainingPipeline:
    """Complete training pipeline"""

    def __init__(self):
        """Initialize training pipeline"""
        self.config_manager = ConfigurationManager()

    def start_data_ingestion(self):
        """Data ingestion stage"""
        try:
            logger.info("Starting data ingestion stage")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
            logger.info("Data ingestion stage completed")
            return train_data_path, test_data_path
        except Exception as e:
            raise CreditDefaultException(f"Error in data ingestion stage: {e}", sys)

    def start_data_validation(self):
        """Data validation stage"""
        try:
            logger.info("Starting data validation stage")
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.initiate_data_validation()
            logger.info("Data validation stage completed")
            return validation_status
        except Exception as e:
            raise CreditDefaultException(f"Error in data validation stage: {e}", sys)

    def start_data_transformation(self, train_data_path: str, test_data_path: str):
        """Data transformation stage"""
        try:
            logger.info("Starting data transformation stage")
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
                train_path=train_data_path,
                test_path=test_data_path
            )
            logger.info("Data transformation stage completed")
            return train_arr, test_arr, preprocessor_path
        except Exception as e:
            raise CreditDefaultException(f"Error in data transformation stage: {e}", sys)

    def start_model_trainer(self, train_array, test_array):
        """Model training stage"""
        try:
            logger.info("Starting model training stage")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            best_model_score = model_trainer.initiate_model_trainer(
                train_array=train_array,
                test_array=test_array
            )
            logger.info("Model training stage completed")
            return best_model_score
        except Exception as e:
            raise CreditDefaultException(f"Error in model training stage: {e}", sys)

    def start_model_explanation(self):
        """Model explanation stage"""
        try:
            logger.info("Starting model explanation stage")
            explainer_config = self.config_manager.get_explainer_config()
            model_explainer = ModelExplainer(config=explainer_config)
            explanation_data = model_explainer.initiate_model_explanation()
            logger.info("Model explanation stage completed")
            return explanation_data
        except Exception as e:
            logger.warning(f"Error in model explanation stage: {e}")
            return None

    def run_pipeline(self):
        """Run the complete training pipeline"""
        try:
            logger.info("=" * 50)
            logger.info("STARTING CREDIT DEFAULT PREDICTION TRAINING PIPELINE")
            logger.info("=" * 50)

            # Stage 1: Data Ingestion
            train_data_path, test_data_path = self.start_data_ingestion()

            # Stage 2: Data Validation
            validation_status = self.start_data_validation()
            if "Validation Failed" in validation_status:
                logger.error("Data validation failed. Pipeline stopped.")
                return False

            # Stage 3: Data Transformation
            train_arr, test_arr, preprocessor_path = self.start_data_transformation(
                train_data_path, test_data_path
            )

            # Stage 4: Model Training
            best_model_score = self.start_model_trainer(train_arr, test_arr)

            # Stage 5: Model Explanation
            explanation_data = self.start_model_explanation()

            logger.info("=" * 50)
            logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Best Model Score: {best_model_score:.4f}")
            logger.info("=" * 50)

            return True

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise CreditDefaultException(f"Error in training pipeline: {e}", sys)


def main():
    """Main function to run training pipeline"""
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise e


if __name__ == "__main__":
    main()
