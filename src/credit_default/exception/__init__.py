"""
Custom exception classes for Credit Default Prediction
"""
import sys
from typing import Optional


class CreditDefaultException(Exception):
    """Base exception class for Credit Default Prediction"""

    def __init__(self, error_message: str, error_detail: Optional[sys] = None):
        super().__init__(error_message)
        self.error_message = error_message

        if error_detail:
            _, _, exc_tb = error_detail.exc_info()
            if exc_tb:
                self.line_number = exc_tb.tb_lineno
                self.file_name = exc_tb.tb_frame.f_code.co_filename
            else:
                self.line_number = None
                self.file_name = None
        else:
            self.line_number = None
            self.file_name = None

    def __str__(self):
        if self.line_number and self.file_name:
            return (f"Error occurred in python script [{self.file_name}] "
                   f"at line number [{self.line_number}]: {self.error_message}")
        return f"Error: {self.error_message}"


class DataIngestionException(CreditDefaultException):
    """Exception raised during data ingestion"""
    pass


class DataValidationException(CreditDefaultException):
    """Exception raised during data validation"""
    pass


class DataTransformationException(CreditDefaultException):
    """Exception raised during data transformation"""
    pass


class ModelTrainerException(CreditDefaultException):
    """Exception raised during model training"""
    pass


class ModelEvaluationException(CreditDefaultException):
    """Exception raised during model evaluation"""
    pass


class PredictionException(CreditDefaultException):
    """Exception raised during prediction"""
    pass


class ConfigurationException(CreditDefaultException):
    """Exception raised for configuration errors"""
    pass


class ExplainerException(CreditDefaultException):
    """Exception raised during explainer operations"""
    pass
