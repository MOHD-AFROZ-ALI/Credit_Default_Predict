"""
Data Validation Component for Credit Default Prediction
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import yaml

from credit_default.entity import DataValidationConfig
from credit_default.exception import DataValidationException
from credit_default.logger import logger


class DataValidation:
    """Data validation component"""

    def __init__(self, config: DataValidationConfig):
        """
        Initialize data validation

        Args:
            config: Data validation configuration
        """
        self.config = config

    def validate_all_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate if all required columns are present

        Args:
            df: Input dataframe

        Returns:
            bool: Validation status
        """
        try:
            validation_status = True
            all_cols = list(df.columns)

            # Get expected columns from schema
            all_schema = self.config.all_schema.columns
            required_columns = list(all_schema.keys())

            # Add target column
            target_col = self.config.all_schema.target.column
            if target_col not in required_columns:
                required_columns.append(target_col)

            # Check for missing columns
            missing_columns = []
            for col in required_columns:
                if col not in all_cols:
                    missing_columns.append(col)
                    validation_status = False

            if missing_columns:
                logger.error(f"Missing columns: {missing_columns}")

            # Check for extra columns
            extra_columns = []
            for col in all_cols:
                if col not in required_columns:
                    extra_columns.append(col)

            if extra_columns:
                logger.warning(f"Extra columns found: {extra_columns}")

            # Log validation results
            logger.info(f"Expected columns: {len(required_columns)}")
            logger.info(f"Actual columns: {len(all_cols)}")
            logger.info(f"Column validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise DataValidationException(f"Error validating columns: {e}", sys)

    def validate_data_types(self, df: pd.DataFrame) -> bool:
        """
        Validate data types of columns

        Args:
            df: Input dataframe

        Returns:
            bool: Validation status
        """
        try:
            validation_status = True
            schema_columns = self.config.all_schema.columns

            type_errors = []

            for column, schema_config in schema_columns.items():
                if column in df.columns:
                    expected_type = schema_config.get('type', '')
                    actual_dtype = str(df[column].dtype)

                    # Check type compatibility
                    if expected_type == 'int64' and not pd.api.types.is_integer_dtype(df[column]):
                        type_errors.append(f"{column}: expected {expected_type}, got {actual_dtype}")
                        validation_status = False
                    elif expected_type == 'float64' and not pd.api.types.is_numeric_dtype(df[column]):
                        type_errors.append(f"{column}: expected {expected_type}, got {actual_dtype}")
                        validation_status = False

            if type_errors:
                logger.error(f"Data type validation errors: {type_errors}")
            else:
                logger.info("Data type validation passed")

            return validation_status

        except Exception as e:
            raise DataValidationException(f"Error validating data types: {e}", sys)

    def validate_data_ranges(self, df: pd.DataFrame) -> bool:
        """
        Validate data ranges and allowed values

        Args:
            df: Input dataframe

        Returns:
            bool: Validation status
        """
        try:
            validation_status = True
            schema_columns = self.config.all_schema.columns

            range_errors = []

            for column, schema_config in schema_columns.items():
                if column in df.columns:
                    series = df[column]

                    # Check minimum value
                    if 'min_value' in schema_config:
                        min_val = schema_config['min_value']
                        if series.min() < min_val:
                            range_errors.append(f"{column}: minimum value {series.min()} < expected {min_val}")
                            validation_status = False

                    # Check maximum value
                    if 'max_value' in schema_config:
                        max_val = schema_config['max_value']
                        if series.max() > max_val:
                            range_errors.append(f"{column}: maximum value {series.max()} > expected {max_val}")
                            validation_status = False

                    # Check allowed values
                    if 'allowed_values' in schema_config:
                        allowed_vals = schema_config['allowed_values']
                        invalid_values = set(series.unique()) - set(allowed_vals)
                        if invalid_values:
                            range_errors.append(f"{column}: invalid values {invalid_values}, allowed: {allowed_vals}")
                            validation_status = False

            if range_errors:
                logger.error(f"Data range validation errors: {range_errors}")
            else:
                logger.info("Data range validation passed")

            return validation_status

        except Exception as e:
            raise DataValidationException(f"Error validating data ranges: {e}", sys)

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality metrics

        Args:
            df: Input dataframe

        Returns:
            Dict[str, Any]: Data quality metrics
        """
        try:
            quality_metrics = {}

            # Missing values
            missing_count = df.isnull().sum().sum()
            missing_percentage = (missing_count / (df.shape[0] * df.shape[1])) * 100
            quality_metrics['missing_count'] = int(missing_count)
            quality_metrics['missing_percentage'] = float(missing_percentage)

            # Duplicate rows
            duplicate_count = df.duplicated().sum()
            duplicate_percentage = (duplicate_count / df.shape[0]) * 100
            quality_metrics['duplicate_count'] = int(duplicate_count)
            quality_metrics['duplicate_percentage'] = float(duplicate_percentage)

            # Data shape
            quality_metrics['rows'] = df.shape[0]
            quality_metrics['columns'] = df.shape[1]

            # Target distribution
            target_col = self.config.all_schema.target.column
            if target_col in df.columns:
                target_dist = df[target_col].value_counts().to_dict()
                quality_metrics['target_distribution'] = {str(k): int(v) for k, v in target_dist.items()}

            logger.info(f"Data quality metrics: {quality_metrics}")

            return quality_metrics

        except Exception as e:
            raise DataValidationException(f"Error checking data quality: {e}", sys)

    def detect_data_drift(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift (basic implementation)

        Args:
            df: Input dataframe

        Returns:
            Dict[str, Any]: Drift report
        """
        try:
            drift_report = {}

            # For now, implement basic statistical checks
            numerical_cols = df.select_dtypes(include=[np.number]).columns

            for col in numerical_cols:
                if col in df.columns:
                    stats = {
                        'mean': float(df[col].mean()),
                        'std': float(df[col].std()),
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'median': float(df[col].median())
                    }
                    drift_report[col] = stats

            # Save drift report
            with open(self.config.drift_report_file_path, 'w') as f:
                yaml.dump(drift_report, f)

            logger.info(f"Drift report saved to {self.config.drift_report_file_path}")

            return drift_report

        except Exception as e:
            raise DataValidationException(f"Error detecting data drift: {e}", sys)

    def initiate_data_validation(self) -> str:
        """
        Initiate complete data validation process

        Returns:
            str: Validation status
        """
        try:
            logger.info("Starting data validation process")

            # Load the raw data
            data_file = self.config.unzip_data_dir / "credit_default.csv"
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_file}")

            df = pd.read_csv(data_file)
            logger.info(f"Loaded data shape: {df.shape}")

            # Perform validations
            validation_results = []

            # Column validation
            column_validation = self.validate_all_columns(df)
            validation_results.append(column_validation)

            # Data type validation
            type_validation = self.validate_data_types(df)
            validation_results.append(type_validation)

            # Range validation
            range_validation = self.validate_data_ranges(df)
            validation_results.append(range_validation)

            # Data quality check
            quality_metrics = self.check_data_quality(df)

            # Data drift detection
            drift_report = self.detect_data_drift(df)

            # Overall validation status
            overall_validation = all(validation_results)

            # Check quality thresholds
            quality_config = self.config.all_schema.get('data_quality', {})
            max_missing = quality_config.get('max_missing_percentage', 5.0)
            max_duplicate = quality_config.get('max_duplicate_percentage', 1.0)

            if quality_metrics['missing_percentage'] > max_missing:
                logger.warning(f"Missing percentage {quality_metrics['missing_percentage']:.2f}% exceeds threshold {max_missing}%")
                overall_validation = False

            if quality_metrics['duplicate_percentage'] > max_duplicate:
                logger.warning(f"Duplicate percentage {quality_metrics['duplicate_percentage']:.2f}% exceeds threshold {max_duplicate}%")
                overall_validation = False

            # Write validation status
            status = "Validation Passed" if overall_validation else "Validation Failed"

            with open(self.config.status_file, 'w') as f:
                f.write(f"Validation Status: {status}\n")
                f.write(f"Column Validation: {'Passed' if column_validation else 'Failed'}\n")
                f.write(f"Type Validation: {'Passed' if type_validation else 'Failed'}\n")
                f.write(f"Range Validation: {'Passed' if range_validation else 'Failed'}\n")
                f.write(f"Missing Percentage: {quality_metrics['missing_percentage']:.2f}%\n")
                f.write(f"Duplicate Percentage: {quality_metrics['duplicate_percentage']:.2f}%\n")

            logger.info(f"Data validation completed. Status: {status}")
            logger.info(f"Validation report saved to {self.config.status_file}")

            return status

        except Exception as e:
            raise DataValidationException(f"Error in data validation process: {e}", sys)
