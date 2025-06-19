"""
Data Transformation Component for Credit Default Prediction
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from credit_default.entity import DataTransformationConfig
from credit_default.exception import DataTransformationException
from credit_default.logger import logger
from credit_default.utils import save_object


class DataTransformation:
    """Data transformation component"""

    def __init__(self, config: DataTransformationConfig):
        """
        Initialize data transformation

        Args:
            config: Data transformation configuration
        """
        self.config = config

    def get_data_transformer_object(self):
        """
        Create data transformation pipeline

        Returns:
            ColumnTransformer: Data transformation pipeline
        """
        try:
            # Define feature columns
            numerical_columns = self.config.numerical_features
            categorical_columns = self.config.categorical_features

            # Numerical pipeline
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler())
            ])

            # Categorical pipeline
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
            ])

            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise DataTransformationException(f"Error creating data transformer: {e}", sys)

    def create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features

        Args:
            df: Input dataframe

        Returns:
            pd.DataFrame: Dataframe with engineered features
        """
        try:
            logger.info("Creating engineered features")
            df_eng = df.copy()

            if self.config.create_ratio_features:
                # Payment to limit ratio
                df_eng['PAY_AMT_TO_LIMIT_RATIO'] = (
                    (df_eng['PAY_AMT1'] + df_eng['PAY_AMT2'] + df_eng['PAY_AMT3'] + 
                     df_eng['PAY_AMT4'] + df_eng['PAY_AMT5'] + df_eng['PAY_AMT6']) / 
                    df_eng['LIMIT_BAL']
                ).fillna(0)

                # Bill to limit ratio
                df_eng['BILL_AMT_TO_LIMIT_RATIO'] = (
                    df_eng['BILL_AMT1'] / df_eng['LIMIT_BAL']
                ).fillna(0)

                # Payment to bill ratio
                df_eng['PAY_TO_BILL_RATIO'] = (
                    df_eng['PAY_AMT1'] / (df_eng['BILL_AMT1'] + 1)
                ).fillna(0)

            if self.config.create_payment_features:
                # Average payment status
                pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
                df_eng['AVG_PAY_STATUS'] = df_eng[pay_cols].mean(axis=1)

                # Maximum payment delay
                df_eng['MAX_PAY_DELAY'] = df_eng[pay_cols].max(axis=1)

                # Number of months with payment delay
                df_eng['MONTHS_WITH_DELAY'] = (df_eng[pay_cols] > 0).sum(axis=1)

            if self.config.create_balance_features:
                # Total bill amount (last 6 months)
                bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
                df_eng['TOTAL_BILL_AMT'] = df_eng[bill_cols].sum(axis=1)

                # Average bill amount
                df_eng['AVG_BILL_AMT'] = df_eng[bill_cols].mean(axis=1)

                # Bill amount trend (difference between recent and older bills)
                df_eng['BILL_AMT_TREND'] = df_eng['BILL_AMT1'] - df_eng['BILL_AMT6']

                # Total payment amount
                pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
                df_eng['TOTAL_PAY_AMT'] = df_eng[pay_amt_cols].sum(axis=1)

                # Average payment amount
                df_eng['AVG_PAY_AMT'] = df_eng[pay_amt_cols].mean(axis=1)

            # Update numerical features list to include engineered features
            new_features = []
            if self.config.create_ratio_features:
                new_features.extend(['PAY_AMT_TO_LIMIT_RATIO', 'BILL_AMT_TO_LIMIT_RATIO', 'PAY_TO_BILL_RATIO'])
            if self.config.create_payment_features:
                new_features.extend(['AVG_PAY_STATUS', 'MAX_PAY_DELAY', 'MONTHS_WITH_DELAY'])
            if self.config.create_balance_features:
                new_features.extend(['TOTAL_BILL_AMT', 'AVG_BILL_AMT', 'BILL_AMT_TREND', 'TOTAL_PAY_AMT', 'AVG_PAY_AMT'])

            # Add new features to numerical features
            self.config.numerical_features.extend(new_features)

            logger.info(f"Created {len(new_features)} engineered features")
            logger.info(f"New features: {new_features}")

            return df_eng

        except Exception as e:
            raise DataTransformationException(f"Error creating engineered features: {e}", sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
        Initiate data transformation process

        Args:
            train_path: Path to training data
            test_path: Path to test data

        Returns:
            tuple: (train_arr, test_arr, preprocessor_path)
        """
        try:
            logger.info("Starting data transformation process")

            # Read train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Train data shape: {train_df.shape}")
            logger.info(f"Test data shape: {test_df.shape}")

            # Create engineered features
            train_df = self.create_engineered_features(train_df)
            test_df = self.create_engineered_features(test_df)

            for df in [train_df, test_df]:
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                df.fillna(0, inplace=True)

            logger.info(f"Train data shape after feature engineering: {train_df.shape}")
            logger.info(f"Test data shape after feature engineering: {test_df.shape}")

            # Separate features and target
            target_column_name = self.config.target_column

            # Training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            logger.info("Applying preprocessing object on training and testing datasets")

            # Transform the data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine features and target
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]
            logger.info(f"Transformed train array shape: {train_arr.shape}")
            logger.info(f"Transformed test array shape: {test_arr.shape}")

            # Save preprocessing object
            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save arrays
            np.save(self.config.train_arr_path, train_arr)
            np.save(self.config.test_arr_path, test_arr)

            logger.info(f"Saved train array to {self.config.train_arr_path}")
            logger.info(f"Saved test array to {self.config.test_arr_path}")
            logger.info(f"Saved preprocessing object to {self.config.preprocessor_obj_file_path}")

            logger.info("Data transformation completed successfully")

            return (
                train_arr,
                test_arr,
                str(self.config.preprocessor_obj_file_path)
            )

        except Exception as e:
            raise DataTransformationException(f"Error in data transformation process: {e}", sys)

    def get_feature_names(self, preprocessor) -> list:
        """
        Get feature names after transformation

        Args:
            preprocessor: Fitted preprocessor

        Returns:
            list: Feature names
        """
        try:
            feature_names = []

            # Numerical feature names
            num_features = self.config.numerical_features
            feature_names.extend(num_features)

            # Categorical feature names (after one-hot encoding)
            cat_features = self.config.categorical_features
            if hasattr(preprocessor.named_transformers_['cat_pipeline']['encoder'], 'get_feature_names_out'):
                cat_feature_names = preprocessor.named_transformers_['cat_pipeline']['encoder'].get_feature_names_out(cat_features)
                feature_names.extend(cat_feature_names)
            else:
                # Fallback for older sklearn versions
                feature_names.extend([f"{cat}_{i}" for cat in cat_features for i in range(2)])  # Assuming binary encoding

            logger.info(f"Generated {len(feature_names)} feature names")

            return feature_names

        except Exception as e:
            logger.warning(f"Error generating feature names: {e}")
            # Return generic names
            return [f"feature_{i}" for i in range(len(self.config.numerical_features) + len(self.config.categorical_features))]
