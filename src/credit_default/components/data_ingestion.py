"""
Data Ingestion Component for Credit Default Prediction
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from urllib.parse import urlparse
import requests
import zipfile

from credit_default.entity import DataIngestionConfig
from credit_default.exception import DataIngestionException
from credit_default.logger import logger


class DataIngestion:
    """Data ingestion component"""

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize data ingestion

        Args:
            config: Data ingestion configuration
        """
        self.config = config

    def download_file(self) -> Path:
        """
        Download file from URL

        Returns:
            Path: Path to downloaded file
        """
        try:
            dataset_url = self.config.source_url
            download_dir = self.config.local_data_file.parent
            download_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading data from {dataset_url}")

            # Download the file
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()

            with open(self.config.local_data_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Downloaded data to {self.config.local_data_file}")
            return self.config.local_data_file

        except Exception as e:
            raise DataIngestionException(f"Error downloading file: {e}", sys)

    def extract_zip_file(self) -> Path:
        """
        Extract zip file if needed

        Returns:
            Path: Path to extracted directory
        """
        try:
            unzip_path = self.config.unzip_dir
            unzip_path.mkdir(parents=True, exist_ok=True)

            # If it's a zip file, extract it
            if str(self.config.local_data_file).endswith('.zip'):
                with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                    zip_ref.extractall(unzip_path)
                logger.info(f"Extracted {self.config.local_data_file} to {unzip_path}")

            return unzip_path

        except Exception as e:
            raise DataIngestionException(f"Error extracting file: {e}", sys)

    def load_and_clean_data(self) -> pd.DataFrame:
        """
        Load and clean the dataset

        Returns:
            pd.DataFrame: Cleaned dataset
        """
        try:
            # Load the Excel file
            if self.config.local_data_file.exists():
                logger.info(f"Loading data from {self.config.local_data_file}")

                # Try to read as Excel file first
                try:
                    df = pd.read_excel(self.config.local_data_file, sheet_name='Data', header=1)
                except Exception:
                    # If Excel fails, try CSV
                    df = pd.read_csv(self.config.local_data_file)

                logger.info(f"Data shape: {df.shape}")
                logger.info(f"Columns: {list(df.columns)}")

                # Drop ID column if exists
                if 'ID' in df.columns:
                    df = df.drop('ID', axis=1)

                # Rename target column for consistency
                target_candidates = ['default payment next month', 'default.payment.next.month', 'Y']
                for candidate in target_candidates:
                    if candidate in df.columns:
                        df = df.rename(columns={candidate: 'default.payment.next.month'})
                        break

                # Handle missing values
                df = df.dropna()

                # Remove duplicates
                df = df.drop_duplicates()

                logger.info(f"Cleaned data shape: {df.shape}")

                # Save as CSV for easier processing
                csv_path = self.config.raw_data_path
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved cleaned data to {csv_path}")

                return df
            else:
                raise FileNotFoundError(f"Data file not found: {self.config.local_data_file}")

        except Exception as e:
            raise DataIngestionException(f"Error loading and cleaning data: {e}", sys)

    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train and test sets

        Args:
            df: Input dataframe

        Returns:
            tuple: (train_df, test_df)
        """
        try:
            logger.info("Splitting data into train and test sets")

            # Prepare features and target
            target_col = 'default.payment.next.month'
            if target_col not in df.columns:
                raise ValueError(f"Target column '{target_col}' not found in dataset")

            X = df.drop(target_col, axis=1)
            y = df[target_col]

            # Split the data
            if self.config.stratify:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state,
                    stratify=y
                )
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y,
                    test_size=self.config.test_size,
                    random_state=self.config.random_state
                )

            # Combine features and target
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            logger.info(f"Train set shape: {train_df.shape}")
            logger.info(f"Test set shape: {test_df.shape}")

            # Save train and test sets
            self.config.train_data_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.test_data_path.parent.mkdir(parents=True, exist_ok=True)

            train_df.to_csv(self.config.train_data_path, index=False)
            test_df.to_csv(self.config.test_data_path, index=False)

            logger.info(f"Saved train data to {self.config.train_data_path}")
            logger.info(f"Saved test data to {self.config.test_data_path}")

            return train_df, test_df

        except Exception as e:
            raise DataIngestionException(f"Error splitting data: {e}", sys)

    def initiate_data_ingestion(self) -> tuple:
        """
        Initiate the complete data ingestion process

        Returns:
            tuple: (train_data_path, test_data_path)
        """
        try:
            logger.info("Starting data ingestion process")

            # Download file
            self.download_file()

            # Extract if needed
            self.extract_zip_file()

            # Load and clean data
            df = self.load_and_clean_data()

            # Split data
            train_df, test_df = self.split_data(df)

            logger.info("Data ingestion completed successfully")

            return (
                str(self.config.train_data_path),
                str(self.config.test_data_path)
            )

        except Exception as e:
            raise DataIngestionException(f"Error in data ingestion process: {e}", sys)
