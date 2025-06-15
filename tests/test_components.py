"""
Unit tests for Credit Default Prediction components
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.credit_default.components.data_ingestion import DataIngestion
from src.credit_default.components.data_validation import DataValidation
from src.credit_default.components.data_transformation import DataTransformation
from src.credit_default.entity import (
    DataIngestionConfig, 
    DataValidationConfig, 
    DataTransformationConfig
)


class TestDataIngestion:
    """Test cases for Data Ingestion component"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = DataIngestionConfig(
            root_dir=Path(self.temp_dir) / "data_ingestion",
            source_url="https://example.com/test.csv",
            local_data_file=Path(self.temp_dir) / "test.csv",
            unzip_dir=Path(self.temp_dir) / "unzip",
            raw_data_path=Path(self.temp_dir) / "raw.csv",
            train_data_path=Path(self.temp_dir) / "train.csv",
            test_data_path=Path(self.temp_dir) / "test.csv"
        )

    def test_initialization(self):
        """Test DataIngestion initialization"""
        data_ingestion = DataIngestion(self.config)
        assert data_ingestion.config == self.config

    def test_split_data(self):
        """Test data splitting functionality"""
        # Create sample data
        data = {
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'default.payment.next.month': np.random.choice([0, 1], 100)
        }
        df = pd.DataFrame(data)

        data_ingestion = DataIngestion(self.config)
        train_df, test_df = data_ingestion.split_data(df)

        # Assertions
        assert len(train_df) + len(test_df) == len(df)
        assert len(train_df) == int(len(df) * (1 - self.config.test_size))
        assert 'default.payment.next.month' in train_df.columns
        assert 'default.payment.next.month' in test_df.columns


class TestDataValidation:
    """Test cases for Data Validation component"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        # Mock schema
        self.mock_schema = {
            'columns': {
                'LIMIT_BAL': {'type': 'int64', 'min_value': 10000},
                'AGE': {'type': 'int64', 'min_value': 18, 'max_value': 100}
            },
            'target': {
                'column': 'default.payment.next.month',
                'type': 'int64',
                'allowed_values': [0, 1]
            }
        }

        self.config = DataValidationConfig(
            root_dir=Path(self.temp_dir) / "validation",
            status_file=Path(self.temp_dir) / "status.txt",
            unzip_data_dir=Path(self.temp_dir),
            all_schema=self.mock_schema,
            drift_report_file_path=Path(self.temp_dir) / "drift.yaml"
        )

    def test_initialization(self):
        """Test DataValidation initialization"""
        data_validation = DataValidation(self.config)
        assert data_validation.config == self.config

    def test_validate_columns(self):
        """Test column validation"""
        # Create test dataframe with correct columns
        df = pd.DataFrame({
            'LIMIT_BAL': [50000, 60000],
            'AGE': [25, 30],
            'default.payment.next.month': [0, 1]
        })

        data_validation = DataValidation(self.config)
        result = data_validation.validate_all_columns(df)
        assert result == True

        # Test with missing column
        df_missing = pd.DataFrame({
            'LIMIT_BAL': [50000, 60000],
            'default.payment.next.month': [0, 1]
        })

        result_missing = data_validation.validate_all_columns(df_missing)
        assert result_missing == False


class TestDataTransformation:
    """Test cases for Data Transformation component"""

    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

        self.config = DataTransformationConfig(
            root_dir=Path(self.temp_dir) / "transformation",
            data_path=Path(self.temp_dir) / "data.csv",
            preprocessor_obj_file_path=Path(self.temp_dir) / "preprocessor.pkl",
            train_data_path=Path(self.temp_dir) / "train.csv",
            test_data_path=Path(self.temp_dir) / "test.csv",
            train_arr_path=Path(self.temp_dir) / "train.npy",
            test_arr_path=Path(self.temp_dir) / "test.npy",
            numerical_features=['LIMIT_BAL', 'AGE'],
            categorical_features=['SEX', 'EDUCATION'],
            target_column='default.payment.next.month'
        )

    def test_initialization(self):
        """Test DataTransformation initialization"""
        data_transformation = DataTransformation(self.config)
        assert data_transformation.config == self.config

    def test_get_data_transformer_object(self):
        """Test data transformer creation"""
        data_transformation = DataTransformation(self.config)
        preprocessor = data_transformation.get_data_transformer_object()

        # Check if preprocessor is created
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit_transform')

    def test_feature_engineering(self):
        """Test feature engineering"""
        # Create sample data
        data = {
            'LIMIT_BAL': [50000, 60000],
            'PAY_AMT1': [1000, 1500],
            'PAY_AMT2': [900, 1400],
            'PAY_AMT3': [800, 1300],
            'PAY_AMT4': [700, 1200],
            'PAY_AMT5': [600, 1100],
            'PAY_AMT6': [500, 1000],
            'BILL_AMT1': [5000, 6000],
            'BILL_AMT2': [4500, 5500],
            'BILL_AMT3': [4000, 5000],
            'BILL_AMT4': [3500, 4500],
            'BILL_AMT5': [3000, 4000],
            'BILL_AMT6': [2500, 3500],
            'PAY_0': [1, 0],
            'PAY_2': [0, 1],
            'PAY_3': [0, 0],
            'PAY_4': [1, 0],
            'PAY_5': [0, 1],
            'PAY_6': [0, 0]
        }
        df = pd.DataFrame(data)

        data_transformation = DataTransformation(self.config)
        df_engineered = data_transformation.create_engineered_features(df)

        # Check if new features are created
        expected_features = [
            'PAY_AMT_TO_LIMIT_RATIO',
            'BILL_AMT_TO_LIMIT_RATIO', 
            'PAY_TO_BILL_RATIO',
            'AVG_PAY_STATUS',
            'MAX_PAY_DELAY',
            'MONTHS_WITH_DELAY'
        ]

        for feature in expected_features:
            assert feature in df_engineered.columns


@pytest.fixture
def sample_credit_data():
    """Fixture for sample credit data"""
    np.random.seed(42)
    n_samples = 100

    data = {
        'LIMIT_BAL': np.random.randint(10000, 500000, n_samples),
        'SEX': np.random.choice([1, 2], n_samples),
        'EDUCATION': np.random.choice([1, 2, 3, 4], n_samples),
        'MARRIAGE': np.random.choice([1, 2, 3], n_samples),
        'AGE': np.random.randint(18, 70, n_samples),
        'PAY_0': np.random.randint(-1, 3, n_samples),
        'PAY_2': np.random.randint(-1, 3, n_samples),
        'PAY_3': np.random.randint(-1, 3, n_samples),
        'PAY_4': np.random.randint(-1, 3, n_samples),
        'PAY_5': np.random.randint(-1, 3, n_samples),
        'PAY_6': np.random.randint(-1, 3, n_samples),
        'BILL_AMT1': np.random.randint(0, 100000, n_samples),
        'BILL_AMT2': np.random.randint(0, 100000, n_samples),
        'BILL_AMT3': np.random.randint(0, 100000, n_samples),
        'BILL_AMT4': np.random.randint(0, 100000, n_samples),
        'BILL_AMT5': np.random.randint(0, 100000, n_samples),
        'BILL_AMT6': np.random.randint(0, 100000, n_samples),
        'PAY_AMT1': np.random.randint(0, 50000, n_samples),
        'PAY_AMT2': np.random.randint(0, 50000, n_samples),
        'PAY_AMT3': np.random.randint(0, 50000, n_samples),
        'PAY_AMT4': np.random.randint(0, 50000, n_samples),
        'PAY_AMT5': np.random.randint(0, 50000, n_samples),
        'PAY_AMT6': np.random.randint(0, 50000, n_samples),
        'default.payment.next.month': np.random.choice([0, 1], n_samples)
    }

    return pd.DataFrame(data)


def test_sample_data_structure(sample_credit_data):
    """Test the structure of sample data"""
    assert len(sample_credit_data) == 100
    assert 'default.payment.next.month' in sample_credit_data.columns
    assert sample_credit_data['SEX'].isin([1, 2]).all()
    assert sample_credit_data['AGE'].between(18, 70).all()


if __name__ == "__main__":
    pytest.main([__file__])
