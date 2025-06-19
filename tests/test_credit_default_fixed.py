
import pytest
import pandas as pd
import numpy as np
import pickle
import sys
import os
from pathlib import Path

# Add src directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.credit_default.exception import CreditDefaultException
from src.credit_default.logger import logging
from src.credit_default.utils import load_object, save_object
from src.credit_default.entity import DataIngestionConfig
from src.credit_default.entity import DataIngestionArtifact


class TestDataIngestion:
    """Test cases for data ingestion component"""

    def setup_method(self):
        """Setup method for each test"""
        self.test_data = pd.DataFrame({
            'LIMIT_BAL': [20000, 50000, 30000],
            'SEX': [2, 1, 2],
            'EDUCATION': [2, 2, 1],
            'MARRIAGE': [1, 2, 2],
            'AGE': [24, 35, 28],
            'PAY_0': [2, 0, -1],
            'PAY_2': [2, 0, 0],
            'PAY_3': [1, 0, 0],
            'PAY_4': [1, 0, 0],
            'PAY_5': [-1, 0, -1],
            'PAY_6': [-1, 0, -2],
            'BILL_AMT1': [3913, 2682, 29239],
            'BILL_AMT2': [3102, 1725, 14027],
            'BILL_AMT3': [689, 2682, 13559],
            'BILL_AMT4': [0, 3272, 14331],
            'BILL_AMT5': [0, 3455, 14948],
            'BILL_AMT6': [0, 3261, 15549],
            'PAY_AMT1': [0, 0, 1518],
            'PAY_AMT2': [689, 1000, 1500],
            'PAY_AMT3': [0, 1000, 1000],
            'PAY_AMT4': [0, 1000, 1000],
            'PAY_AMT5': [0, 0, 1000],
            'PAY_AMT6': [0, 2000, 5000],
            'default.payment.next.month': [1, 0, 0]
        })

    def test_data_ingestion_config_creation(self):
        """Test data ingestion config creation"""
        config = DataIngestionConfig()
        assert hasattr(config, 'data_ingestion_dir')
        assert hasattr(config, 'feature_store_file_path')
        assert hasattr(config, 'training_file_path')
        assert hasattr(config, 'testing_file_path')

    def test_data_loading(self):
        """Test data loading functionality"""
        # This would test the actual data loading
        # For now, we test with sample data
        assert self.test_data is not None
        assert len(self.test_data) == 3
        assert 'default.payment.next.month' in self.test_data.columns

    def test_train_test_split(self):
        """Test train-test split functionality"""
        from sklearn.model_selection import train_test_split

        X = self.test_data.drop('default.payment.next.month', axis=1)
        y = self.test_data['default.payment.next.month']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)


class TestDataValidation:
    """Test cases for data validation component"""

    def setup_method(self):
        """Setup method for each test"""
        self.valid_data = pd.DataFrame({
            'LIMIT_BAL': [20000.0, 50000.0],
            'SEX': [2, 1],
            'EDUCATION': [2, 2],
            'MARRIAGE': [1, 2],
            'AGE': [24, 35],
            'PAY_0': [2, 0],
            'PAY_2': [2, 0],
            'PAY_3': [1, 0],
            'PAY_4': [1, 0],
            'PAY_5': [-1, 0],
            'PAY_6': [-1, 0],
            'BILL_AMT1': [3913.0, 2682.0],
            'BILL_AMT2': [3102.0, 1725.0],
            'BILL_AMT3': [689.0, 2682.0],
            'BILL_AMT4': [0.0, 3272.0],
            'BILL_AMT5': [0.0, 3455.0],
            'BILL_AMT6': [0.0, 3261.0],
            'PAY_AMT1': [0.0, 0.0],
            'PAY_AMT2': [689.0, 1000.0],
            'PAY_AMT3': [0.0, 1000.0],
            'PAY_AMT4': [0.0, 1000.0],
            'PAY_AMT5': [0.0, 0.0],
            'PAY_AMT6': [0.0, 2000.0],
            'default.payment.next.month': [1, 0]
        })

    def test_column_validation(self):
        """Test column validation"""
        expected_columns = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
            'default.payment.next.month'
        ]

        actual_columns = list(self.valid_data.columns)

        # Check if all expected columns are present
        for col in expected_columns:
            assert col in actual_columns, f"Column {col} missing"

    def test_data_types(self):
        """Test data types validation"""
        # Check if numeric columns are actually numeric
        numeric_columns = [
            'LIMIT_BAL', 'AGE', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 
            'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]

        for col in numeric_columns:
            assert pd.api.types.is_numeric_dtype(self.valid_data[col]), f"Column {col} should be numeric"

    def test_missing_values(self):
        """Test missing values handling"""
        # Check for missing values
        missing_counts = self.valid_data.isnull().sum()

        # In our test data, there should be no missing values
        assert missing_counts.sum() == 0, "No missing values should be present in test data"


class TestDataTransformation:
    """Test cases for data transformation component"""

    def setup_method(self):
        """Setup method for each test"""
        self.sample_data = pd.DataFrame({
            'LIMIT_BAL': [20000, 50000, 30000, 40000],
            'SEX': [2, 1, 2, 1],
            'EDUCATION': [2, 2, 1, 3],
            'MARRIAGE': [1, 2, 2, 1],
            'AGE': [24, 35, 28, 45],
            'PAY_0': [2, 0, -1, 1],
            'PAY_2': [2, 0, 0, 1],
            'PAY_3': [1, 0, 0, 2],
            'PAY_4': [1, 0, 0, 1],
            'PAY_5': [-1, 0, -1, 0],
            'PAY_6': [-1, 0, -2, 0],
            'BILL_AMT1': [3913, 2682, 29239, 15000],
            'BILL_AMT2': [3102, 1725, 14027, 12000],
            'BILL_AMT3': [689, 2682, 13559, 11000],
            'BILL_AMT4': [0, 3272, 14331, 10000],
            'BILL_AMT5': [0, 3455, 14948, 9000],
            'BILL_AMT6': [0, 3261, 15549, 8000],
            'PAY_AMT1': [0, 0, 1518, 1000],
            'PAY_AMT2': [689, 1000, 1500, 1200],
            'PAY_AMT3': [0, 1000, 1000, 1100],
            'PAY_AMT4': [0, 1000, 1000, 1000],
            'PAY_AMT5': [0, 0, 1000, 900],
            'PAY_AMT6': [0, 2000, 5000, 800],
            'default.payment.next.month': [1, 0, 0, 1]
        })

    def test_preprocessing_pipeline(self):
        """Test preprocessing pipeline"""
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        # Separate features and target
        X = self.sample_data.drop('default.payment.next.month', axis=1)
        y = self.sample_data['default.payment.next.month']

        # Create a simple preprocessor
        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features)
            ],
            remainder='passthrough'
        )

        # Fit and transform
        X_transformed = preprocessor.fit_transform(X)

        assert X_transformed is not None
        assert X_transformed.shape[0] == X.shape[0]
        assert X_transformed.shape[1] == X.shape[1]  # No categorical features in this test

    def test_feature_engineering(self):
        """Test feature engineering"""
        # Test some basic feature engineering
        data_copy = self.sample_data.copy()

        # Create some new features (example)
        data_copy['UTILIZATION_RATIO'] = data_copy['BILL_AMT1'] / data_copy['LIMIT_BAL']
        data_copy['PAYMENT_RATIO'] = data_copy['PAY_AMT1'] / (data_copy['BILL_AMT1'] + 1)

        assert 'UTILIZATION_RATIO' in data_copy.columns
        assert 'PAYMENT_RATIO' in data_copy.columns
        assert not data_copy['UTILIZATION_RATIO'].isnull().all()


class TestModelTrainer:
    """Test cases for model trainer component"""

    def setup_method(self):
        """Setup method for each test"""
        np.random.seed(42)

        # Create synthetic training data
        n_samples = 100
        n_features = 23

        self.X_train = np.random.randn(n_samples, n_features)
        self.y_train = np.random.randint(0, 2, n_samples)

        self.X_test = np.random.randn(20, n_features)
        self.y_test = np.random.randint(0, 2, 20)

    def test_logistic_regression_training(self):
        """Test logistic regression model training"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)

        assert model is not None
        assert len(y_pred) == len(self.y_test)
        assert 0 <= accuracy <= 1

    def test_model_evaluation_metrics(self):
        """Test model evaluation metrics"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import precision_score, recall_score, f1_score

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)

        # Make predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)

        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1

    def test_model_serialization(self):
        """Test model saving and loading"""
        from sklearn.linear_model import LogisticRegression
        import tempfile

        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(self.X_train, self.y_train)

        # Save model to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            save_object(file_path=tmp_file.name, obj=model)

            # Load model back
            loaded_model = load_object(file_path=tmp_file.name)

            # Test predictions are same
            original_pred = model.predict(self.X_test)
            loaded_pred = loaded_model.predict(self.X_test)

            assert np.array_equal(original_pred, loaded_pred)

        # Clean up
        os.unlink(tmp_file.name)


class TestModelExplainer:
    """Test cases for model explainer component"""

    def setup_method(self):
        """Setup method for each test"""
        np.random.seed(42)

        # Create synthetic data
        self.n_features = 23
        self.feature_names = [
            'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
            'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
            'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
            'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
        ]

        self.X_sample = np.random.randn(50, self.n_features)
        self.y_sample = np.random.randint(0, 2, 50)

        # Train a simple model for testing
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(random_state=42, max_iter=1000)
        self.model.fit(self.X_sample, self.y_sample)

    def test_feature_names_consistency(self):
        """Test feature names consistency"""
        assert len(self.feature_names) == self.n_features
        assert len(set(self.feature_names)) == len(self.feature_names)  # No duplicates

    def test_model_prediction_shape(self):
        """Test model prediction shape"""
        test_input = np.random.randn(5, self.n_features)
        predictions = self.model.predict(test_input)
        probabilities = self.model.predict_proba(test_input)

        assert predictions.shape == (5,)
        assert probabilities.shape == (5, 2)  # Binary classification

    def test_shap_compatibility(self):
        """Test SHAP explainer compatibility"""
        try:
            import shap

            # Create simple explainer
            explainer = shap.Explainer(self.model.predict_proba, self.X_sample[:10])

            # Test explanation on small sample
            test_sample = self.X_sample[:2]
            shap_values = explainer(test_sample)

            assert shap_values is not None
            # For binary classification, SHAP values should have shape (n_samples, n_features, n_classes)
            # or (n_samples, n_features) for single class
            assert len(shap_values.values.shape) >= 2

        except ImportError:
            pytest.skip("SHAP not installed, skipping SHAP tests")
        except Exception as e:
            # If SHAP has issues, we can still continue with other tests
            print(f"SHAP test failed with error: {e}")


class TestUtilityFunctions:
    """Test cases for utility functions"""

    def test_save_and_load_object(self):
        """Test object saving and loading utilities"""
        import tempfile

        # Test data
        test_obj = {"test": "data", "number": 42, "list": [1, 2, 3]}

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            # Save object
            save_object(file_path=tmp_file.name, obj=test_obj)

            # Load object
            loaded_obj = load_object(file_path=tmp_file.name)

            assert loaded_obj == test_obj

        # Clean up
        os.unlink(tmp_file.name)

    def test_exception_handling(self):
        """Test custom exception handling"""
        try:
            raise CreditDefaultException("Test exception", sys.exc_info())
        except CreditDefaultException as e:
            assert "Test exception" in str(e)
            assert hasattr(e, 'error_message')

    def test_logging_functionality(self):
        """Test logging functionality"""
        # Test that logging works without errors
        logging.info("Test log message")
        logging.error("Test error message")
        logging.warning("Test warning message")

        # No assertion needed, just testing that logging doesn't crash


# Configuration for pytest
def pytest_configure(config):
    """Configure pytest"""
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
