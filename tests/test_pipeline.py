"""
Integration tests for Credit Default Prediction pipelines
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys
from unittest.mock import patch, MagicMock

# Add project path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.credit_default.pipeline.prediction_pipeline import PredictionPipeline
from src.credit_default.entity import PredictionInput


class TestPredictionPipeline:
    """Test cases for Prediction Pipeline"""

    def setup_method(self):
        """Setup test fixtures"""
        self.sample_input = {
            'LIMIT_BAL': 50000,
            'SEX': 2,
            'EDUCATION': 2,
            'MARRIAGE': 1,
            'AGE': 35,
            'PAY_0': 1,
            'PAY_2': 2,
            'PAY_3': 0,
            'PAY_4': 0,
            'PAY_5': 0,
            'PAY_6': 0,
            'BILL_AMT1': 15000,
            'BILL_AMT2': 14000,
            'BILL_AMT3': 13000,
            'BILL_AMT4': 12000,
            'BILL_AMT5': 11000,
            'BILL_AMT6': 10000,
            'PAY_AMT1': 1500,
            'PAY_AMT2': 1400,
            'PAY_AMT3': 1300,
            'PAY_AMT4': 1200,
            'PAY_AMT5': 1100,
            'PAY_AMT6': 1000
        }

    @patch('credit_default.pipeline.prediction_pipeline.load_object')
    def test_initialization_with_mock(self, mock_load_object):
        """Test pipeline initialization with mocked artifacts"""
        # Mock the model and preprocessor
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

        mock_preprocessor = MagicMock()
        mock_preprocessor.transform.return_value = np.array([[1, 2, 3, 4, 5]])

        # Configure the mock to return different objects based on the path
        def side_effect(path):
            if 'model' in str(path):
                return mock_model
            elif 'preprocessor' in str(path):
                return mock_preprocessor
            else:
                return None

        mock_load_object.side_effect = side_effect

        # Mock path existence
        with patch('pathlib.Path.exists', return_value=True):
            pipeline = PredictionPipeline()

            assert pipeline.model is not None
            assert pipeline.preprocessor is not None

    def test_feature_engineering(self):
        """Test feature engineering functionality"""
        # Create a mock pipeline to test feature engineering
        pipeline = PredictionPipeline.__new__(PredictionPipeline)

        df = pd.DataFrame([self.sample_input])
        df_engineered = pipeline._apply_feature_engineering(df)

        # Check if new features are created
        expected_features = [
            'PAY_AMT_TO_LIMIT_RATIO',
            'BILL_AMT_TO_LIMIT_RATIO',
            'PAY_TO_BILL_RATIO',
            'AVG_PAY_STATUS',
            'MAX_PAY_DELAY',
            'MONTHS_WITH_DELAY',
            'TOTAL_BILL_AMT',
            'AVG_BILL_AMT',
            'BILL_AMT_TREND',
            'TOTAL_PAY_AMT',
            'AVG_PAY_AMT'
        ]

        for feature in expected_features:
            assert feature in df_engineered.columns

        # Check if ratios are calculated correctly
        assert df_engineered['PAY_AMT_TO_LIMIT_RATIO'].iloc[0] >= 0
        assert df_engineered['BILL_AMT_TO_LIMIT_RATIO'].iloc[0] >= 0

    def test_batch_prediction_data_structure(self):
        """Test batch prediction data structure"""
        # Create sample batch data
        batch_data = pd.DataFrame([self.sample_input] * 5)

        # Mock pipeline
        pipeline = PredictionPipeline.__new__(PredictionPipeline)

        # Mock methods
        pipeline._apply_feature_engineering = MagicMock(return_value=batch_data)
        pipeline.preprocessor = MagicMock()
        pipeline.preprocessor.transform.return_value = np.random.rand(5, 10)

        pipeline.model = MagicMock()
        pipeline.model.predict.return_value = np.array([0, 1, 0, 1, 0])
        pipeline.model.predict_proba.return_value = np.array([
            [0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.4, 0.6], [0.7, 0.3]
        ])

        results = pipeline.batch_predict(batch_data)

        # Assertions
        assert len(results) == 5
        assert 'prediction' in results.columns
        assert 'probability' in results.columns
        assert 'risk_score' in results.columns
        assert 'risk_category' in results.columns


class TestInputValidation:
    """Test cases for input validation"""

    def test_valid_input_structure(self):
        """Test valid input structure"""
        valid_input = PredictionInput(
            LIMIT_BAL=50000,
            SEX=2,
            EDUCATION=2,
            MARRIAGE=1,
            AGE=35,
            PAY_0=1,
            PAY_2=2,
            PAY_3=0,
            PAY_4=0,
            PAY_5=0,
            PAY_6=0,
            BILL_AMT1=15000,
            BILL_AMT2=14000,
            BILL_AMT3=13000,
            BILL_AMT4=12000,
            BILL_AMT5=11000,
            BILL_AMT6=10000,
            PAY_AMT1=1500,
            PAY_AMT2=1400,
            PAY_AMT3=1300,
            PAY_AMT4=1200,
            PAY_AMT5=1100,
            PAY_AMT6=1000
        )

        assert valid_input.LIMIT_BAL == 50000
        assert valid_input.SEX == 2
        assert valid_input.AGE == 35

    def test_invalid_input_values(self):
        """Test invalid input values"""
        with pytest.raises(ValueError):
            # Invalid age (too young)
            PredictionInput(
                LIMIT_BAL=50000,
                SEX=2,
                EDUCATION=2,
                MARRIAGE=1,
                AGE=15,  # Invalid age
                PAY_0=1,
                PAY_2=2,
                PAY_3=0,
                PAY_4=0,
                PAY_5=0,
                PAY_6=0,
                BILL_AMT1=15000,
                BILL_AMT2=14000,
                BILL_AMT3=13000,
                BILL_AMT4=12000,
                BILL_AMT5=11000,
                BILL_AMT6=10000,
                PAY_AMT1=1500,
                PAY_AMT2=1400,
                PAY_AMT3=1300,
                PAY_AMT4=1200,
                PAY_AMT5=1100,
                PAY_AMT6=1000
            )


def test_risk_category_mapping():
    """Test risk category mapping"""
    from src.credit_default.utils import create_risk_category

    assert create_risk_category(0.1) == "Low Risk"
    assert create_risk_category(0.45) == "Medium Risk"
    assert create_risk_category(0.8) == "High Risk"

    # Boundary cases
    assert create_risk_category(0.3) == "Medium Risk"
    assert create_risk_category(0.6) == "High Risk"


@pytest.fixture
def mock_trained_artifacts():
    """Fixture for mock trained artifacts"""
    return {
        'model': MagicMock(),
        'preprocessor': MagicMock(),
        'explainer': MagicMock(),
        'feature_names': ['feature_1', 'feature_2', 'feature_3']
    }


def test_prediction_output_structure(mock_trained_artifacts):
    """Test prediction output structure"""
    from src.credit_default.entity import PredictionOutput

    output = PredictionOutput(
        prediction=1,
        probability=0.75,
        risk_score=75.0,
        risk_category="High Risk",
        shap_explanation={'feature_importance': [0.1, 0.2, 0.3]}
    )

    assert output.prediction == 1
    assert output.probability == 0.75
    assert output.risk_score == 75.0
    assert output.risk_category == "High Risk"
    assert output.shap_explanation is not None


if __name__ == "__main__":
    pytest.main([__file__])
