"""
Prediction Pipeline for Credit Default Prediction
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from credit_default.entity import PredictionInput, PredictionOutput
from credit_default.exception import PredictionException
from credit_default.logger import logger
from credit_default.utils import load_object, create_risk_category


class PredictionPipeline:
    """Prediction pipeline for credit default prediction"""

    def __init__(self):
        """Initialize prediction pipeline"""
        self.model_path = Path("artifacts/model_trainer/best_model.pkl")
        self.preprocessor_path = Path("artifacts/data_transformation/preprocessor.pkl")
        self.explainer_path = Path("artifacts/explainer/shap_explainer.pkl")
        self.feature_names_path = Path("artifacts/explainer/feature_names.pkl")

        # Load artifacts
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model artifacts"""
        try:
            # Load model
            if self.model_path.exists():
                self.model = load_object(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            else:
                raise FileNotFoundError(f"Model not found at {self.model_path}")

            # Load preprocessor
            if self.preprocessor_path.exists():
                self.preprocessor = load_object(self.preprocessor_path)
                logger.info(f"Loaded preprocessor from {self.preprocessor_path}")
            else:
                raise FileNotFoundError(f"Preprocessor not found at {self.preprocessor_path}")

            # Load explainer (optional)
            try:
                if self.explainer_path.exists():
                    self.explainer = load_object(self.explainer_path)
                    logger.info(f"Loaded explainer from {self.explainer_path}")
                else:
                    self.explainer = None
                    logger.warning("Explainer not found - predictions will not include explanations")
            except Exception as e:
                logger.warning(f"Could not load explainer: {e}")
                self.explainer = None

            # Load feature names (optional)
            try:
                if self.feature_names_path.exists():
                    self.feature_names = load_object(self.feature_names_path)
                    logger.info(f"Loaded feature names from {self.feature_names_path}")
                else:
                    self.feature_names = None
            except Exception as e:
                logger.warning(f"Could not load feature names: {e}")
                self.feature_names = None

        except Exception as e:
            raise PredictionException(f"Error loading artifacts: {e}", sys)

    def predict(self, input_data: dict) -> PredictionOutput:
        """
        Make prediction for single instance

        Args:
            input_data: Dictionary with input features

        Returns:
            PredictionOutput: Prediction results
        """
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])

            # Apply feature engineering (same as training)
            df = self._apply_feature_engineering(df)

            # Preprocess the data
            processed_data = self.preprocessor.transform(df)

            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            prediction_proba = self.model.predict_proba(processed_data)[0]

            # Calculate risk score and category
            risk_score = float(prediction_proba[1])  # Probability of default
            risk_category = create_risk_category(risk_score)

            # Generate explanation if explainer is available
            shap_explanation = None
            if self.explainer is not None:
                try:
                    shap_explanation = self._generate_explanation(processed_data[0])
                except Exception as e:
                    logger.warning(f"Could not generate explanation: {e}")

            # Create output
            output = PredictionOutput(
                prediction=int(prediction),
                probability=risk_score,
                risk_score=risk_score * 100,  # Convert to percentage
                risk_category=risk_category,
                shap_explanation=shap_explanation
            )

            logger.info(f"Prediction completed: {risk_category} (Score: {risk_score:.4f})")

            return output

        except Exception as e:
            raise PredictionException(f"Error making prediction: {e}", sys)

    def batch_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Make batch predictions

        Args:
            data: DataFrame with input features

        Returns:
            pd.DataFrame: DataFrame with predictions
        """
        try:
            logger.info(f"Making batch predictions for {len(data)} instances")

            # Clean inf/-inf/NaN values before feature engineering
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.fillna(0, inplace=True)

            # Apply feature engineering
            processed_df = self._apply_feature_engineering(data.copy())

            # Preprocess the data
            processed_data = self.preprocessor.transform(processed_df)

            # Make predictions
            predictions = self.model.predict(processed_data)
            prediction_probas = self.model.predict_proba(processed_data)[:, 1]

            # Create results DataFrame
            results = data.copy()
            results['prediction'] = predictions
            results['probability'] = prediction_probas
            results['risk_score'] = prediction_probas * 100
            results['risk_category'] = [create_risk_category(p) for p in prediction_probas]

            logger.info("Batch predictions completed")

            return results

        except Exception as e:
            raise PredictionException(f"Error in batch prediction: {e}", sys)

    def _apply_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the same feature engineering as in training

        Args:
            df: Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        try:
            df_eng = df.copy()

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

            # Average payment status
            pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
            df_eng['AVG_PAY_STATUS'] = df_eng[pay_cols].mean(axis=1)

            # Maximum payment delay
            df_eng['MAX_PAY_DELAY'] = df_eng[pay_cols].max(axis=1)

            # Number of months with payment delay
            df_eng['MONTHS_WITH_DELAY'] = (df_eng[pay_cols] > 0).sum(axis=1)

            # Total bill amount (last 6 months)
            bill_cols = ['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']
            df_eng['TOTAL_BILL_AMT'] = df_eng[bill_cols].sum(axis=1)

            # Average bill amount
            df_eng['AVG_BILL_AMT'] = df_eng[bill_cols].mean(axis=1)

            # Bill amount trend
            df_eng['BILL_AMT_TREND'] = df_eng['BILL_AMT1'] - df_eng['BILL_AMT6']

            # Total payment amount
            pay_amt_cols = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
            df_eng['TOTAL_PAY_AMT'] = df_eng[pay_amt_cols].sum(axis=1)

            # Average payment amount
            df_eng['AVG_PAY_AMT'] = df_eng[pay_amt_cols].mean(axis=1)

            return df_eng

        except Exception as e:
            raise PredictionException(f"Error in feature engineering: {e}", sys)

    def _generate_explanation(self, instance: np.ndarray) -> dict:
        """
        Generate SHAP explanation for an instance

        Args:
            instance: Single instance array

        Returns:
            dict: SHAP explanation
        """
        try:
            if self.explainer is None:
                return None

            # Calculate SHAP values
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(instance.reshape(1, -1))
                if isinstance(shap_values, list):
                    shap_values = shap_values[1][0]  # Positive class
                else:
                    shap_values = shap_values[0]
            else:
                shap_values = self.explainer(instance.reshape(1, -1))
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values[0]

            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values.tolist() if hasattr(shap_values, 'tolist') else shap_values,
                'feature_values': instance.tolist(),
                'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
            }

            # Add feature names if available
            if self.feature_names:
                explanation['feature_names'] = self.feature_names

                # Top contributing features
                # contributions = [
                #     {'feature': name, 'contribution': float(shap_val)}
                #     for name, shap_val in zip(self.feature_names, shap_values)
                # ]
            contributions = []
            for name, shap_val in zip(self.feature_names, shap_values):
                # If shap_val is array-like, take first element
                if hasattr(shap_val, '__len__') and not isinstance(shap_val, str):
                    val = float(shap_val[0])
                else:
                    val = float(shap_val)
                contributions.append({'feature': name, 'contribution': val})
            contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
            explanation['top_contributions'] = contributions[:5]

            return explanation

        except Exception as e:
            logger.warning(f"Error generating explanation: {e}")
            return None


def main():
    """Main function for testing prediction pipeline"""
    try:
        # Create sample input
        sample_input = {
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

        # Make prediction
        pipeline = PredictionPipeline()
        result = pipeline.predict(sample_input)

        print(f"Prediction: {result.prediction}")
        print(f"Risk Score: {result.risk_score:.2f}%")
        print(f"Risk Category: {result.risk_category}")

    except Exception as e:
        logger.error(f"Prediction pipeline test failed: {e}")


if __name__ == "__main__":
    main()
