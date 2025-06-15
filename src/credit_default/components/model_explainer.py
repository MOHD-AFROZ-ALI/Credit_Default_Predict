"""
Model Explainer Component for Credit Default Prediction
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from sklearn.inspection import permutation_importance

from credit_default.entity import ExplainerConfig
from credit_default.exception import ExplainerException
from credit_default.logger import logger
from credit_default.utils import load_object, save_object


class ModelExplainer:
    """Model explainer component using SHAP"""

    def __init__(self, config: ExplainerConfig):
        """
        Initialize model explainer

        Args:
            config: Explainer configuration
        """
        self.config = config
        self.explainer = None
        self.feature_names = None

    def load_model_and_data(self):
        """Load trained model and test data"""
        try:
            # Load the trained model
            self.model = load_object(self.config.model_path)
            logger.info(f"Loaded model from {self.config.model_path}")

            # Load test data
            test_data = np.load(self.config.test_data_path)
            self.X_test = test_data[:, :-1]
            self.y_test = test_data[:, -1]

            logger.info(f"Loaded test data shape: {self.X_test.shape}")

            # Sample data for SHAP if too large
            if self.X_test.shape[0] > self.config.sample_size:
                sample_indices = np.random.choice(
                    self.X_test.shape[0], 
                    self.config.sample_size, 
                    replace=False
                )
                self.X_sample = self.X_test[sample_indices]
                self.y_sample = self.y_test[sample_indices]
                logger.info(f"Sampled {self.config.sample_size} instances for SHAP analysis")
            else:
                self.X_sample = self.X_test
                self.y_sample = self.y_test

        except Exception as e:
            raise ExplainerException(f"Error loading model and data: {e}", sys)

    def create_feature_names(self) -> List[str]:
        """Create feature names for the model"""
        try:
            # Try to load feature names if available
            if self.config.feature_names_path.exists():
                self.feature_names = load_object(self.config.feature_names_path)
            else:
                # Generate generic feature names
                n_features = self.X_test.shape[1]
                self.feature_names = [f"feature_{i}" for i in range(n_features)]

                # Save feature names
                save_object(self.config.feature_names_path, self.feature_names)

            logger.info(f"Created {len(self.feature_names)} feature names")
            return self.feature_names

        except Exception as e:
            raise ExplainerException(f"Error creating feature names: {e}", sys)

    def initialize_explainer(self):
        """Initialize SHAP explainer"""
        try:
            logger.info("Initializing SHAP explainer")

            # Choose appropriate explainer based on model type
            model_name = type(self.model).__name__

            if "Tree" in model_name or "Forest" in model_name or "XGB" in model_name or "Gradient" in model_name:
                # Tree-based models
                self.explainer = shap.TreeExplainer(self.model)
                logger.info("Using TreeExplainer")
            else:
                # Linear models or others
                # Use a subset for background data
                background = shap.sample(self.X_sample, min(100, len(self.X_sample)))
                self.explainer = shap.KernelExplainer(self.model.predict_proba, background)
                logger.info("Using KernelExplainer")

            # Save the explainer
            save_object(self.config.explainer_path, self.explainer)
            logger.info(f"Saved explainer to {self.config.explainer_path}")

        except Exception as e:
            raise ExplainerException(f"Error initializing explainer: {e}", sys)

    def generate_global_explanations(self) -> Dict[str, Any]:
        """Generate global model explanations"""
        try:
            logger.info("Generating global explanations")

            # Calculate SHAP values
            if hasattr(self.explainer, 'shap_values'):
                # For tree explainers
                shap_values = self.explainer.shap_values(self.X_sample)
            else:
                # For other explainers
                shap_values = self.explainer(self.X_sample)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class for binary classification

            global_explanations = {}

            # Feature importance (mean absolute SHAP values)
            feature_importance = np.abs(shap_values).mean(axis=0)
            feature_importance_dict = {
                name: float(importance) 
                for name, importance in zip(self.feature_names, feature_importance)
            }
            global_explanations['feature_importance'] = feature_importance_dict

            # Top important features
            sorted_features = sorted(
                feature_importance_dict.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            global_explanations['top_features'] = sorted_features[:10]

            # Summary statistics
            global_explanations['summary_stats'] = {
                'mean_shap_values': shap_values.mean(axis=0).tolist(),
                'std_shap_values': shap_values.std(axis=0).tolist(),
                'n_samples': int(shap_values.shape[0]),
                'n_features': int(shap_values.shape[1])
            }

            logger.info("Generated global explanations")
            return global_explanations

        except Exception as e:
            raise ExplainerException(f"Error generating global explanations: {e}", sys)

    def generate_local_explanation(self, instance: np.ndarray) -> Dict[str, Any]:
        """
        Generate local explanation for a single instance

        Args:
            instance: Single instance to explain

        Returns:
            Dict[str, Any]: Local explanation
        """
        try:
            if instance.ndim == 1:
                instance = instance.reshape(1, -1)

            # Calculate SHAP values for the instance
            if hasattr(self.explainer, 'shap_values'):
                shap_values = self.explainer.shap_values(instance)
            else:
                shap_values = self.explainer(instance)
                if hasattr(shap_values, 'values'):
                    shap_values = shap_values.values

            # Handle multi-class output
            if isinstance(shap_values, list):
                shap_values = shap_values[1][0]  # Use positive class for binary classification
            else:
                shap_values = shap_values[0] if shap_values.ndim > 1 else shap_values

            # Make prediction
            prediction = self.model.predict(instance)[0]
            prediction_proba = self.model.predict_proba(instance)[0]

            local_explanation = {
                'prediction': int(prediction),
                'prediction_probability': prediction_proba.tolist(),
                'shap_values': shap_values.tolist(),
                'feature_values': instance[0].tolist(),
                'feature_names': self.feature_names,
                'base_value': float(self.explainer.expected_value) if hasattr(self.explainer, 'expected_value') else 0.0
            }

            # Feature contributions (SHAP values with feature names)
            contributions = [
                {'feature': name, 'value': float(val), 'shap_value': float(shap_val)}
                for name, val, shap_val in zip(self.feature_names, instance[0], shap_values)
            ]

            # Sort by absolute SHAP value
            contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
            local_explanation['feature_contributions'] = contributions[:10]  # Top 10

            return local_explanation

        except Exception as e:
            raise ExplainerException(f"Error generating local explanation: {e}", sys)

    def create_visualizations(self, shap_values: np.ndarray, save_dir: Path):
        """Create and save SHAP visualizations"""
        try:
            logger.info("Creating SHAP visualizations")
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

            # Summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                self.X_sample, 
                feature_names=self.feature_names,
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.close()

            # Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values, 
                self.X_sample, 
                feature_names=self.feature_names,
                plot_type="bar",
                show=False
            )
            plt.tight_layout()
            plt.savefig(save_dir / 'feature_importance_plot.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"Saved visualizations to {save_dir}")

        except Exception as e:
            logger.warning(f"Error creating visualizations: {e}")

    def initiate_model_explanation(self) -> Dict[str, Any]:
        """Initiate complete model explanation process"""
        try:
            logger.info("Starting model explanation process")

            # Load model and data
            self.load_model_and_data()

            # Create feature names
            self.create_feature_names()

            # Initialize explainer
            self.initialize_explainer()

            # Generate global explanations
            global_explanations = self.generate_global_explanations()

            # Create visualizations
            viz_dir = self.config.root_dir / "visualizations"
            # Note: Visualization creation might fail in some environments
            try:
                if hasattr(self.explainer, 'shap_values'):
                    shap_values = self.explainer.shap_values(self.X_sample)
                    if isinstance(shap_values, list):
                        shap_values = shap_values[1]
                    self.create_visualizations(shap_values, viz_dir)
            except Exception as e:
                logger.warning(f"Could not create visualizations: {e}")

            # Save all explanations
            explanation_data = {
                'global_explanations': global_explanations,
                'model_info': {
                    'model_type': type(self.model).__name__,
                    'n_features': len(self.feature_names),
                    'sample_size': len(self.X_sample)
                },
                'feature_names': self.feature_names
            }

            explanation_path = self.config.root_dir / "explanations.pkl"
            save_object(explanation_path, explanation_data)

            logger.info("Model explanation completed successfully")
            logger.info(f"Saved explanations to {explanation_path}")

            return explanation_data

        except Exception as e:
            raise ExplainerException(f"Error in model explanation process: {e}", sys)
