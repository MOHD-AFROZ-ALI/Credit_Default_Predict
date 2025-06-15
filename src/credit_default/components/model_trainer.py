"""
Model Trainer Component for Credit Default Prediction
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb

from credit_default.entity import ModelTrainerConfig
from credit_default.exception import ModelTrainerException
from credit_default.logger import logger
from credit_default.utils import save_object, evaluate_models


class ModelTrainer:
    """Model trainer component"""

    def __init__(self, config: ModelTrainerConfig):
        """
        Initialize model trainer

        Args:
            config: Model trainer configuration
        """
        self.config = config

    def get_models_and_params(self) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Get models and their hyperparameter grids

        Returns:
            Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]: Models and parameter grids
        """
        try:
            models = {
                #"Random Forest": RandomForestClassifier(random_state=self.config.random_state),
                #"Gradient Boosting": GradientBoostingClassifier(random_state=self.config.random_state),
                "Logistic Regression": LogisticRegression(random_state=self.config.random_state, max_iter=1000),
                #"XGBClassifier": xgb.XGBClassifier(random_state=self.config.random_state, eval_metric='logloss')
            }

            # Get hyperparameter grids from config
            algorithms_config = self.config.algorithms

            params = {}

            # Random Forest parameters
            # if "random_forest" in algorithms_config:
            #     rf_params = algorithms_config["random_forest"]["hyperparameters"]
            #     params["Random Forest"] = rf_params
            # else:
            #     params["Random Forest"] = {
            #         'n_estimators': [100, 200],
            #         'max_depth': [10, 20],
            #         'min_samples_split': [2, 5],
            #         'min_samples_leaf': [1, 2]
            #     }

            # # Gradient Boosting parameters
            # if "gradient_boosting" in algorithms_config:
            #     gb_params = algorithms_config["gradient_boosting"]["hyperparameters"]
            #     params["Gradient Boosting"] = gb_params
            # else:
            #     params["Gradient Boosting"] = {
            #         'n_estimators': [100, 200],
            #         'learning_rate': [0.01, 0.1],
            #         'max_depth': [3, 5]
            #     }

            # Logistic Regression parameters
            if "logistic_regression" in algorithms_config:
                lr_params = algorithms_config["logistic_regression"]["hyperparameters"]
                params["Logistic Regression"] = lr_params
            else:
                params["Logistic Regression"] = {
                    'C': [0.1, 1, 10],
                    'penalty': ['l2'],
                    'solver': ['liblinear']
                }

            # # XGBoost parameters
            # if "xgboost" in algorithms_config:
            #     xgb_params = algorithms_config["xgboost"]["hyperparameters"]
            #     params["XGBClassifier"] = xgb_params
            # else:
            #     params["XGBClassifier"] = {
            #         'n_estimators': [100, 200],
            #         'max_depth': [3, 5],
            #         'learning_rate': [0.01, 0.1],
            #         'subsample': [0.8, 1.0]
            #     }

            return models, params

        except Exception as e:
            raise ModelTrainerException(f"Error getting models and parameters: {e}", sys)

    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]

            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }

            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            metrics['confusion_matrix'] = cm.tolist()

            return metrics

        except Exception as e:
            raise ModelTrainerException(f"Error evaluating model: {e}", sys)

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Initiate model training process

        Args:
            train_array: Training data array
            test_array: Test data array

        Returns:
            float: Best model score
        """
        try:
            logger.info("Starting model training process")

            # Split the data
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            logger.info(f"Training data shape: {X_train.shape}")
            logger.info(f"Test data shape: {X_test.shape}")

            # Get models and parameters
            models, params = self.get_models_and_params()

            logger.info(f"Training {len(models)} models")

            # Evaluate models
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param_grids=params
            )

            # Get best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            logger.info(f"Best model: {best_model_name} with score: {best_model_score:.4f}")

            if best_model_score < 0.6:
                logger.warning(f"Best model score {best_model_score:.4f} is below threshold 0.6")

            # Train the best model with best parameters
            best_model = models[best_model_name]
            best_params = params[best_model_name]

            # Perform grid search for the best model
            grid_search = GridSearchCV(
                estimator=best_model,
                param_grid=best_params,
                scoring=self.config.scoring,
                cv=self.config.cv_folds,
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)
            best_model_fitted = grid_search.best_estimator_

            logger.info(f"Best parameters for {best_model_name}: {grid_search.best_params_}")

            # Evaluate the best model
            metrics = self.evaluate_model(best_model_fitted, X_test, y_test)

            logger.info("Model evaluation metrics:")
            for metric, value in metrics.items():
                if metric != 'confusion_matrix':
                    logger.info(f"{metric}: {value:.4f}")

            # Save the best model
            save_object(
                file_path=self.config.best_model_path,
                obj=best_model_fitted
            )

            # Save model metrics
            metrics_data = {
                'best_model': best_model_name,
                'best_params': grid_search.best_params_,
                'best_score': float(best_model_score),
                'evaluation_metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                                     for k, v in metrics.items()},
                'all_model_scores': {k: float(v) for k, v in model_report.items()}
            }

            with open(self.config.metric_file_path, 'w') as f:
                yaml.dump(metrics_data, f)

            logger.info(f"Saved best model to {self.config.best_model_path}")
            logger.info(f"Saved metrics to {self.config.metric_file_path}")

            # Also save feature importance if available
            if hasattr(best_model_fitted, 'feature_importances_'):
                feature_importance = best_model_fitted.feature_importances_
                importance_data = {
                    'feature_importance': feature_importance.tolist(),
                    'n_features': len(feature_importance)
                }

                importance_path = self.config.root_dir / "feature_importance.yaml"
                with open(importance_path, 'w') as f:
                    yaml.dump(importance_data, f)

                logger.info(f"Saved feature importance to {importance_path}")

            logger.info("Model training completed successfully")

            return best_model_score

        except Exception as e:
            raise ModelTrainerException(f"Error in model training process: {e}", sys)

    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform cross-validation on model

        Args:
            model: Model to validate
            X: Features
            y: Target

        Returns:
            Dict[str, float]: Cross-validation results
        """
        try:
            logger.info(f"Performing {self.config.cv_folds}-fold cross-validation")

            cv_scores = cross_val_score(
                model, X, y, 
                cv=self.config.cv_folds, 
                scoring=self.config.scoring,
                n_jobs=-1
            )

            cv_results = {
                'mean_cv_score': float(cv_scores.mean()),
                'std_cv_score': float(cv_scores.std()),
                'cv_scores': cv_scores.tolist()
            }

            logger.info(f"Cross-validation results: {cv_results['mean_cv_score']:.4f} ± {cv_results['std_cv_score']:.4f}")

            return cv_results

        except Exception as e:
            raise ModelTrainerException(f"Error in cross-validation: {e}", sys)
