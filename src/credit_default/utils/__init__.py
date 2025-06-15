"""
Utility functions for Credit Default Prediction
"""
import os
import sys
import pickle
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from box import ConfigBox
# from box import Box
# from ensure import ensure_annotations

from credit_default.logger import logger
from credit_default.exception import CreditDefaultException


# @ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Read yaml file and return ConfigBox

    Args:
        path_to_yaml: Path to yaml file

    Returns:
        ConfigBox: ConfigBox type

    Raises:
        ValueError: If yaml file is empty
        CreditDefaultException: For other errors
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        raise CreditDefaultException(f"Error reading yaml file: {e}", sys)


# @ensure_annotations
def create_directories(path_to_directories: List[Path], verbose: bool = True):
    """
    Create list of directories

    Args:
        path_to_directories: List of path of directories
        verbose: Ignore if multiple dirs is to be created
    """
    for path in path_to_directories:
        path.mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


# @ensure_annotations
def save_json(path: Path, data: Dict[str, Any]):
    """
    Save json data

    Args:
        path: Path to json file
        data: Data to be saved in json file
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        raise CreditDefaultException(f"Error saving JSON file: {e}", sys)


# @ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Load json files data

    Args:
        path: Path to json file

    Returns:
        ConfigBox: Data as class attributes instead of dict
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded successfully from: {path}")
        return ConfigBox(content)
    except Exception as e:
        raise CreditDefaultException(f"Error loading JSON file: {e}", sys)


# @ensure_annotations
def save_bin(data: Any, path: Path):
    """
    Save binary file

    Args:
        data: Data to be saved as binary
        path: Path to binary file
    """
    try:
        joblib.dump(value=data, filename=path)
        logger.info(f"Binary file saved at: {path}")
    except Exception as e:
        raise CreditDefaultException(f"Error saving binary file: {e}", sys)


# @ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Load binary data

    Args:
        path: Path to binary file

    Returns:
        Any: Object stored in the file
    """
    try:
        data = joblib.load(path)
        logger.info(f"Binary file loaded from: {path}")
        return data
    except Exception as e:
        raise CreditDefaultException(f"Error loading binary file: {e}", sys)


# @ensure_annotations
def get_size(path: Path) -> str:
    """
    Get size in KB

    Args:
        path: Path of the file

    Returns:
        str: Size in KB
    """
    size_in_kb = round(path.stat().st_size / 1024)
    return f"~ {size_in_kb} KB"


# @ensure_annotations
def save_object(file_path: Path, obj: Any):
    """
    Save object as pickle file

    Args:
        file_path: Path to save the object
        obj: Object to save
    """
    try:
        dir_path = file_path.parent
        dir_path.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CreditDefaultException(f"Error saving object: {e}", sys)


# @ensure_annotations
def load_object(file_path: Path) -> Any:
    """
    Load object from pickle file

    Args:
        file_path: Path to the pickle file

    Returns:
        Any: Loaded object
    """
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logger.info(f"Object loaded from: {file_path}")
        return obj
    except Exception as e:
        raise CreditDefaultException(f"Error loading object: {e}", sys)


# @ensure_annotations
def evaluate_models(
    X_train: np.ndarray,
    y_train: np.ndarray, 
    X_test: np.ndarray,
    y_test: np.ndarray,
    models: Dict[str, Any],
    param_grids: Dict[str, Dict[str, Any]]
) -> Dict[str, float]:
    """
    Evaluate multiple models with hyperparameter tuning

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: Dictionary of models
        param_grids: Dictionary of parameter grids for each model

    Returns:
        Dict[str, float]: Model scores
    """
    try:
        from sklearn.model_selection import GridSearchCV
        from sklearn.metrics import roc_auc_score

        model_report = {}

        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")

            # Get parameter grid for the model
            param_grid = param_grids.get(model_name, {})

            if param_grid:
                # Perform grid search
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    scoring='roc_auc',
                    cv=5,
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train, y_train)

                # Use best model for prediction
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict_proba(X_test)[:, 1]
                test_score = roc_auc_score(y_test, y_pred)

                logger.info(f"{model_name} - Best params: {grid_search.best_params_}")
                logger.info(f"{model_name} - Test ROC-AUC: {test_score:.4f}")

            else:
                # Train model without hyperparameter tuning
                model.fit(X_train, y_train)
                y_pred = model.predict_proba(X_test)[:, 1]
                test_score = roc_auc_score(y_test, y_pred)

                logger.info(f"{model_name} - Test ROC-AUC: {test_score:.4f}")

            model_report[model_name] = test_score

        return model_report

    except Exception as e:
        raise CreditDefaultException(f"Error evaluating models: {e}", sys)


# @ensure_annotations
def create_risk_category(probability: float) -> str:
    """
    Create risk category based on probability

    Args:
        probability: Prediction probability

    Returns:
        str: Risk category
    """
    if probability < 0.3:
        return "Low Risk"
    elif probability < 0.6:
        return "Medium Risk"
    else:
        return "High Risk"


# @ensure_annotations
def validate_input_data(data: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Validate input data against schema

    Args:
        data: Input data
        schema: Data schema

    Returns:
        bool: Validation result
    """
    try:
        for column, config in schema.get("columns", {}).items():
            if column not in data:
                logger.error(f"Missing column: {column}")
                return False

            value = data[column]
            expected_type = config.get("type")

            # Type validation
            if expected_type == "int64" and not isinstance(value, (int, np.integer)):
                logger.error(f"Invalid type for {column}: expected int, got {type(value)}")
                return False
            elif expected_type == "float64" and not isinstance(value, (int, float, np.number)):
                logger.error(f"Invalid type for {column}: expected float, got {type(value)}")
                return False

            # Range validation
            if "min_value" in config and value < config["min_value"]:
                logger.error(f"Value {value} for {column} is below minimum {config['min_value']}")
                return False
            if "max_value" in config and value > config["max_value"]:
                logger.error(f"Value {value} for {column} is above maximum {config['max_value']}")
                return False

            # Allowed values validation
            if "allowed_values" in config and value not in config["allowed_values"]:
                logger.error(f"Value {value} for {column} not in allowed values {config['allowed_values']}")
                return False

        return True

    except Exception as e:
        logger.error(f"Error validating input data: {e}")
        return False


# Add missing import
import json
