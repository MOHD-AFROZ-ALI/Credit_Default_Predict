from pathlib import Path

from credit_default.utils import load_object
from credit_default.logger import logger
# Path to the explanations.pkl file
explanation_file_path = Path("C:\\Users\\hp\\Desktop\\credit card\\credit_default_prediction_complete\\credit_default_prediction\\artifacts\\explainer\\explanations.pkl")
# Load and summarize the explanations


def load_and_summarize_explanations(explanation_file_path: Path) -> None:
    """
    Load explanations from a pickle file and print a summary.

    Args:
        file_path: Path to the explanations.pkl file.
    """
    try:
        # Load the explanations data
        explanations = load_object(explanation_file_path)
        
        # Print model information
        model_info = explanations.get('model_info', {})
        print("Model Information:")
        print(f"  Model Type: {model_info.get('model_type', 'N/A')}")
        print(f"  Number of Features: {model_info.get('n_features', 'N/A')}")
        print(f"  Sample Size: {model_info.get('sample_size', 'N/A')}")
        
        # Print feature names
        feature_names = explanations.get('feature_names', [])
        print("\nFeature Names:")
        print(", ".join(feature_names))
        
        # Print global explanations
        global_explanations = explanations.get('global_explanations', {})
        print("\nGlobal Explanations:")
        
        # Feature importance
        feature_importance = global_explanations.get('feature_importance', {})
        print("\nFeature Importance:")
        for feature, importance in feature_importance.items():
            print(f"  {feature}: {importance:.4f}")
        
        # Top features
        top_features = global_explanations.get('top_features', [])
        print("\nTop Features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
        
        # Summary statistics
        summary_stats = global_explanations.get('summary_stats', {})
        print("\nSummary Statistics:")
        print(f"  Mean SHAP Values: {summary_stats.get('mean_shap_values', [])}")
        print(f"  Std SHAP Values: {summary_stats.get('std_shap_values', [])}")
        print(f"  Number of Samples: {summary_stats.get('n_samples', 'N/A')}")
        print(f"  Number of Features: {summary_stats.get('n_features', 'N/A')}")


        feature_names_path = Path("C:\\Users\\hp\\Desktop\\credit card\\credit_default_prediction_complete\\credit_default_prediction\\artifacts\\explainer\\feature_names.pkl")

        # Load and print the feature names
        try:
            feature_names = load_object(feature_names_path)
            print("Feature Names:")
            print(feature_names)
        except Exception as e:
            print(f"Error loading feature names: {e}")

    except Exception as e:
        logger.error(f"Error loading and summarizing explanations: {e}")

        

    # Load and print the feature names
    try:
        feature_names = load_object(feature_names_path)
        print("Feature Names:")
        print(feature_names)
    except Exception as e:
        print(f"Error loading feature names: {e}")

if __name__ == "__main__":
    # Run the function to load and summarize explanations
    load_and_summarize_explanations(explanation_file_path)