import sys
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from credit_default.logger import logging
from credit_default.exception import ExplainerException


class ModelExplainer:
    """
    Model Explainer class for generating SHAP explanations and visualizations
    """
    
    def __init__(self, 
                 model_path: str,
                 test_data_path: str,
                 visualization_dir: str):
        """
        Initialize ModelExplainer
        
        Args:
            model_path: Path to the trained model
            test_data_path: Path to test data
            visualization_dir: Directory to save visualizations
        """
        self.model_path = model_path
        self.test_data_path = test_data_path
        self.visualization_dir = visualization_dir
        
        # Create visualization directory
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Initialize components
        self.model = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.explainer = None
        self.shap_values = None
        
        logging.info("ModelExplainer initialized")
    
    def load_model_and_data(self):
        """Load the trained model and test data"""
        try:
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info(f"Model loaded from {self.model_path}")
            
            # Load test data
            test_data = pd.read_csv(self.test_data_path)
            
            # Separate features and target
            if 'default.payment.next.month' in test_data.columns:
                target_col = 'default.payment.next.month'
            elif 'default_payment_next_month' in test_data.columns:
                target_col = 'default_payment_next_month'
            else:
                # Assume last column is target
                target_col = test_data.columns[-1]
            
            self.X_test = test_data.drop(columns=[target_col])
            self.y_test = test_data[target_col]
            
            # Get actual feature names
            self.feature_names = list(self.X_test.columns)
            
            logging.info(f"Test data loaded with shape: {self.X_test.shape}")
            logging.info(f"Feature names: {self.feature_names}")
            
        except Exception as e:
            raise ExplainerException(e, sys)
    
    def create_feature_names(self) -> List[str]:
        """
        Create meaningful feature names for visualization
        
        Returns:
            List of feature names
        """
        try:
            if self.feature_names is not None:
                # Map technical names to more readable names
                feature_mapping = {
                    'LIMIT_BAL': 'Credit Limit',
                    'SEX': 'Gender',
                    'EDUCATION': 'Education Level',
                    'MARRIAGE': 'Marital Status',
                    'AGE': 'Age',
                    'PAY_0': 'Payment Status (Current)',
                    'PAY_2': 'Payment Status (2 months ago)',
                    'PAY_3': 'Payment Status (3 months ago)',
                    'PAY_4': 'Payment Status (4 months ago)',
                    'PAY_5': 'Payment Status (5 months ago)',
                    'PAY_6': 'Payment Status (6 months ago)',
                    'BILL_AMT1': 'Bill Amount (Current)',
                    'BILL_AMT2': 'Bill Amount (2 months ago)',
                    'BILL_AMT3': 'Bill Amount (3 months ago)',
                    'BILL_AMT4': 'Bill Amount (4 months ago)',
                    'BILL_AMT5': 'Bill Amount (5 months ago)',
                    'BILL_AMT6': 'Bill Amount (6 months ago)',
                    'PAY_AMT1': 'Payment Amount (Current)',
                    'PAY_AMT2': 'Payment Amount (2 months ago)',
                    'PAY_AMT3': 'Payment Amount (3 months ago)',
                    'PAY_AMT4': 'Payment Amount (4 months ago)',
                    'PAY_AMT5': 'Payment Amount (5 months ago)',
                    'PAY_AMT6': 'Payment Amount (6 months ago)',
                    # Add engineered features
                    'credit_utilization_ratio': 'Credit Utilization Ratio',
                    'avg_payment_ratio': 'Avg Payment Ratio',
                    'payment_consistency': 'Payment Consistency',
                    'recent_payment_trend': 'Recent Payment Trend',
                    'bill_payment_ratio': 'Bill to Payment Ratio',
                    'balance_trend': 'Balance Trend'
                }
                
                # Create readable names
                readable_names = []
                for feature in self.feature_names:
                    if feature in feature_mapping:
                        readable_names.append(feature_mapping[feature])
                    else:
                        # Clean up feature name
                        clean_name = feature.replace('_', ' ').title()
                        readable_names.append(clean_name)
                
                return readable_names
            else:
                return [f"Feature_{i}" for i in range(len(self.X_test.columns))]
                
        except Exception as e:
            logging.error(f"Error creating feature names: {str(e)}")
            return [f"Feature_{i}" for i in range(len(self.X_test.columns))]
    
    def create_explainer(self):
        """Create SHAP explainer"""
        try:
            # Use TreeExplainer for tree-based models, LinearExplainer for linear models
            model_type = str(type(self.model)).lower()
            
            if 'xgb' in model_type or 'lgb' in model_type or 'randomforest' in model_type or 'gradientboosting' in model_type:
                # Tree-based model
                self.explainer = shap.TreeExplainer(self.model)
                logging.info("Created TreeExplainer for tree-based model")
            else:
                # Use Explainer for other models
                background_data = shap.sample(self.X_test, min(100, len(self.X_test)))
                self.explainer = shap.Explainer(self.model, background_data)
                logging.info("Created general Explainer")
            
            # Calculate SHAP values for a subset (for performance)
            sample_size = min(500, len(self.X_test))
            sample_indices = np.random.choice(len(self.X_test), sample_size, replace=False)
            X_sample = self.X_test.iloc[sample_indices]
            
            self.shap_values = self.explainer.shap_values(X_sample)
            
            # Handle binary classification case
            if isinstance(self.shap_values, list) and len(self.shap_values) == 2:
                self.shap_values = self.shap_values[1]  # Use positive class
            
            self.X_sample = X_sample
            
            logging.info(f"SHAP values computed for {sample_size} samples")
            
        except Exception as e:
            raise ExplainerException(e, sys)
    
    def generate_summary_plot(self):
        """Generate SHAP summary plot"""
        try:
            plt.figure(figsize=(12, 8))
            readable_names = self.create_feature_names()
            
            # Create a copy of X_sample with readable names
            X_display = self.X_sample.copy()
            X_display.columns = readable_names[:len(X_display.columns)]
            
            shap.summary_plot(
                self.shap_values, 
                X_display,
                show=False,
                max_display=15
            )
            plt.title('SHAP Summary Plot - Feature Impact on Credit Default Prediction', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, 'shap_summary_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Summary plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating summary plot: {str(e)}")
    
    def generate_bar_plot(self):
        """Generate SHAP bar plot"""
        try:
            plt.figure(figsize=(10, 8))
            readable_names = self.create_feature_names()
            
            # Create a copy of X_sample with readable names
            X_display = self.X_sample.copy()
            X_display.columns = readable_names[:len(X_display.columns)]
            
            shap.plots.bar(
                shap.Explanation(
                    values=self.shap_values,
                    data=X_display.values,
                    feature_names=X_display.columns
                ),
                show=False,
                max_display=15
            )
            plt.title('SHAP Bar Plot - Average Feature Importance', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, 'shap_bar_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Bar plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating bar plot: {str(e)}")
    
    def generate_beeswarm_plot(self):
        """Generate SHAP beeswarm plot"""
        try:
            plt.figure(figsize=(12, 8))
            readable_names = self.create_feature_names()
            
            # Create a copy of X_sample with readable names
            X_display = self.X_sample.copy()
            X_display.columns = readable_names[:len(X_display.columns)]
            
            shap.plots.beeswarm(
                shap.Explanation(
                    values=self.shap_values,
                    data=X_display.values,
                    feature_names=X_display.columns
                ),
                show=False,
                max_display=15
            )
            plt.title('SHAP Beeswarm Plot - Feature Impact Distribution', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, 'shap_beeswarm_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Beeswarm plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating beeswarm plot: {str(e)}")
    
    def generate_waterfall_plot(self, instance_idx: int = 0):
        """Generate SHAP waterfall plot for a specific instance"""
        try:
            plt.figure(figsize=(12, 8))
            readable_names = self.create_feature_names()
            
            # Create a copy of X_sample with readable names
            X_display = self.X_sample.copy()
            X_display.columns = readable_names[:len(X_display.columns)]
            
            # Create explanation object for the specific instance
            explanation = shap.Explanation(
                values=self.shap_values[instance_idx],
                base_values=self.explainer.expected_value,
                data=X_display.iloc[instance_idx].values,
                feature_names=X_display.columns
            )
            
            shap.plots.waterfall(explanation, show=False, max_display=15)
            plt.title(f'SHAP Waterfall Plot - Individual Prediction Explanation (Sample {instance_idx})', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, f'shap_waterfall_plot_sample_{instance_idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Waterfall plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating waterfall plot: {str(e)}")
    
    def generate_force_plot(self, instance_idx: int = 0):
        """Generate SHAP force plot for a specific instance"""
        try:
            readable_names = self.create_feature_names()
            
            # Create force plot
            shap_html = shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_idx],
                self.X_sample.iloc[instance_idx],
                feature_names=readable_names[:len(self.X_sample.columns)],
                show=False
            )
            
            # Save as HTML
            save_path = os.path.join(self.visualization_dir, f'shap_force_plot_sample_{instance_idx}.html')
            shap.save_html(save_path, shap_html)
            
            logging.info(f"Force plot saved to {save_path}")
            
            # Also create a static version
            plt.figure(figsize=(15, 3))
            shap.force_plot(
                self.explainer.expected_value,
                self.shap_values[instance_idx],
                self.X_sample.iloc[instance_idx],
                feature_names=readable_names[:len(self.X_sample.columns)],
                matplotlib=True,
                show=False
            )
            plt.title(f'SHAP Force Plot - Individual Prediction (Sample {instance_idx})', 
                     fontsize=12, fontweight='bold', pad=10)
            
            # Save static plot
            static_save_path = os.path.join(self.visualization_dir, f'shap_force_plot_sample_{instance_idx}.png')
            plt.savefig(static_save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logging.error(f"Error generating force plot: {str(e)}")
    
    def generate_dependence_plots(self, top_features: int = 5):
        """Generate SHAP dependence plots for top features"""
        try:
            # Get feature importance
            feature_importance = np.abs(self.shap_values).mean(0)
            top_indices = np.argsort(feature_importance)[-top_features:]
            
            readable_names = self.create_feature_names()
            
            for i, feature_idx in enumerate(top_indices):
                plt.figure(figsize=(10, 6))
                
                # Create a copy of X_sample with readable names
                X_display = self.X_sample.copy()
                X_display.columns = readable_names[:len(X_display.columns)]
                
                shap.plots.partial_dependence(
                    feature_idx, 
                    self.model.predict, 
                    X_display, 
                    ice=False,
                    model_expected_value=True,
                    feature_expected_value=True,
                    show=False
                )
                
                plt.title(f'SHAP Dependence Plot - {readable_names[feature_idx]}', 
                         fontsize=14, fontweight='bold', pad=20)
                plt.tight_layout()
                
                # Save plot
                feature_name_clean = readable_names[feature_idx].replace(' ', '_').replace('/', '_')
                save_path = os.path.join(self.visualization_dir, f'shap_dependence_{feature_name_clean}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logging.info(f"Dependence plot for {readable_names[feature_idx]} saved to {save_path}")
                
        except Exception as e:
            logging.error(f"Error generating dependence plots: {str(e)}")
    
    def generate_decision_plot(self, num_samples: int = 20):
        """Generate SHAP decision plot"""
        try:
            plt.figure(figsize=(12, 8))
            readable_names = self.create_feature_names()
            
            # Select random samples
            sample_indices = np.random.choice(len(self.shap_values), 
                                            min(num_samples, len(self.shap_values)), 
                                            replace=False)
            
            shap.decision_plot(
                self.explainer.expected_value,
                self.shap_values[sample_indices],
                self.X_sample.iloc[sample_indices],
                feature_names=readable_names[:len(self.X_sample.columns)],
                show=False,
                feature_display_range=slice(None, -16, -1)  # Show top 15 features
            )
            
            plt.title('SHAP Decision Plot - Prediction Paths', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, 'shap_decision_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Decision plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating decision plot: {str(e)}")
    
    def generate_heatmap_plot(self, num_samples: int = 50):
        """Generate SHAP heatmap plot"""
        try:
            plt.figure(figsize=(14, 10))
            readable_names = self.create_feature_names()
            
            # Select random samples
            sample_indices = np.random.choice(len(self.shap_values), 
                                            min(num_samples, len(self.shap_values)), 
                                            replace=False)
            
            # Create a copy of X_sample with readable names
            X_display = self.X_sample.copy()
            X_display.columns = readable_names[:len(X_display.columns)]
            
            shap.plots.heatmap(
                shap.Explanation(
                    values=self.shap_values[sample_indices],
                    data=X_display.iloc[sample_indices].values,
                    feature_names=X_display.columns
                ),
                show=False,
                max_display=15
            )
            
            plt.title('SHAP Heatmap - Feature Contributions Across Samples', 
                     fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, 'shap_heatmap_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Heatmap plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating heatmap plot: {str(e)}")
    
    def generate_feature_importance_plot(self):
        """Generate traditional feature importance plot"""
        try:
            # Calculate feature importance from SHAP values
            feature_importance = np.abs(self.shap_values).mean(0)
            readable_names = self.create_feature_names()
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': readable_names[:len(feature_importance)],
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            # Plot
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'][-15:], importance_df['importance'][-15:])
            plt.xlabel('Mean |SHAP Value|')
            plt.title('Feature Importance (Top 15 Features)', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            # Save plot
            save_path = os.path.join(self.visualization_dir, 'feature_importance_plot.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"Feature importance plot saved to {save_path}")
            
        except Exception as e:
            logging.error(f"Error generating feature importance plot: {str(e)}")
    
    def generate_summary_statistics(self):
        """Generate summary statistics about SHAP values"""
        try:
            readable_names = self.create_feature_names()
            
            # Calculate statistics
            stats = {
                'feature_names': readable_names[:len(self.shap_values[0])],
                'mean_shap_values': np.mean(self.shap_values, axis=0),
                'std_shap_values': np.std(self.shap_values, axis=0),
                'mean_abs_shap_values': np.mean(np.abs(self.shap_values), axis=0),
                'expected_value': self.explainer.expected_value,
                'num_samples': len(self.shap_values),
                'num_features': len(self.shap_values[0])
            }
            
            # Create summary DataFrame
            summary_df = pd.DataFrame({
                'Feature': stats['feature_names'],
                'Mean_SHAP': stats['mean_shap_values'],
                'Std_SHAP': stats['std_shap_values'],
                'Mean_Abs_SHAP': stats['mean_abs_shap_values']
            })
            
            # Sort by absolute importance
            summary_df = summary_df.sort_values('Mean_Abs_SHAP', ascending=False)
            
            # Save summary
            save_path = os.path.join(self.visualization_dir, 'shap_summary_statistics.csv')
            summary_df.to_csv(save_path, index=False)
            
            # Save detailed statistics
            stats_save_path = os.path.join(self.visualization_dir, 'shap_statistics.pkl')
            with open(stats_save_path, 'wb') as f:
                pickle.dump(stats, f)
            
            logging.info(f"Summary statistics saved to {save_path}")
            
            return summary_df
            
        except Exception as e:
            logging.error(f"Error generating summary statistics: {str(e)}")
            return None
    def explain_instance(self, instance_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate SHAP explanation for a single instance
        
        Args:
            instance_data: Single row DataFrame with feature values
            
        Returns:
            Dictionary containing SHAP values and explanations
        """
        try:
            # Calculate SHAP values for the instance
            shap_values_instance = self.explainer.shap_values(instance_data)
            
            # Handle binary classification case
            if isinstance(shap_values_instance, list):
                shap_values_instance = shap_values_instance[1]  # Use positive class
            
            readable_names = self.create_feature_names()
            
            # Create explanation dictionary
            explanation = {
                'shap_values': shap_values_instance[0],
                'feature_names': readable_names[:len(instance_data.columns)],
                'feature_values': instance_data.iloc[0].values,
                'expected_value': self.explainer.expected_value,
                'prediction': self.model.predict_proba(instance_data)[0][1] if hasattr(self.model, 'predict_proba') else self.model.predict(instance_data)[0]
            }
            
            return explanation
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def generate_all_visualizations(self):
        """Generate all SHAP visualizations"""
        try:
            logging.info("Starting SHAP visualization generation...")
            
            # Load model and data
            self.load_model_and_data()
            
            # Create explainer
            self.create_explainer()
            
            # Generate all plots
            self.generate_summary_plot()
            self.generate_bar_plot()
            self.generate_beeswarm_plot()
            self.generate_waterfall_plot(0)  # First sample
            self.generate_waterfall_plot(1)  # Second sample
            self.generate_force_plot(0)  # First sample
            self.generate_force_plot(1)  # Second sample
            self.generate_dependence_plots(5)  # Top 5 features
            self.generate_decision_plot(20)  # 20 samples
            self.generate_heatmap_plot(50)  # 50 samples
            self.generate_feature_importance_plot()
            
            # Generate summary statistics
            summary_stats = self.generate_summary_statistics()
            
            logging.info("All SHAP visualizations generated successfully!")
            
            return summary_stats
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    try:
        # Example usage
        model_path = "C:\\Users\\hp\\Desktop\\credit card\\credit_default_prediction_complete\\credit_default_prediction\\artifacts\\model_trainer\\best_model.pkl"
        test_data_path = "C:\\Users\\hp\\Desktop\\credit card\\credit_default_prediction_complete\\credit_default_prediction\\artifacts\\data_transformation\\test.csv"
        visualization_dir = "C:\\Users\\hp\\Desktop\\credit card\\credit_default_prediction_complete\\credit_default_prediction\\artifacts\\explainer\\new_output"

        explainer = ModelExplainer(
            model_path=model_path,
            test_data_path=test_data_path,
            visualization_dir=visualization_dir
        )
        
        # Generate all visualizations
        summary_stats = explainer.generate_all_visualizations()
        
        if summary_stats is not None:
            print("\nTop 10 Most Important Features:")
            print(summary_stats.head(10))
        
    except Exception as e:
        logging.error(f"Error in ModelExplainer execution: {str(e)}")
        raise CustomException(e, sys)