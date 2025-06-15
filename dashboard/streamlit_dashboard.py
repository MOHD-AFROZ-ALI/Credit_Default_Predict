"""
Streamlit Dashboard for Credit Default Prediction
"""
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, Any
import yaml
import shap
# Add project path
sys.path.append(str(Path(__file__).parent.parent))

from src.credit_default import pipeline
from src.credit_default.pipeline.prediction_pipeline import PredictionPipeline
from src.credit_default.exception import PredictionException
from src.credit_default.logger import logger
from src.credit_default.utils import load_object
from src.credit_default.pipeline.prediction_pipeline import PredictionPipeline



# shap_explainer = load_object("artifacts/explainer/shap_explainer.pkl")
# feature_names = load_object("artifacts/explainer/feature_names.pkl")

# Configure Streamlit page
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-container {
        background-color: #0D1B2A;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #ffebee;
        border: 2px solid #f44336;
    }
    .risk-medium {
        background-color: #fff3e0;
        border: 2px solid #ff9800;
    }
    .risk-low {
        background-color: #e8f5e8;
        border: 2px solid #4caf50;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Load the prediction pipeline """
    try:
        return PredictionPipeline()
    except Exception as e:
        st.error(f"Failed to load prediction pipeline: {e}")
        return None


def create_risk_gauge(risk_score: float, risk_category: str) -> go.Figure:
    """Create a risk gauge chart"""
    # Color mapping
    color_map = {
        "Low Risk": "#4CAF50",
        "Medium Risk": "#FF9800", 
        "High Risk": "#F44336"
    }

    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = risk_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Risk Score<br><span style='font-size:0.8em;color:gray'>{risk_category}</span>"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': color_map.get(risk_category, "#999999")},
            'steps': [
                {'range': [0, 30], 'color': "#E8F5E8"},
                {'range': [30, 60], 'color': "#FFF3E0"},
                {'range': [60, 100], 'color': "#FFEBEE"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(height=300)
    return fig


def create_feature_importance_chart(explanation: Dict[str, Any]) -> go.Figure:
    """Create feature importance chart from SHAP explanation"""
    if not explanation or 'top_contributions' not in explanation:
        return None

    contributions = explanation['top_contributions']
    features = [item['feature'] for item in contributions]
    values = [item['contribution'] for item in contributions]

    # Create horizontal bar chart
    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=['red' if v > 0 else 'green' for v in values],
            opacity=0.7
        )
    ))

    fig.update_layout(
        title="Top Feature Contributions (SHAP Values)",
        xaxis_title="SHAP Value",
        yaxis_title="Features",
        height=400,
        yaxis={'categoryorder': 'total ascending'}
    )

    return fig


def render_prediction_form():
    """Render the prediction input form"""
    st.header("Single Customer Risk Assessment")

    with st.form("prediction_form"):
        # Personal Information
        st.subheader("Personal Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            limit_bal = st.number_input("Credit Limit", min_value=10000, max_value=1000000, value=50000, step=1000)
            age = st.number_input("Age", min_value=18, max_value=100, value=35)

        with col2:
            sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
            education = st.selectbox("Education", options=[1, 2, 3, 4, 5, 6], 
                                   format_func=lambda x: ["Graduate School", "University", "High School", 
                                                         "Others", "Unknown", "Unknown"][x-1])

        with col3:
            marriage = st.selectbox("Marital Status", options=[1, 2, 3],
                                  format_func=lambda x: ["Married", "Single", "Others"][x-1])

        # Payment History
        st.subheader("Payment History (Past 6 Months)")
        pay_months = ["Sep", "Aug", "Jul", "Jun", "May", "Apr"]
        pay_cols = st.columns(6)
        pay_status = []

        for i, col in enumerate(pay_cols):
            with col:
                pay_val = st.selectbox(
                    pay_months[i],
                    options=list(range(-2, 9)),
                    format_func=lambda x: f"Pay {x}" if x >= 0 else "No consumption" if x == -1 else "Paid in full"
                )
                pay_status.append(pay_val)
        # pay_cols = st.columns(6)
        # pay_status = []
        # for i, col in enumerate(pay_cols):
        #     with col:
        #         if i == 0:
        #             pay_val = st.selectbox(f"Sep", options=list(range(-2, 9)), 
        #                                  format_func=lambda x: f"Pay {x}" if x >= 0 else f"No consumption" if x == -1 else "Paid in full")
        #         else:
        #             pay_val = st.selectbox(f"Aug-{i}", options=list(range(-2, 9)),
        #                                  format_func=lambda x: f"Pay {x}" if x >= 0 else f"No consumption" if x == -1 else "Paid in full")
        #         pay_status.append(pay_val)

        # Bill Amounts
        st.subheader("Bill Amounts (Past 6 Months)")
        bill_cols = st.columns(6)
        bill_amounts = []

        for i, col in enumerate(bill_cols):
            with col:
                bill_amt = st.number_input(
                    f"Bill {6-i}",
                    value=float(15000 - i*1000),
                    step=100.0
                )
                bill_amounts.append(bill_amt)

        # Payment Amounts
        st.subheader("Payment Amounts (Past 6 Months)")
        pay_amt_cols = st.columns(6)
        pay_amounts = []

        for i, col in enumerate(pay_amt_cols):
            with col:
                pay_amt = st.number_input(
                    f"Payment {6-i}",
                    min_value=0.0,
                    value=float(1500 - i*100),
                    step=50.0
                )
                pay_amounts.append(pay_amt)

        submitted = st.form_submit_button("Predict Risk", type="primary")

        if submitted:
            # Prepare input data
            input_data = {
                'LIMIT_BAL': int(limit_bal),
                'SEX': sex,
                'EDUCATION': education,
                'MARRIAGE': marriage,
                'AGE': int(age),
                'PAY_0': pay_status[0],
                'PAY_2': pay_status[1],
                'PAY_3': pay_status[2],
                'PAY_4': pay_status[3],
                'PAY_5': pay_status[4],
                'PAY_6': pay_status[5],
                'BILL_AMT1': bill_amounts[0],
                'BILL_AMT2': bill_amounts[1],
                'BILL_AMT3': bill_amounts[2],
                'BILL_AMT4': bill_amounts[3],
                'BILL_AMT5': bill_amounts[4],
                'BILL_AMT6': bill_amounts[5],
                'PAY_AMT1': pay_amounts[0],
                'PAY_AMT2': pay_amounts[1],
                'PAY_AMT3': pay_amounts[2],
                'PAY_AMT4': pay_amounts[3],
                'PAY_AMT5': pay_amounts[4],
                'PAY_AMT6': pay_amounts[5]
            }

            return input_data

    return None


def render_batch_prediction():
    """Render batch prediction interface"""
    st.header("Batch Risk Assessment")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer data", 
        type=['csv'],
        help="Upload a CSV file with customer data for batch prediction"
    )

    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)

            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions..."):
                    pipeline = load_pipeline()
                    if pipeline:
                        try:
                            results_df = pipeline.batch_predict(df)

                            # Display results
                            st.subheader("Prediction Results")
                            st.dataframe(results_df, use_container_width=True)

                            # Summary statistics
                            col1, col2, col3, col4 = st.columns(4)

                            with col1:
                                st.metric("Total Records", len(results_df))
                            with col2:
                                high_risk = (results_df['risk_category'] == 'High Risk').sum()
                                st.metric("High Risk", high_risk)
                            with col3:
                                medium_risk = (results_df['risk_category'] == 'Medium Risk').sum()
                                st.metric("Medium Risk", medium_risk)
                            with col4:
                                low_risk = (results_df['risk_category'] == 'Low Risk').sum()
                                st.metric("Low Risk", low_risk)

                            # Risk distribution chart
                            risk_dist = results_df['risk_category'].value_counts()
                            fig = px.pie(values=risk_dist.values, names=risk_dist.index, 
                                       title="Risk Distribution")
                            st.plotly_chart(fig, use_container_width=True)

                            # Download results
                            csv = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="batch_predictions.csv",
                                mime="text/csv"
                            )

                        except Exception as e:
                            st.error(f"Error in batch prediction: {e}")
                    else:
                        st.error("Prediction pipeline not available")

        except Exception as e:
            st.error(f"Error reading file: {e}")


def render_model_analytics():
    """Render model analytics and information"""
    st.header("Model Analytics")

    pipeline = load_pipeline()
    if not pipeline:
        st.error("Model not available")
        return

    # Model Information
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Information")
        model_info = {
            "Model Type": type(pipeline.model).__name__,
            "Features": "23+ engineered features",
            "Algorithm": "Tree-based ensemble",
            "Explainability": "SHAP-based" if pipeline.explainer else "Not available"
        }

        for key, value in model_info.items():
            st.write(f"**{key}:** {value}")

    with col2:
        st.subheader("Performance Metrics")
        # Load metrics if available
        metrics_path = Path("artifacts/model_trainer/metrics.yaml")
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = yaml.safe_load(f)

                eval_metrics = metrics.get('evaluation_metrics', {})

                metric_cols = st.columns(2)
                with metric_cols[0]:
                    st.metric("Accuracy", f"{eval_metrics.get('accuracy', 0):.3f}")
                    st.metric("Precision", f"{eval_metrics.get('precision', 0):.3f}")

                with metric_cols[1]:
                    st.metric("Recall", f"{eval_metrics.get('recall', 0):.3f}")
                    st.metric("F1-Score", f"{eval_metrics.get('f1_score', 0):.3f}")

                st.metric("ROC-AUC", f"{eval_metrics.get('roc_auc', 0):.3f}")

            except Exception as e:
                st.warning(f"Could not load metrics: {e}")
        else:
            st.warning("Model metrics not available")

    # Feature Engineering Information
    st.subheader("Feature Engineering")
    st.write("""
    The model uses the following engineered features:
    - **Payment Ratios**: Payment amount to credit limit ratio
    - **Bill Ratios**: Bill amount to credit limit ratio  
    - **Payment Patterns**: Average payment status, maximum delays
    - **Spending Trends**: Bill amount trends over time
    - **Financial Behavior**: Total and average payment amounts
    """)
def about_page():
    """About page with project information"""

    st.markdown("""
    <div style="text-align:center;">
        <h1>💳 Credit Default Prediction Dashboard</h1>
        <h3>End-to-End ML Engineering Showcase</h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🚀 **Features**
    - 🧑‍💼 **Single Prediction**: Predict default risk for individual customers
    - 📂 **Batch Prediction**: Process multiple customers at once via CSV upload
    - 🧠 **Explainable AI**: SHAP-based explanations for model predictions
    - 📊 **Interactive Visualizations**: Risk gauges, feature importance plots, and more
    - 📈 **Model Analytics**: Insights into model performance and prediction history

    ### 🗂️ **Model Details**
    - **Dataset**: UCI Credit Default Dataset
    - **Features**: 23 customer attributes including payment history, bill amounts, and demographics
    - **Target**: Binary classification (Default vs No Default)
    - **Algorithms**: Multiple ML algorithms with hyperparameter tuning
    - **Explainability**: SHAP (SHapley Additive exPlanations) for interpretability

    ### 🛠️ **Technology Stack**
    - 🖥️ **Frontend**: Streamlit
    - ⚡ **Backend**: FastAPI
    - 🤖 **ML Framework**: Scikit-learn, XGBoost
    - 🧩 **Explainability**: SHAP
    - 📊 **Visualizations**: Plotly, Matplotlib
    - 🐳 **Deployment**: Docker, AWS

    ### 📋 **Usage Instructions**
    1. 🧑‍💼 **Single Prediction**: Fill in customer details and get instant risk assessment
    2. 📂 **Batch Prediction**: Upload CSV file with customer data for bulk processing
    3. 📈 **Model Analytics**: View model performance metrics and prediction insights

    ### 📊 **Risk Levels**
    - 🟢 **LOW RISK** (0-30%): Customer unlikely to default
    - 🟡 **MEDIUM RISK** (30-70%): Moderate default risk
    - 🔴 **HIGH RISK** (70-100%): High probability of default

    ---

    <div style="text-align:center;">
        <h4>Developed by <b>MOHD AFROZ ALI</b></h4>
        <a href="https://github.com/MOHD-AFROZ-ALI" target="_blank">
            <img src="https://img.shields.io/badge/GitHub-MOHD--AFROZ--ALI-blue?style=flat-square" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/mohd-afroz-ali" target="_blank">
            <img src="https://img.shields.io/badge/LinkedIn-MOHD--AFROZ--ALI-blue?style=flat-square&logo=linkedin" alt="LinkedIn">
        </a>
        <br>
        <span style="font-size:0.9em;">For technical details and source code, visit the project repository.</span>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application"""
    # Title and description
    st.title("💳 Credit Default Prediction Dashboard")
    st.markdown("""
    **End-to-End ML Dashboard for Credit Risk Assessment with Explainable AI**

    This dashboard provides real-time credit default risk prediction with detailed explanations 
    to help financial institutions make informed lending decisions.
    """)

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page",
        ["Single Prediction", "Batch Prediction", "Model Analytics", "About"]
    )

    # Load pipeline
    pipeline = load_pipeline()

    if pipeline is None:
        st.error("Failed to load prediction pipeline. Please check the model files.")
        st.stop()

    # Page routing
    if page == "Single Prediction":
        input_data = render_prediction_form()

        if input_data:
            with st.spinner("Making prediction..."):
                try:
                    result = pipeline.predict(input_data)
                #      # SHAP explanation
                #     if result.shap_explanation:
                #         st.subheader("Model Explanation")
                #         fig_shap = create_feature_importance_chart(result.shap_explanation)
                #         if fig_shap:
                #             st.plotly_chart(fig_shap, use_container_width=True)

                #     # --- NEW: SHAP force and waterfall plots ---
                #     st.subheader("SHAP Force Plot (Local Explanation)")
                #     def st_shap(plot, height=None):
                #         import streamlit.components.v1 as components
                #         shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                #         components.html(shap_html, height=height)

                #     # Prepare input as DataFrame
                #     input_df = pd.DataFrame([input_data])

                #     # Apply feature engineering as in the pipeline
                #     processed_df = pipeline._apply_feature_engineering(input_df.copy())
                    
                #     processed_df = processed_df.reindex(columns=feature_names, fill_value=0)
                #     # Debug shapes
                #     st.write("processed_df.shape:", processed_df.shape)
                #     st.write("feature_names length:", len(feature_names))
                #    # SHAP explanation
                #     try:
                #         shap_values = shap_explainer(processed_df)
                #         st_shap(
                #             shap.force_plot(
                #                 shap_explainer.expected_value, 
                #                 shap_values.values[0], 
                #                 processed_df.iloc[0]
                #             ),
                #             height=300
                #         )
                #         st.subheader("SHAP Waterfall Plot")
                #         shap.plots.waterfall(shap_values[0], max_display=15)
                #     except Exception as e:
                #             st.error(f"SHAP waterfall plot failed: {e}")
                    # Display results
                    st.success("Prediction completed!")

                    # Risk gauge
                    col1, col2 = st.columns([1, 2])

                    with col1:
                        fig_gauge = create_risk_gauge(result.risk_score, result.risk_category)
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with col2:
                        # Risk details
                        risk_class = "risk-" + result.risk_category.lower().replace(" ", "-")
                        st.markdown(f"""
                        <div class="metric-container {risk_class}">
                            <h3>Risk Assessment Results</h3>
                            <p><strong>Prediction:</strong> {"Default Risk" if result.prediction == 1 else "No Default Risk"}</p>
                            <p><strong>Risk Score:</strong> {result.risk_score:.2f}%</p>
                            <p><strong>Risk Category:</strong> {result.risk_category}</p>
                            <p><strong>Confidence:</strong> {result.probability:.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # SHAP explanation
                    if result.shap_explanation:
                        st.subheader("Model Explanation")
                        fig_shap = create_feature_importance_chart(result.shap_explanation)
                        if fig_shap:
                            st.plotly_chart(fig_shap, use_container_width=True)

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

    elif page == "Batch Prediction":
        render_batch_prediction()

    elif page == "Model Analytics":
        render_model_analytics()
    
    elif page == "About":
        about_page()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.info("""
    This dashboard uses advanced machine learning algorithms 
    with SHAP-based explainability to provide transparent 
    credit risk assessments.

    Built with Streamlit, FastAPI, and scikit-learn.
    """)


if __name__ == "__main__":
    main()
