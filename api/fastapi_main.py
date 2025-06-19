"""
FastAPI Application for Credit Default Prediction
"""
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from io import StringIO

from fastapi import FastAPI, HTTPException, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from src.credit_default.pipeline.prediction_pipeline import PredictionPipeline
from src.credit_default.exception import PredictionException
from src.credit_default.logger import logger
from src.credit_default.utils import validate_input_data, read_yaml
from src.credit_default.constants import CONFIG_DIR


# Pydantic models for API
class PredictionRequest(BaseModel):
    """Request model for single prediction"""
    LIMIT_BAL: int = Field(..., description="Credit limit", ge=10000, le=1000000)
    SEX: int = Field(..., description="Gender (1=male, 2=female)", ge=1, le=2)
    EDUCATION: int = Field(..., description="Education level", ge=1, le=6)
    MARRIAGE: int = Field(..., description="Marital status", ge=1, le=3)
    AGE: int = Field(..., description="Age in years", ge=18, le=100)
    PAY_0: int = Field(..., description="Payment status in September", ge=-2, le=8)
    PAY_2: int = Field(..., description="Payment status in August", ge=-2, le=8)
    PAY_3: int = Field(..., description="Payment status in July", ge=-2, le=8)
    PAY_4: int = Field(..., description="Payment status in June", ge=-2, le=8)
    PAY_5: int = Field(..., description="Payment status in May", ge=-2, le=8)
    PAY_6: int = Field(..., description="Payment status in April", ge=-2, le=8)
    BILL_AMT1: float = Field(..., description="Bill amount in September")
    BILL_AMT2: float = Field(..., description="Bill amount in August")
    BILL_AMT3: float = Field(..., description="Bill amount in July")
    BILL_AMT4: float = Field(..., description="Bill amount in June")
    BILL_AMT5: float = Field(..., description="Bill amount in May")
    BILL_AMT6: float = Field(..., description="Bill amount in April")
    PAY_AMT1: float = Field(..., description="Payment amount in September", ge=0)
    PAY_AMT2: float = Field(..., description="Payment amount in August", ge=0)
    PAY_AMT3: float = Field(..., description="Payment amount in July", ge=0)
    PAY_AMT4: float = Field(..., description="Payment amount in June", ge=0)
    PAY_AMT5: float = Field(..., description="Payment amount in May", ge=0)
    PAY_AMT6: float = Field(..., description="Payment amount in April", ge=0)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int = Field(..., description="Prediction (0=no default, 1=default)")
    probability: float = Field(..., description="Probability of default")
    risk_score: float = Field(..., description="Risk score (0-100)")
    risk_category: str = Field(..., description="Risk category")
    explanation: Optional[Dict[str, Any]] = Field(None, description="SHAP explanation")


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    total_predictions: int = Field(..., description="Total number of predictions")
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    summary_stats: Dict[str, Any] = Field(..., description="Summary statistics")


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str
    message: str
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    """Response model for model information"""
    model_loaded: bool
    model_type: Optional[str] = None
    features_count: Optional[int] = None
    explainer_available: bool = False


# Initialize FastAPI app
app = FastAPI(
    title="Credit Default Prediction API",
    description="End-to-end ML API for credit default prediction with explainable AI",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
prediction_pipeline = None
schema_config = None


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup"""
    global prediction_pipeline, schema_config
    try:
        # Load prediction pipeline
        prediction_pipeline = PredictionPipeline()
        logger.info("Prediction pipeline loaded successfully")

        # Load schema configuration
        schema_path = CONFIG_DIR / "schema.yaml"
        if schema_path.exists():
            schema_config = read_yaml(schema_path)
            logger.info("Schema configuration loaded")

        logger.info("FastAPI application started successfully")

    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise e


def get_prediction_pipeline():
    """Dependency to get prediction pipeline"""
    if prediction_pipeline is None:
        raise HTTPException(status_code=503, detail="Prediction pipeline not initialized")
    return prediction_pipeline


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint"""
    return HealthResponse(
        status="active",
        message="Credit Default Prediction API is running"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        pipeline = get_prediction_pipeline()
        return HealthResponse(
            status="healthy",
            message="All services are operational"
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy", 
            message=f"Service error: {str(e)}"
        )


@app.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info(pipeline: PredictionPipeline = Depends(get_prediction_pipeline)):
    """Get model information"""
    try:
        return ModelInfoResponse(
            model_loaded=True,
            model_type=type(pipeline.model).__name__,
            features_count=pipeline.preprocessor.n_features_in_ if hasattr(pipeline.preprocessor, 'n_features_in_') else None,
            explainer_available=pipeline.explainer is not None
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(
    request: PredictionRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline)
):
    """Make prediction for single instance"""
    try:
        # Convert request to dictionary
        input_data = request.dict()

        # Validate input data
        if schema_config:
            is_valid = validate_input_data(input_data, schema_config)
            if not is_valid:
                raise HTTPException(status_code=400, detail="Input data validation failed")

        # Make prediction
        result = pipeline.predict(input_data)

        # Create response
        response = PredictionResponse(
            prediction=result.prediction,
            probability=result.probability,
            risk_score=result.risk_score,
            risk_category=result.risk_category,
            explanation=result.shap_explanation
        )

        logger.info(f"Prediction completed: {result.risk_category}")
        return response

    except PredictionException as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# @app.post("/explain", response_model=PredictionResponse)
# async def explain_prediction(
#     request: PredictionRequest,
#     pipeline: PredictionPipeline = Depends(get_prediction_pipeline)
# ):
#     """Get prediction with detailed explanation"""
#     try:
#         if pipeline.explainer is None:
#             raise HTTPException(status_code=503, detail="Model explainer not available")

#         # Convert request to dictionary
#         input_data = request.dict()

#         # Make prediction with explanation
#         result = pipeline.predict(input_data)

#         # Create response with explanation
#         response = PredictionResponse(
#             prediction=result.prediction,
#             probability=result.probability,
#             risk_score=result.risk_score,
#             risk_category=result.risk_category,
#             explanation=result.shap_explanation
#         )

#         return response

#     except Exception as e:
#         logger.error(f"Explanation error: {e}")
#         raise HTTPException(status_code=500, detail=str(e))

import shap
import pickle

# Load SHAP explainer at startup
explainer = None
try:
    with open('artifacts/explainer/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
except:
    print("SHAP explainer not found")

@app.post("/explain", response_model=PredictionResponse)
async def explain_prediction(
    request: PredictionRequest,
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline)
):
    """Get SHAP explanation for a prediction"""
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])

        # Preprocess
        X_processed = pipeline.preprocessor.transform(input_data)

        # Get prediction
        prediction_prob = pipeline.model.predict_proba(X_processed)[0][1]

        # Get SHAP explanation if available
        explanation = None
        if explainer is not None:
            shap_values = explainer(X_processed)
            if hasattr(shap_values, 'values'):
                explanation = {
                    "shap_values": shap_values.values[0].tolist(),
                    "feature_names": pipeline.feature_names,
                    "base_value": float(explainer.expected_value[1]) if hasattr(explainer, 'expected_value') else 0.0
                }

        return PredictionResponse(
            prediction=1 if prediction_prob > 0.5 else 0,
            probability=float(prediction_prob),
            risk_score=float(prediction_prob * 100),  # Assuming risk score is a scaled probability
            risk_category="High" if prediction_prob > 0.7 else "Medium" if prediction_prob > 0.4 else "Low",
            explanation=explanation
        )

    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict(
    file: UploadFile = File(...),
    pipeline: PredictionPipeline = Depends(get_prediction_pipeline)
):
    """Make batch predictions from CSV file"""
    try:
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")

        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(StringIO(contents.decode('utf-8')))

        logger.info(f"Processing batch prediction for {len(df)} records")

        # Make batch predictions
        results_df = pipeline.batch_predict(df)

        # Convert results to list of dictionaries
        predictions = results_df.to_dict('records')

        # Calculate summary statistics
        summary_stats = {
            'total_records': len(results_df),
            'default_predictions': int(results_df['prediction'].sum()),
            'no_default_predictions': int(len(results_df) - results_df['prediction'].sum()),
            'avg_risk_score': float(results_df['risk_score'].mean()),
            'high_risk_count': int((results_df['risk_category'] == 'High Risk').sum()),
            'medium_risk_count': int((results_df['risk_category'] == 'Medium Risk').sum()),
            'low_risk_count': int((results_df['risk_category'] == 'Low Risk').sum())
        }

        response = BatchPredictionResponse(
            total_predictions=len(predictions),
            predictions=predictions,
            summary_stats=summary_stats
        )

        logger.info(f"Batch prediction completed: {len(predictions)} records processed")
        return response

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/sample-prediction")
async def get_sample_prediction(pipeline: PredictionPipeline = Depends(get_prediction_pipeline)):
    """Generate a sample prediction for testing"""
    try:
        # Sample input data
        sample_data = {
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
        result = pipeline.predict(sample_data)

        return {
            'input_data': sample_data,
            'prediction_result': {
                'prediction': result.prediction,
                'probability': result.probability,
                'risk_score': result.risk_score,
                'risk_category': result.risk_category
            }
        }

    except Exception as e:
        logger.error(f"Sample prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/feature-schema")
async def get_feature_schema():
    """Get the expected input feature schema"""
    try:
        if schema_config is None:
            raise HTTPException(status_code=503, detail="Schema configuration not available")

        return {
            "features": schema_config.columns,
            "target": schema_config.target,
            "data_quality_thresholds": schema_config.get("data_quality", {})
        }

    except Exception as e:
        logger.error(f"Error getting feature schema: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "fastapi_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
