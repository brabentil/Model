from controllers.prediction_controller import PredictionController
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from models.data_models import DetailedPredictionResponse, PredictedResponse, PredictionResponse, RawTransactionRequest
from pydantic import BaseModel
import sys
import os
from typing import Dict, Any, List
import numpy as np
import joblib
import logging

from routes.api_routes import get_controller

# Add the parent directory to path so we can import from the Model directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model components - you'll need to adjust these imports based on your actual model structure
try:
    from model import predict  # Adjust this import based on your actual model structure
except ImportError:
    print("Warning: Could not import model module. Make sure your model is properly configured.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print Python version and environment info for debugging
logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.executable}")

# APP SWAGGER UI HERE
# Initialize FastAPI app
app = FastAPI(title="AegisX Model API")

# Define paths
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "fraud_detection_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "scaler.pkl")

# Log paths for debugging
logger.info(f"Model path: {MODEL_PATH}")
logger.info(f"Scaler path: {SCALER_PATH}")

# Initialize global variables
model = None
scaler = None
feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

# Try to import pandas, but provide a fallback if it's not available
try:
    import pandas as pd
    logger.info("Successfully imported pandas")
    PANDAS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import pandas: {e}")
    PANDAS_AVAILABLE = False

# Load model and scaler
try:
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
    
    # Try to load the scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        logger.info(f"Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        logger.warning(f"Could not load scaler: {e}")
        scaler = None
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class PredictionInput(BaseModel):
    features: Dict[str, Any]

class PredictionOutput(BaseModel):
    prediction: Any
    confidence: float = None

@app.get("/")
def read_root():
    return {"status": "API is running", "model": "Aegis Model"}

@app.post("/predict", response_model=PredictionOutput)
async def make_prediction(input_data: PredictionInput):
    try:
        # Adjust this based on your model's prediction function
        # result = predict(input_data.features)
        print(input_data.features)
        result = PredictionController().predict_from_features(input_data.features)
        # print the result
        logger.info(f"Prediction result: {result}")
        
        # Modify this based on your model's output structure
        return {
            "prediction": result["fraud_prediction"] if isinstance(result, dict) else result,
            "confidence": result.get("probability") if isinstance(result, dict) else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
@app.post("/raw", response_model=PredictedResponse)
async def predict_raw(request: RawTransactionRequest, controller: PredictionController = Depends(get_controller)):
    """Make prediction from raw transaction data"""
    try:
        logger.info(f"Processing raw transaction request: {request}")
        result = controller.predict_from_raw_transaction(request.dict())
        logger.info(f"Raw prediction result: {result}")
        
        if "error" in result:
            logger.error(f"Error in raw prediction: {result['error']}")
            return JSONResponse(status_code=400, content=result)
        
        return {
            "fraud_prediction": result["fraud_prediction"],
            "probability": result["probability"],
        }
    except Exception as e:
        logger.error(f"Error in /raw endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Raw prediction error: {str(e)}")


@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "pandas_available": PANDAS_AVAILABLE,
        "python_version": sys.version
    }
