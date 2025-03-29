from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print Python version and environment info for debugging
logger.info(f"Python version: {sys.version}")
logger.info(f"Python path: {sys.executable}")

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

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

# Define request data model
class TransactionData(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running!"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "pandas_available": PANDAS_AVAILABLE,
        "python_version": sys.version
    }

@app.post("/predict")
async def predict(data: TransactionData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        features = np.array(data.features).reshape(1, -1)
        
        # Check dimensions
        if features.shape[1] != 30:  # Assuming 30 features
            raise HTTPException(status_code=400, detail=f"Expected 30 features, got {features.shape[1]}")
        
        # Use pandas if available for feature names
        if PANDAS_AVAILABLE:
            features_df = pd.DataFrame(features, columns=feature_names)
            
            # Scale features if scaler is available
            if scaler is not None:
                features_to_predict = scaler.transform(features_df)
            else:
                logger.warning("Using unscaled features as scaler is not available")
                features_to_predict = features_df
        else:
            # Fallback if pandas is not available
            logger.warning("Using raw numpy arrays as pandas is not available")
            features_to_predict = features if scaler is None else scaler.transform(features)
        
        # Make prediction
        prediction = model.predict(features_to_predict)[0]
        probability = model.predict_proba(features_to_predict)[0][1]
        
        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
