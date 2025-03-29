from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys
import os
import logging
from typing import List

# Basic logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print diagnostic information
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Directory contents: {os.listdir('.')}")

# Initialize FastAPI app
app = FastAPI(title="Fallback Credit Card Fraud API")

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Fallback API is running", "status": "model not available"}

@app.get("/health")
def health_check():
    return {
        "status": "limited",
        "message": "This is a fallback API that doesn't require external dependencies",
        "python_version": sys.version,
        "environment": dict(os.environ)
    }

@app.post("/predict")
async def predict(data: PredictionInput):
    """
    Fallback prediction that always returns 0 (non-fraud)
    This is just to ensure the API endpoint works even if dependencies fail
    """
    features = np.array(data.features)
    
    # Log incoming request
    logger.info(f"Received prediction request with {len(features)} features")
    
    return {
        "prediction": 0,
        "fraud_probability": 0.01,
        "is_fraud": False,
        "note": "This is a fallback response as the actual model could not be loaded"
    }
