from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sys
import os
import logging
from typing import List
from datetime import datetime

# Add the parent directory to path so we can import the TransactionTransformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from transaction_transformer import TransactionTransformer
    transformer = TransactionTransformer()
    transformer_available = True
except ImportError:
    transformer_available = False
    logging.warning("TransactionTransformer not available in fallback API")

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

class RawTransactionInput(BaseModel):
    """Simple model for raw transaction data"""
    card_number: str = None
    transaction_date: str = None
    transaction_amount: float
    merchant_category: str = None
    merchant_name: str = None
    # Allow for additional fields
    class Config:
        extra = "allow"

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

@app.post("/predict/raw")
async def predict_raw_transaction(transaction: RawTransactionInput):
    """Predict fraud probability based on raw transaction data using fallback logic"""
    try:
        # Use the transformer if available
        if transformer_available:
            transformed_data = transformer.transform_raw_transaction(transaction.dict())
            # In the fallback API, we'll just return a simple probability
            # based on some heuristics from the transformed features
            
            # Check if certain high-risk features are present
            fraud_indicators = [
                transformed_data["V1"] < -1.0,
                transformed_data["V3"] < -0.5,
                transformed_data["V4"] < -0.5,
                transformed_data["Amount"] > 200
            ]
            
            # Calculate probability based on number of indicators
            fraud_probability = min(0.95, sum(fraud_indicators) * 0.25)
            
        else:
            # If transformer isn't available, use simplified fallback logic
            amount = transaction.transaction_amount
            # Simple heuristic - higher amounts have higher fraud risk
            fraud_probability = min(0.95, amount / 1000)
        
        # Return prediction result
        return {
            "prediction": {
                "is_fraud": fraud_probability > 0.5,
                "fraud_probability": fraud_probability,
                "risk_level": "high" if fraud_probability > 0.7 else 
                             ("medium" if fraud_probability > 0.3 else "low")
            },
            "note": "This is a fallback prediction with limited accuracy",
            "processed_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logging.error(f"Error in fallback prediction: {str(e)}")
        # Even in case of error, provide a safe default response
        return {
            "prediction": {
                "is_fraud": False,
                "fraud_probability": 0.1,
                "risk_level": "low"
            },
            "note": "Error occurred, returning safe default prediction",
            "error": str(e)
        }
