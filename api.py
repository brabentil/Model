from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Dict, Any, Optional
from transaction_transformer import TransactionTransformer
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API", 
              description="API for detecting fraudulent credit card transactions",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define data models
class RawTransactionData(BaseModel):
    amount: float
    timestamp: str
    merchant_name: str
    merchant_category: Optional[str] = None
    is_online: Optional[bool] = False
    card_present: Optional[bool] = True
    country: Optional[str] = None
    unusual_location: Optional[bool] = False
    high_frequency: Optional[bool] = False
    # Add any other raw transaction fields you might have

class TransactionData(BaseModel):
    features: List[float]

# Load model and scaler on startup
@app.on_event("startup")
async def load_model():
    global model, scaler, transformer, feature_names
    model_path = os.path.join(os.path.dirname(__file__), "model", "fraud_detection_model.pkl")
    scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
    
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        transformer = TransactionTransformer()
        # Define feature names that match what was used during training
        feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    except Exception as e:
        print(f"Error loading model or scaler: {str(e)}")
        model = None
        scaler = None
        transformer = None
        feature_names = None

@app.get("/")
async def root():
    return {"message": "Credit Card Fraud Detection API is running"}

@app.post("/predict")
async def predict(data: TransactionData):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded")
    
    try:
        # Convert features to numpy array and reshape for prediction
        features = np.array(data.features).reshape(1, -1)
        
        # Check if dimensions match
        if features.shape[1] != 30:  # Assuming 30 features (Time + V1-V28 + Amount)
            return {"error": f"Expected 30 features, got {features.shape[1]}"}
        
        # Convert to DataFrame with feature names to avoid the warning
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)[:, 1]
        
        return {
            "prediction": int(prediction[0]),
            "fraud_probability": float(prediction_proba[0]),
            "is_fraud": bool(prediction[0] == 1)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/raw")
async def predict_raw_transaction(transaction: RawTransactionData):
    """
    Endpoint that accepts raw transaction data and transforms it
    into the format expected by the model
    """
    if model is None or scaler is None or transformer is None:
        raise HTTPException(status_code=500, detail="Model, scaler or transformer not loaded")
    
    try:
        # Convert transaction to dictionary
        transaction_dict = transaction.dict()
        
        # Transform raw transaction into features
        features = transformer.transform_raw_transaction(transaction_dict)
        
        # Reshape for prediction
        features = features.reshape(1, -1)
        
        # Convert to DataFrame with feature names to avoid the warning
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)
        prediction_proba = model.predict_proba(features_scaled)[:, 1]
        
        # Get accuracy estimate
        accuracy_info = transformer.get_accuracy_estimate()
        
        return {
            "prediction": int(prediction[0]),
            "fraud_probability": float(prediction_proba[0]),
            "is_fraud": bool(prediction[0] == 1),
            "transaction_amount": transaction.amount,
            "merchant": transaction.merchant_name,
            "accuracy_info": accuracy_info,
            "warning": "This prediction is based on transformed data and may have reduced accuracy"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
