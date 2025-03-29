from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "fraud_detection_model.pkl")
SCALER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "scaler.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    # Define feature names that match what was used during training
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Try to load the scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded successfully from {SCALER_PATH}")
    except Exception as e:
        print(f"Warning: Could not load scaler: {e}")
        scaler = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = None
    scaler = None

# Define request data model
class TransactionData(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running!"}

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True, "scaler_loaded": scaler is not None}
    return {"status": "unhealthy", "model_loaded": False}

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
        
        # Convert to DataFrame with feature names to avoid the warning
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Scale features if scaler is available
        if scaler is not None:
            features_to_predict = scaler.transform(features_df)
        else:
            # If no scaler, just use the raw features (assuming they're already normalized)
            print("Warning: Using unscaled features for prediction as scaler is not available")
            features_to_predict = features_df
        
        # Make prediction
        prediction = model.predict(features_to_predict)[0]
        probability = model.predict_proba(features_to_predict)[0][1]
        
        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
