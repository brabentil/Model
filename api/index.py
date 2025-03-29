from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List
import pandas as pd  # Add pandas import

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API")

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "fraud_detection_model.pkl")

try:
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
    # Define feature names that match what was used during training
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    feature_names = None

# Define request data model
class TransactionData(BaseModel):
    features: List[float]

@app.get("/")
def read_root():
    return {"message": "Credit Card Fraud Detection API is running!"}

@app.get("/health")
def health_check():
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/predict")
async def predict(data: TransactionData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Convert features to numpy array
    features = np.array(data.features).reshape(1, -1)
    
    # Make prediction
    try:
        # Convert to DataFrame with feature names to avoid the warning
        features_df = pd.DataFrame(features, columns=feature_names)
        
        prediction = model.predict(features_df)[0]
        probability = model.predict_proba(features_df)[0][1]
        
        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "is_fraud": bool(prediction == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")
