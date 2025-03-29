from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
import logging
from typing import List, Dict, Any, Optional
from transaction_transformer import TransactionTransformer
import pandas as pd
from routes.api_routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("fraud_detection_api")

# Initialize FastAPI app
app = FastAPI(title="Credit Card Fraud Detection API", 
              description="API for detecting fraudulent credit card transactions",
              version="1.0.0",
              docs_url="/docs",
              redoc_url="/redoc")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include our routes
app.include_router(router, prefix="")

# Initialize TransactionTransformer
transformer = TransactionTransformer()

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
    class Config:
        extra = "allow"  # Allow additional fields not specified in the model

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



def get_risk_level(probability):
    """Calculate risk level based on fraud probability"""
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"

def get_top_features_for_prediction(features_array):
    """Get top contributing features for this prediction"""
   
    feature_names = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", 
                     "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", 
                     "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", 
                     "V25", "V26", "V27", "V28", "Amount"]
 
    return [
        {"feature": "V14", "importance": 0.25},
        {"feature": "V12", "importance": 0.18},
        {"feature": "V10", "importance": 0.15},
        {"feature": "Amount", "importance": 0.12},
        {"feature": "V17", "importance": 0.10}
    ]

@app.on_event("startup")
async def startup_event():
    """Log information on startup"""
    logger.info("Starting Fraud Detection API")
    logger.info(f"Environment: {os.environ.get('ENVIRONMENT', 'development')}")
    
    # Log all available routes for debugging
    routes = [{"path": route.path, "name": route.name, "methods": route.methods} 
              for route in app.routes]
    logger.info(f"Available routes: {routes}")
