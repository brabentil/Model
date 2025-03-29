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

@app.post("/predict/raw")
async def predict_raw_transaction(transaction: RawTransactionData):
    """
    Predict fraud probability based on raw transaction data
    
    This endpoint accepts raw transaction details, transforms them to the required feature format,
    and returns a fraud prediction.
    """
    try:
        logging.info(f"Received raw transaction: {transaction.dict()}")
        
        # Transform raw transaction into model-compatible features
        transformed_data = transformer.transform_raw_transaction(transaction.dict())
        logging.info(f"Transformed data: {transformed_data}")
        
        # Convert the dictionary to the format expected by the model
        features_array = np.array([
            transformed_data["Time"],
            transformed_data["V1"],
            transformed_data["V2"],
            transformed_data["V3"],
            transformed_data["V4"],
            transformed_data["V5"],
            transformed_data["V6"],
            transformed_data["V7"],
            transformed_data["V8"],
            transformed_data["V9"],
            transformed_data["V10"],
            transformed_data["V11"],
            transformed_data["V12"],
            transformed_data["V13"],
            transformed_data["V14"],
            transformed_data["V15"],
            transformed_data["V16"],
            transformed_data["V17"],
            transformed_data["V18"],
            transformed_data["V19"],
            transformed_data["V20"],
            transformed_data["V21"],
            transformed_data["V22"],
            transformed_data["V23"],
            transformed_data["V24"],
            transformed_data["V25"],
            transformed_data["V26"],
            transformed_data["V27"],
            transformed_data["V28"],
            transformed_data["Amount"]
        ]).reshape(1, -1)
        
        # Apply scaler if available
        if 'scaler' in globals() and scaler is not None:
            features_array = scaler.transform(features_array)
        
        # Make prediction
        fraud_probability = float(model.predict_proba(features_array)[0, 1])
        is_fraud = bool(fraud_probability > 0.5)
        
        # Return prediction result
        return {
            "prediction": {
                "is_fraud": is_fraud,
                "fraud_probability": fraud_probability,
                "risk_level": get_risk_level(fraud_probability)
            },
            "transaction_id": transaction.dict().get("transaction_id", "unknown"),
            "processed_at": pd.Timestamp.now().isoformat(),
            "feature_importance": get_top_features_for_prediction(features_array)
        }
        
    except Exception as e:
        logging.error(f"Error predicting: {str(e)}")
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
    # This is a simplified example - you would need to implement
    # feature importance calculation based on your model
    feature_names = ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", 
                     "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", 
                     "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", 
                     "V25", "V26", "V27", "V28", "Amount"]
    
    # Placeholder for feature importance - this should be adapted to your model
    # For example, for tree-based models you might use model.feature_importances_
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
