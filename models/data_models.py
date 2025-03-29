from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class RawTransactionRequest(BaseModel):
    """Raw transaction data coming from client systems"""
    amount: float = Field(..., description="Transaction amount")
    timestamp: Optional[str] = Field(None, description="Transaction timestamp")
    merchant_name: Optional[str] = Field(None, description="Merchant name")
    merchant_category: Optional[str] = Field(None, description="Merchant category code")
    is_online: Optional[bool] = Field(False, description="Whether transaction was online")
    card_present: Optional[bool] = Field(True, description="Whether physical card was used")
    country: Optional[str] = Field(None, description="Country code of transaction")
    unusual_location: Optional[bool] = Field(False, description="Flag for unusual location")
    high_frequency: Optional[bool] = Field(False, description="Flag for high transaction frequency")
    
    class Config:
        extra = "allow"  # Allow additional fields

class FeatureVectorRequest(BaseModel):
    """Pre-processed feature vector for direct prediction"""
    features: List[float] = Field(..., description="List of features (Time, V1-V28, Amount)")

class PredictionResponse(BaseModel):
    """Standard prediction response"""
    is_fraud: bool = Field(..., description="Whether transaction is predicted as fraud")
    fraud_probability: float = Field(..., description="Probability of fraud (0-1)")
    risk_level: str = Field(..., description="Risk level (low/medium/high)")

class DetailedPredictionResponse(BaseModel):
    """Detailed prediction response with metadata"""
    prediction: PredictionResponse
    transaction_id: str = Field("unknown", description="Transaction ID if provided")
    processed_at: str = Field(..., description="Timestamp when prediction was made")
    feature_importance: Optional[List[Dict[str, Any]]] = Field(None, description="Top contributing features")
    model_version: Optional[str] = Field(None, description="Model version used")
    note: Optional[str] = Field(None, description="Additional information")

class PredictedResponse(BaseModel):
    """Response model for the prediction endpoint"""
    fraud_prediction: str = Field(..., description="Whether transaction is predicted as fraud")
    probability: float = Field(..., description="Probability of fraud (0-1)")


class HealthResponse(BaseModel):
    """API health check response"""
    status: str
    model_loaded: bool
    scaler_loaded: bool
    uptime_seconds: float
    version: str = "1.0.0"
    environment: Optional[str] = None

class TestResponse(BaseModel):
    """Simple test response"""
    success: bool
    message: str
    timestamp: str

class TransformResponse(BaseModel):
    """Response model for the transform endpoint"""
    features: List[float] = Field(..., description="Transformed feature vector")
    feature_names: List[str] = Field(..., description="Names of the features")
    timestamp: str = Field(..., description="Timestamp of the transformation")
    transaction_id: str = Field(..., description="Unique identifier for this transaction")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [0, -1.3598071336738, -0.0727811733098497, 2.53634673796914, 1.37815522427443, -0.338320769942518, 0.462387777762292, 0.239598554061257, 0.0986979012610507, 0.363786969611213, 0.0908226130794177, -0.551599533260813, -0.617800855762348, -0.991389847235408, -0.311169353699879, 1.46817697209427, -0.470400525259478, 0.207971241929242, 0.0257905801985591, 0.403992960255733, 0.251412098239705, -0.018306777944153, 0.277837575558899, -0.110473910188767, 0.0669280749146731, 0.128539358273528, -0.189114843888824, 0.133558376740387, -0.0210530534538215, 100], 
                "feature_names": ["Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"],
                "timestamp": "2023-08-02T15:30:45.123456",
                "transaction_id": "550e8400-e29b-41d4-a716-446655440000"
            }
        }
