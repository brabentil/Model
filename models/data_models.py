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
