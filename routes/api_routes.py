from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Dict, Any

from models.data_models import (
    RawTransactionRequest, 
    FeatureVectorRequest,
    PredictionResponse, 
    DetailedPredictionResponse,
    HealthResponse,
    TestResponse
)
from controllers.prediction_controller import PredictionController

# Configure logging
logger = logging.getLogger("api_routes")

router = APIRouter()

# Create a single controller instance for reuse
prediction_controller = PredictionController()

def get_controller():
    """Dependency to get the prediction controller"""
    return prediction_controller

@router.get("/", response_model=Dict[str, str])
async def root():
    """API root endpoint"""
    return {"message": "Fraud Detection API is running. See /docs for endpoints."}

@router.get("/health", response_model=HealthResponse)
async def health(controller: PredictionController = Depends(get_controller)):
    """API health check"""
    return controller.get_health_status()

@router.post("/predict", response_model=DetailedPredictionResponse)
async def predict(request: FeatureVectorRequest, controller: PredictionController = Depends(get_controller)):
    """Make prediction from pre-processed feature vector"""
    try:
        result = controller.predict_from_features(request.features)
        if "error" in result:
            return JSONResponse(status_code=400, content=result)
        return result
    except Exception as e:
        logger.error(f"Error in /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.post("/raw", response_model=DetailedPredictionResponse)
async def predict_raw(request: RawTransactionRequest, controller: PredictionController = Depends(get_controller)):
    """Make prediction from raw transaction data"""
    try:
        logger.info(f"Processing raw transaction request: {request}")
        result = controller.predict_from_raw_transaction(request.dict())
        logger.info(f"Raw prediction result: {result}")
        
        if "error" in result:
            logger.error(f"Error in raw prediction: {result['error']}")
            return JSONResponse(status_code=400, content=result)
        
        return result
    except Exception as e:
        logger.error(f"Error in /raw endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Raw prediction error: {str(e)}")

@router.post("/test", response_model=TestResponse)
async def test_endpoint():
    """Simple test endpoint that always succeeds"""
    return {
        "success": True,
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat()
    }
