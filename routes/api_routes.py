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
    TestResponse,
    TransformResponse
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

@router.post("/transform", response_model=TransformResponse)
async def transform_transaction(request: RawTransactionRequest, controller: PredictionController = Depends(get_controller)):
    """Transform raw transaction data into feature vector without making predictions"""
    try:
        # Log the incoming request
        logger.info(f"Processing transformation request: {request.dict(exclude={'card_number'})}")
        
        # Add a debug log to verify controller is available
        logger.debug(f"Controller status: transformer={controller.transformer is not None}, model={controller.model is not None}")
        
        # Call the controller to transform the data
        result = controller.transform_raw_transaction(request.dict())
        
        # Handle errors
        if "error" in result:
            logger.error(f"Error in transformation: {result['error']}")
            return JSONResponse(status_code=400, content=result)
        
        # Log success
        logger.info(f"Successfully transformed transaction: {result.get('transaction_id')}")
        logger.debug(f"Feature count: {len(result.get('features', []))}")
        
        return result
    except Exception as e:
        import traceback
        stack_trace = traceback.format_exc()
        logger.error(f"Error in /transform endpoint: {str(e)}\n{stack_trace}")
        raise HTTPException(status_code=500, detail=f"Transformation error: {str(e)}")

# Add a simple GET endpoint for checking if transform is mapped
@router.get("/transform/status", response_model=Dict[str, str])
async def transform_status():
    """Check if the transform endpoint is properly mapped"""
    return {
        "status": "available",
        "message": "The transform endpoint is properly mapped",
        "endpoint": "/transform (POST)",
        "timestamp": datetime.now().isoformat()
    }

@router.post("/pipeline", response_model=DetailedPredictionResponse)
async def pipeline(request: RawTransactionRequest, controller: PredictionController = Depends(get_controller)):
    """Process raw transaction through full pipeline - transform and predict in one step"""
    try:
        logger.info(f"Processing transaction through full pipeline")
        result = controller.process_transaction_pipeline(request.dict())
        
        if "error" in result:
            logger.error(f"Error in pipeline processing: {result['error']}")
            return JSONResponse(status_code=400, content=result)
        
        return result
    except Exception as e:
        logger.error(f"Error in /pipeline endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Pipeline processing error: {str(e)}")

@router.post("/test", response_model=TestResponse)
async def test_endpoint():
    """Simple test endpoint that always succeeds"""
    return {
        "success": True,
        "message": "API is working correctly",
        "timestamp": datetime.now().isoformat()
    }
