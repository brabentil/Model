from api.fallback import Transaction
import numpy as np
import pandas as pd
import logging
import os
import joblib
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional, Union

from transaction_transformer import TransactionTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("prediction_controller")


class PredictionController:
    """Controller for handling fraud predictions"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.transformer = None
        self.feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
        self.start_time = datetime.now()
        self.model_version = "1.0.0"
        self.load_dependencies()
    
    def load_dependencies(self):
        """Load model, scaler and transformer"""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "fraud_detection_model.pkl")
            scaler_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "scaler.pkl")
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.transformer = TransactionTransformer()
            logger.info("Successfully loaded model, scaler and transformer")
        except Exception as e:
            logger.error(f"Error loading dependencies: {e}")
            # Don't raise - we'll handle missing dependencies gracefully
    
    def get_health_status(self) -> Dict[str, Any]:
        """Check health of prediction services"""
        uptime = (datetime.now() - self.start_time).total_seconds()
        return {
            "status": "healthy" if self.model and self.scaler else "limited",
            "model_loaded": self.model is not None,
            "scaler_loaded": self.scaler is not None,
            "uptime_seconds": uptime,
            "version": self.model_version,
            "environment": os.getenv("ENVIRONMENT", "development")
        }
    
    def predict_from_features(self, features: Union[List[float], pd.DataFrame]) -> Dict[str, Any]:
        """Make prediction from pre-processed feature vector"""
        if self.model is None:
            return self._get_fallback_prediction()
            
        try:
            # Handle different input types
            if isinstance(features, pd.DataFrame):
                # If already a DataFrame, use it directly
                features_df = features
                # Ensure it has the right number of columns
                if len(features_df.columns) != 30:
                    logger.warning(f"Expected 30 features, got {len(features_df.columns)}")
                    return {
                        "error": f"Expected 30 features, got {len(features_df.columns)}",
                        "prediction": self._get_fallback_prediction()["prediction"]
                    }
            else:
                # Convert list to array then to DataFrame
                # Validate feature count
                if len(features) != 30:
                    logger.warning(f"Expected 30 features, got {len(features)}")
                    return {
                        "error": f"Expected 30 features, got {len(features)}",
                        "prediction": self._get_fallback_prediction()["prediction"]
                    }
                
                features_array = np.array(features).reshape(1, -1)
                features_df = pd.DataFrame(features_array, columns=self.feature_names)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features_df)
            else:
                features_scaled = features_df.values
            
            # Make prediction
            prediction = bool(self.model.predict(features_scaled)[0] == 1)
            probability = float(self.model.predict_proba(features_scaled)[0, 1])
            risk_level = self._get_risk_level(probability)
            
            return {
                "prediction": {
                    "is_fraud": prediction,
                    "fraud_probability": probability,
                    "risk_level": risk_level
                },
                "processed_at": datetime.now().isoformat(),
                "model_version": self.model_version,
                "feature_importance": self._get_feature_importance(features_scaled)
            }
            
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Error making prediction from features: {e}\n{stack_trace}")
            return {
                "error": str(e),
                "prediction": self._get_fallback_prediction()["prediction"]
            }
    
    def predict_from_raw_transaction(self, transaction_data: dict) -> Dict[str, Any]:
        """Make prediction from raw transaction data"""
        try:
            # Transform raw transaction to features
            if self.transformer is None:
                self.transformer = TransactionTransformer()
                
            # Log the transaction with sensitive data removed
            safe_transaction = {**transaction_data}
            if 'card_number' in safe_transaction:
                safe_transaction['card_number'] = '****' + safe_transaction['card_number'][-4:]
            logger.info(f"Processing transaction: {safe_transaction}")
            
            # Transform the transaction
            transaction = self.transformer.transform_raw_transaction(transaction_data)

            # Fix: Pass the transaction as a positional argument, not as a keyword argument
            result = PredictionController().predict_from_features(transaction)
            # print the result
            logger.info(f"Prediction result: {result}")
            
            # Modify this based on your model's output structure
            return {
                "fraud_prediction": result["fraud_prediction"] if isinstance(result, dict) else result,
                "probability": result.get("probability") if isinstance(result, dict) else None
            }
            
        except Exception as e:
            logger.error(f"Error processing raw transaction: {str(e)}", exc_info=True)
            return {"error": f"Failed to process transaction: {str(e)}"}
    
    def transform_raw_transaction(self, transaction_data: dict) -> dict:
        """
        Transform raw transaction data into feature vector without making predictions
        
        Args:
            transaction_data: Dictionary containing raw transaction fields
            
        Returns:
            Dictionary with transformed features that can be passed to predict endpoint
        """
        try:
            # Log incoming transaction data for debugging
            safe_transaction = {k: v for k, v in transaction_data.items() if k != 'card_number'}
            logger.info(f"Transforming transaction: {safe_transaction}")
            
            # Ensure transformer is available
            if self.transformer is None:
                logger.info("Creating new transformer instance")
                self.transformer = TransactionTransformer()
            
            # Get the raw result from transformer
            raw_features = self.transformer.transform_raw_transaction(transaction_data)
            logger.debug(f"Raw features from transformer: {raw_features}")
            
            # Handle different return types from transformer
            if isinstance(raw_features, dict):
                # If it's a dictionary, extract features in correct order
                features = [
                    raw_features.get("Time", 0),
                    *[raw_features.get(f"V{i}", 0) for i in range(1, 29)],
                    raw_features.get("Amount", 0)
                ]
            elif isinstance(raw_features, (list, np.ndarray)):
                # If it's already a list or array, use it directly
                features = raw_features
            else:
                raise TypeError(f"Unexpected feature type: {type(raw_features)}")
            
            # Convert to numpy array for validation
            features_np = np.array(features).reshape(1, -1)
            logger.info(f"Feature array shape: {features_np.shape}")
            
            # Ensure we have the expected number of features
            if features_np.shape[1] != len(self.feature_names):
                error_msg = f"Feature mismatch: got {features_np.shape[1]} features, expected {len(self.feature_names)}"
                logger.error(error_msg)
                return {"error": error_msg}
            
            # Return features as a list (to match the API format)
            return {
                "features": features_np[0].tolist(),
                "feature_names": self.feature_names,
                "timestamp": datetime.now().isoformat(),
                "transaction_id": str(uuid.uuid4())
            }
        except Exception as e:
            stack_trace = traceback.format_exc()
            logger.error(f"Error transforming transaction: {str(e)}\n{stack_trace}")
            return {"error": f"Failed to transform transaction: {str(e)}"}
    
    def process_transaction_pipeline(self, transaction_data: dict) -> dict:
        """
        Process raw transaction through complete pipeline - transform and predict in one step
        
        Args:
            transaction_data: Dictionary containing raw transaction fields
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Step 1: Transform raw transaction to features
            transform_result = self.transform_raw_transaction(transaction_data)
            
            # Check for errors in transformation
            if "error" in transform_result:
                logger.error(f"Error in transformation step: {transform_result['error']}")
                return transform_result
            
            # Step 2: Make prediction from features
            features = transform_result["features"]
            transaction_id = transform_result["transaction_id"]
            
            # Get prediction
            prediction_result = self.predict_from_features(features)
            
            # Check for errors in prediction
            if "error" in prediction_result:
                logger.error(f"Error in prediction step: {prediction_result['error']}")
                return prediction_result
                
            # Add transaction ID to the result
            prediction_result["transaction_id"] = transaction_id
            
            # Add transformed feature names for reference
            prediction_result["feature_names"] = transform_result["feature_names"]
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error in transaction pipeline: {str(e)}", exc_info=True)
            return {"error": f"Pipeline processing failed: {str(e)}"}
    
    def _get_risk_level(self, probability: float) -> str:
        """Determine risk level from probability"""
        if probability < 0.3:
            return "low"
        elif probability < 0.7:
            return "medium"
        else:
            return "high"
    
    def _get_feature_importance(self, features: np.ndarray) -> List[Dict[str, Any]]:
        """Get feature importance for this prediction"""
        # This is a simplified implementation - should be customized based on your model
        return [
            {"feature": "V14", "importance": 0.25},
            {"feature": "V12", "importance": 0.18},
            {"feature": "V10", "importance": 0.15},
            {"feature": "Amount", "importance": 0.12},
            {"feature": "V17", "importance": 0.10}
        ]
    
    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Provide fallback prediction when model is unavailable"""
        return {
            "prediction": {
                "is_fraud": False,
                "fraud_probability": 0.01,
                "risk_level": "low"
            },
            "processed_at": datetime.now().isoformat(),
            "note": "Fallback prediction - model not available"
        }
