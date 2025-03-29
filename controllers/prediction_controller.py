import numpy as np
import pandas as pd
import logging
import os
import joblib
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

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
    
    def predict_from_features(self, features: List[float]) -> Dict[str, Any]:
        """Make prediction from pre-processed feature vector"""
        if self.model is None:
            return self._get_fallback_prediction()
            
        try:
            # Validate feature count
            if len(features) != 30:
                logger.warning(f"Expected 30 features, got {len(features)}")
                return {
                    "error": f"Expected 30 features, got {len(features)}",
                    "prediction": self._get_fallback_prediction()["prediction"]
                }
            
            # Reshape for prediction and apply scaler if available
            features_array = np.array(features).reshape(1, -1)
            if self.scaler:
                features_df = pd.DataFrame(features_array, columns=self.feature_names)
                features_scaled = self.scaler.transform(features_df)
            else:
                features_scaled = features_array
            
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
            logger.error(f"Error making prediction from features: {e}")
            return {
                "error": str(e),
                "prediction": self._get_fallback_prediction()["prediction"]
            }
    
    def predict_from_raw_transaction(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction from raw transaction data"""
        try:
            # Transform raw transaction to features
            if self.transformer is None:
                self.transformer = TransactionTransformer()
                
            # Log the transaction with sensitive data removed
            safe_transaction = {**transaction}
            if 'card_number' in safe_transaction:
                safe_transaction['card_number'] = '****' + safe_transaction['card_number'][-4:]
            logger.info(f"Processing transaction: {safe_transaction}")
            
            # Transform the transaction
            transformed_data = self.transformer.transform_raw_transaction(transaction)
            logger.debug(f"Transformed features: {transformed_data}")
            
            # Extract values from transformed data in correct order
            features = [
                transformed_data["Time"],
                *[transformed_data[f"V{i}"] for i in range(1, 29)],
                transformed_data["Amount"]
            ]
            
            # Use the standard prediction method
            result = self.predict_from_features(features)
            
            # Add transaction ID if available
            if 'transaction_id' in transaction:
                result['transaction_id'] = transaction['transaction_id']
            else:
                result['transaction_id'] = 'unknown'
                
            return result
            
        except Exception as e:
            logger.error(f"Error in raw transaction prediction: {e}")
            return {
                "error": str(e),
                "prediction": self._get_fallback_prediction()["prediction"],
                "note": "Error processing raw transaction, using fallback"
            }
    
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
