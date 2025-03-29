import numpy as np
import pandas as pd
from datetime import datetime
import joblib
import os

class TransactionTransformer:
    """
    Utility to transform raw credit card transaction data into the format 
    required by the model trained on the Kaggle Credit Card Fraud dataset.
    
    Since the original dataset features (V1-V28) are PCA-transformed for privacy,
    we can't exactly reverse-engineer them. This class provides approximations.
    """
    
    def __init__(self):
        # Reference statistical values from the original dataset
        # These would ideally be calculated from your training data
        self.time_mean = 94813.86
        self.time_std = 47488.15
        self.amount_mean = 88.35
        self.amount_std = 250.12
        
        # Load the scaler if available
        scaler_path = os.path.join(os.path.dirname(__file__), "model", "scaler.pkl")
        try:
            self.scaler = joblib.load(scaler_path)
        except:
            self.scaler = None
            print("Scaler not found. Using basic normalization.")
            
    def transform_raw_transaction(self, transaction_data):
        """
        Transform a raw transaction into the format expected by the model.
        
        Args:
            transaction_data (dict): Raw transaction with fields like:
                - amount: Transaction amount
                - timestamp: Transaction timestamp
                - merchant_category: Category code
                - merchant_name: Store/service name
                - is_online: Whether transaction was online
                - card_present: Whether physical card was used
                - country: Country of transaction
                - etc...
                
        Returns:
            np.array: Array of 30 features in the format expected by the model
        """
        # Extract available data
        amount = transaction_data.get('amount', 0)
        
        # Calculate time feature (seconds since midnight or since first transaction)
        if 'timestamp' in transaction_data:
            try:
                timestamp = transaction_data['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_feature = timestamp.timestamp() % 86400  # seconds since midnight
            except:
                time_feature = 0
        else:
            time_feature = 0
            
        # Normalize time and amount (if scaler not available)
        if self.scaler is None:
            time_feature = (time_feature - self.time_mean) / self.time_std
            amount_normalized = (amount - self.amount_mean) / self.amount_std
        else:
            # The scaler will handle this during prediction
            time_feature = time_feature
            amount_normalized = amount
            
        # Generate synthetic V1-V28 features based on available transaction data
        # This is an approximation - in a real system you'd need domain expertise
        v_features = self._generate_v_features(transaction_data)
        
        # Combine all features
        features = np.array([time_feature] + v_features + [amount_normalized])
        
        return features
    
    def get_accuracy_estimate(self):
        """
        Provides an estimate of how accurate the transformation is likely to be.
        """
        return {
            "estimated_accuracy": "low to medium",
            "confidence": 0.4,
            "limitations": [
                "PCA transformation matrix is unknown",
                "Feature distributions may not match training data",
                "Heuristic rules are approximations"
            ],
            "recommendation": "Consider retraining the model on raw transaction features for production use"
        }
    
    def _generate_v_features(self, transaction_data):
        """
        Generate approximate V1-V28 features based on transaction attributes.
        This is a very simplified approach and would need refinement for production.
        
        In a real system, you might:
        1. Use domain knowledge to map transaction attributes to meaningful features
        2. Train an autoencoder to generate V1-V28 like features
        3. Use a subset of the most important V features based on your model
        """
        # Start with zeros for V1-V28
        v_features = [0.0] * 28
        
        # Extract useful information that might correlate with fraud patterns
        amount = transaction_data.get('amount', 0)
        is_online = transaction_data.get('is_online', False)
        merchant_category = transaction_data.get('merchant_category', '')
        unusual_location = transaction_data.get('unusual_location', False)
        high_frequency = transaction_data.get('high_frequency', False)
        
        # Set some values based on transaction characteristics
        # These mappings are arbitrary and would need tuning based on your data
        
        # V1: Often correlates with transaction type
        v_features[0] = -1.2 if is_online else 0.5
        
        # V2: Often correlates with amount
        v_features[1] = -0.5 if amount > 200 else 0.3
        
        # V3: Could relate to merchant category
        if merchant_category in ['jewelry', 'electronics', 'travel']:
            v_features[2] = -0.7  # Higher risk categories
        else:
            v_features[2] = 0.2
            
        # V4: Might indicate location anomalies
        v_features[3] = -1.0 if unusual_location else 0.1
        
        # V5: Could relate to transaction frequency
        v_features[4] = -0.8 if high_frequency else 0.2
        
        # For remaining features, we use small random values to approximate distribution
        # In a real system, you'd map these to actual transaction attributes
        for i in range(5, 28):
            v_features[i] = np.random.normal(0, 0.3)  # centered around 0 with small variance
        
        return v_features

