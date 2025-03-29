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
            dict: Dictionary with features in the format expected by the model
        """
        # Extract available data with validation
        try:
            amount = float(transaction_data.get('amount', 0))
        except (ValueError, TypeError):
            amount = 0
            print("Warning: Invalid amount value, defaulting to 0")
        
        # Calculate time feature (seconds since midnight or since first transaction)
        time_feature = 0
        if 'timestamp' in transaction_data:
            try:
                timestamp = transaction_data['timestamp']
                if isinstance(timestamp, str):
                    # Handle different timestamp formats
                    try:
                        timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    except ValueError:
                        # Try with different format
                        timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                elif isinstance(timestamp, (int, float)):
                    # Unix timestamp
                    timestamp = datetime.fromtimestamp(timestamp)
                    
                # Calculate seconds since midnight
                time_feature = (timestamp.hour * 3600 + timestamp.minute * 60 + timestamp.second)
            except Exception as e:
                print(f"Warning: Could not parse timestamp: {e}, defaulting to 0")
        
        # Normalize time and amount (if scaler not available)
        if self.scaler is None:
            time_feature = (time_feature - self.time_mean) / self.time_std if self.time_std != 0 else 0
            amount_normalized = (amount - self.amount_mean) / self.amount_std if self.amount_std != 0 else 0
        else:
            # The scaler will handle this during prediction
            time_feature = time_feature
            amount_normalized = amount
            
        # Generate V1-V28 features
        v_features_dict = self._generate_v_features(transaction_data)
        
        # Format the output as a dictionary matching the required structure
        output = {
            "Time": time_feature,
            **v_features_dict,
            "Amount": amount
        }

        print(f"Transformed features: {output}")
        
        return output
    
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
        
        Returns:
            dict: Dictionary with V1-V28 features
        """
        # Extract useful information that might correlate with fraud patterns
        try:
            amount = float(transaction_data.get('amount', 0))
        except (ValueError, TypeError):
            amount = 0
        
        # Boolean values with proper defaults
        is_online = bool(transaction_data.get('is_online', False))
        unusual_location = bool(transaction_data.get('unusual_location', False))
        high_frequency = bool(transaction_data.get('high_frequency', False))
        card_present = bool(transaction_data.get('card_present', True))
        
        # String values with proper defaults
        merchant_category = str(transaction_data.get('merchant_category', ''))
        merchant_name = str(transaction_data.get('merchant_name', ''))
        country = str(transaction_data.get('country', ''))
        
        # Initialize the V features dictionary
        v_features = {}
        
        # V1: Often correlates with transaction type
        v_features["V1"] = -1.2 if is_online else 0.5
        
        # V2: Often correlates with amount
        v_features["V2"] = -0.5 if amount > 200 else 0.3
        
        # V3: Could relate to merchant category
        high_risk_categories = ['jewelry', 'electronics', 'travel', 'gambling', 'cryptocurrency']
        v_features["V3"] = -0.7 if any(category in merchant_category.lower() for category in high_risk_categories) else 0.2
        
        # V4: Might indicate location anomalies
        v_features["V4"] = -1.0 if unusual_location else 0.1
        
        # V5: Could relate to transaction frequency
        v_features["V5"] = -0.8 if high_frequency else 0.2
        
        # V6: Card present vs not present
        v_features["V6"] = 0.4 if card_present else -0.6
        
        # V7: Country risk factor
        high_risk_countries = ['NG', 'RO', 'RU', 'UA']
        v_features["V7"] = -0.9 if country in high_risk_countries else 0.3
        
        # For remaining features, we use small random values to approximate distribution
        # Using a fixed seed for consistency in predictions
        np.random.seed(hash(str(transaction_data)) % 2**32)
        for i in range(8, 29):
            v_features[f"V{i}"] = np.random.normal(0, 0.3)
            
        return v_features

