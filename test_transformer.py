import json
import logging
import numpy as np
import requests
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("transformer_tester")

def load_json_data(file_path):
    """Load JSON data from the specified file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Successfully loaded data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Failed to load data from {file_path}: {str(e)}")
        return None

def transform_transaction_data(transaction_data):
    """Transform raw transaction data to model features"""
    logger.info("Transforming transaction data to features")
    
    # Initialize features array (adjust size based on your model's requirements)
    features = [0] * 30
    
    # Extract and transform features from transaction data
    # These are example transformations - modify based on your actual model's requirements
    try:
        # Amount feature
        features[0] = float(transaction_data.get('amount', 0))
        
        # Time-based features
        timestamp_str = transaction_data.get('timestamp')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            # Hour of day (0-23)
            features[1] = timestamp.hour
            # Day of week (0-6)
            features[2] = timestamp.weekday()
            # Is weekend
            features[3] = 1 if timestamp.weekday() >= 5 else 0
        
        # Merchant category encoding (simplified)
        merchant_cat = transaction_data.get('merchant_category', '').lower()
        categories = ['retail', 'food', 'electronics', 'travel', 'other']
        for i, cat in enumerate(categories):
            features[4 + i] = 1 if merchant_cat == cat else 0
        
        # Boolean features
        features[9] = 1 if transaction_data.get('is_online', False) else 0
        features[10] = 1 if transaction_data.get('card_present', False) else 0
        features[11] = 1 if transaction_data.get('unusual_location', False) else 0
        features[12] = 1 if transaction_data.get('high_frequency', False) else 0
        
        # Country encoding (simplified)
        country = transaction_data.get('country', '').upper()
        countries = ['US', 'CA', 'GB', 'AU', 'OTHER']
        for i, c in enumerate(countries):
            features[13 + i] = 1 if country == c else 0
            
        logger.info("Transformation completed successfully")
        return features
        
    except Exception as e:
        logger.error(f"Error during transformation: {str(e)}")
        return None

def try_endpoint(endpoint, data, base_url="https://model-91mn.onrender.com", max_retries=3, backoff_factor=1.5):
    """Try an endpoint with exponential backoff retry logic"""
    url = f"{base_url}{endpoint}"
    retry_count = 0
    
    while retry_count < max_retries:
        logger.info(f"Trying endpoint: {url} (Attempt {retry_count+1}/{max_retries})")
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            logger.info(f"Status code: {response.status_code}")
            
            if response.ok:
                logger.info("Success!")
                try:
                    return response.json()
                except:
                    return {"text_response": response.text}
            else:
                logger.warning(f"Failed with status {response.status_code}: {response.text}")
                
            if response.status_code == 404:
                # Don't retry if endpoint doesn't exist
                break
                
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
        
        # Exponential backoff
        wait_time = backoff_factor ** retry_count
        logger.info(f"Waiting {wait_time:.1f} seconds before retry...")
        time.sleep(wait_time)
        retry_count += 1
    
    return None

def main():
    # File path
    json_file_path = 'predictRawRequestBody.json'
    
    # Load JSON data
    data = load_json_data(json_file_path)
    if not data:
        return
    
    # Print the input data
    logger.info(f"Input transaction data: {json.dumps(data, indent=2)}")
    
    # Transform the data
    features = transform_transaction_data(data)
    
    # Print the transformed features
    if features:
        logger.info(f"Transformed features: {features}")
        logger.info(f"Number of features: {len(features)}")
        
        # Format for the /predict endpoint
        predict_request_body = {
            "features": features
        }
        logger.info(f"Request body for /predict endpoint: {json.dumps(predict_request_body, indent=2)}")

        # Verify if this is the format the API expects by trying the prediction endpoint
        logger.info("Testing the /predict endpoint with transformed data...")
        result = try_endpoint("/predict", predict_request_body)
        
        if result:
            logger.info(f"Prediction result: {json.dumps(result, indent=2)}")
            logger.info("✅ The /predict endpoint accepted our format!")
        else:
            logger.error("❌ The /predict endpoint did not accept our format or is unavailable.")
            
        # Also try the raw endpoint with the original JSON for comparison
        logger.info("Testing the /raw endpoint with original JSON data...")
        raw_result = try_endpoint("/raw", data)
        
        if raw_result:
            logger.info(f"Raw prediction result: {json.dumps(raw_result, indent=2)}")
            if "prediction" in raw_result:
                logger.info(f"Raw endpoint prediction: {raw_result['prediction']}")
        else:
            logger.warning("Raw endpoint test failed")

if __name__ == "__main__":
    main()
