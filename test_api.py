import requests
import json
import numpy as np
from datetime import datetime
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_tester")

# Load the request body for raw prediction
try:
    with open('predictRawRequestBody.json', 'r') as f:
        raw_request_body = json.load(f)
    logger.info("Loaded test data successfully")
except Exception as e:
    logger.error(f"Failed to load test data: {str(e)}")
    raw_request_body = {
        "card_number": "1234567890123456",
        "transaction_date": datetime.now().isoformat(),
        "transaction_amount": 100.0,
        "merchant_category": "retail",
        "merchant_name": "Test Store"
    }
    logger.warning("Using fallback test data")

# URL of your API on Render
base_url = "https://model-91mn.onrender.com"

# Check if the API is responsive at all
logger.info(f"Testing API at {base_url}...")
try:
    health_response = requests.get(f"{base_url}/health", timeout=10)
    if health_response.ok:
        logger.info(f"API is responsive: {health_response.json()}")
    else:
        logger.warning(f"API health check returned: {health_response.status_code}")
except Exception as e:
    logger.error(f"API seems to be down or unreachable: {str(e)}")

# Check if there are any docs or OpenAPI spec available
logger.info("Checking API documentation...")
try:
    docs_response = requests.get(f"{base_url}/docs", timeout=5)
    if docs_response.ok:
        logger.info(f"API docs available at: {base_url}/docs")
    else:
        logger.info("No API docs found.")
except Exception as e:
    logger.error(f"Error checking docs: {str(e)}")

# Try the raw endpoint with retry logic
def try_endpoint(endpoint, data, max_retries=3, backoff_factor=1.5):
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

# Try the raw prediction endpoint
raw_result = try_endpoint("/predict/raw", raw_request_body)

if raw_result:
    logger.info(f"Raw prediction result: {json.dumps(raw_result, indent=2)}")
else:
    logger.warning("Raw prediction failed, trying regular prediction endpoint...")
    
    # Transform raw transaction data to features (simplified version)
    def transform_raw_transaction(transaction_data):
        # Very simplified transformation - in practice you'd have proper feature engineering
        features = [0] * 30  # Create 30 features (typical for fraud models)
        # Set amount as feature 29 (index 28)
        features[28] = transaction_data.get("transaction_amount", 0)
        # Set time as feature 0
        features[0] = datetime.now().timestamp() % 86400  # Seconds since midnight
        return features
    
    # Transform the raw data
    features = transform_raw_transaction(raw_request_body)
    
    # Format for the /predict endpoint
    predict_request_body = {
        "features": features
    }
    
    # Try the standard prediction endpoint as fallback
    predict_result = try_endpoint("/predict", predict_request_body)
    
    if predict_result:
        logger.info(f"Standard prediction result: {json.dumps(predict_result, indent=2)}")
    else:
        logger.error("All prediction attempts failed")
