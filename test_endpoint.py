import requests
import json
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base URL - change this to match your deployment
BASE_URL = "http://localhost:8000"  # Local development
# BASE_URL = "https://model-91mn.onrender.com"  # Render deployment

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        logger.info(f"Health check status: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing health endpoint: {e}")
        return False

def test_raw_prediction():
    """Test the raw prediction endpoint"""
    try:
        # Sample transaction data
        data = {
            "amount": 150.00,
            "timestamp": datetime.now().isoformat(),
            "merchant_name": "Test Store",
            "merchant_category": "retail",
            "is_online": False,
            "card_present": True,
            "country": "US",
            "unusual_location": False,
            "high_frequency": False
        }
        
        logger.info(f"Sending test transaction to /raw: {data}")
        response = requests.post(f"{BASE_URL}/raw", json=data)
        
        logger.info(f"Raw prediction status: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            logger.error(f"Error response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing raw prediction: {e}")
        return False

def test_feature_prediction():
    """Test the feature vector prediction endpoint"""
    try:
        # Sample feature vector (30 features: Time + V1-V28 + Amount)
        features = [0.0] * 30
        features[0] = 0  # Time
        features[1] = -1.35  # V1
        features[29] = 150.0  # Amount
        
        data = {"features": features}
        
        logger.info("Sending test feature vector to /predict")
        response = requests.post(f"{BASE_URL}/predict", json=data)
        
        logger.info(f"Feature prediction status: {response.status_code}")
        if response.status_code == 200:
            logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            logger.error(f"Error response: {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing feature prediction: {e}")
        return False

def test_simple_endpoint():
    """Test the simple test endpoint"""
    try:
        response = requests.post(f"{BASE_URL}/test")
        logger.info(f"Test endpoint status: {response.status_code}")
        logger.info(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error testing simple endpoint: {e}")
        return False

if __name__ == "__main__":
    logger.info(f"Testing API at {BASE_URL}")
    
    # First test the simplest endpoint to check if API is running
    if test_simple_endpoint():
        logger.info("Basic API test passed! Server is responding.")
    else:
        logger.error("Basic API test failed. Server might be down.")
        exit(1)
    
    # Test health endpoint
    test_health()
    
    # Test raw prediction
    test_raw_prediction()
    
    # Test feature vector prediction
    test_feature_prediction()
    
    logger.info("All tests completed!")
