import requests
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transform_test")

# API configuration
BASE_URL = "https://model-91mn.onrender.com"
TRANSFORM_ENDPOINT = f"{BASE_URL}/transform"

def create_test_transaction() -> Dict[str, Any]:
    """Create a sample transaction for testing"""
    return {
        "amount": 120.99,
        "timestamp": datetime.now().isoformat(),
        "merchant_name": "Test Online Store",
        "merchant_category": "retail",
        "is_online": True,
        "card_present": False,
        "country": "US",
        "unusual_location": False,
        "high_frequency": False,
        "test_flag": True  # To mark this as a test transaction
    }

def validate_transform_response(response_data: Dict[str, Any]) -> bool:
    """Validate that the transform response has the expected structure"""
    # Check that required fields are present
    required_fields = ["features", "feature_names", "timestamp", "transaction_id"]
    for field in required_fields:
        if field not in response_data:
            logger.error(f"Missing required field: {field}")
            return False
    
    # Check that features is a list of numbers with the expected length
    if not isinstance(response_data["features"], list):
        logger.error("Features is not a list")
        return False
    
    if len(response_data["features"]) != 30:  # Expecting 30 features
        logger.error(f"Expected 30 features, got {len(response_data['features'])}")
        return False
    
    # Check that feature_names matches the expected pattern
    expected_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    if response_data["feature_names"] != expected_names:
        logger.error(f"Feature names don't match the expected pattern")
        return False
    
    return True

def test_transform_endpoint():
    """Test the transform endpoint with a sample transaction"""
    # Create test transaction
    transaction = create_test_transaction()
    logger.info(f"Testing transform endpoint with transaction: {transaction}")
    
    try:
        # Send request to transform endpoint
        start_time = time.time()
        response = requests.post(
            TRANSFORM_ENDPOINT,
            json=transaction,
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = time.time() - start_time
        
        # Log the response time
        logger.info(f"Transform API response time: {elapsed_time:.2f} seconds")
        
        # Check status code
        if response.status_code != 200:
            logger.error(f"Error: Received status code {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return False
        
        # Parse response
        response_data = response.json()
        logger.info(f"Received response: {json.dumps(response_data, indent=2)}")
        
        # Validate response
        if validate_transform_response(response_data):
            logger.info("✅ Transform endpoint test PASSED")
            return True
        else:
            logger.error("❌ Transform endpoint test FAILED: Invalid response format")
            return False
            
    except Exception as e:
        logger.error(f"❌ Transform endpoint test FAILED with exception: {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Starting transform endpoint test")
    success = test_transform_endpoint()
    exit(0 if success else 1)
