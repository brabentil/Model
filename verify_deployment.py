import requests
import json
import logging
import time
from datetime import datetime
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deployment_verifier")

# API configuration
BASE_URL = "https://model-91mn.onrender.com"

def check_version_compatibility():
    """Check version compatibility by making a direct request"""
    logger.info("Checking API version compatibility")
    
    try:
        # Try a simple health check request
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        response.raise_for_status()  # Raise exception for non-200 responses
        
        # Parse response
        data = response.json()
        logger.info(f"API Health Check: {data}")
        
        return True, data
        
    except Exception as e:
        logger.error(f"API compatibility check failed: {str(e)}")
        return False, {"error": str(e)}

def create_test_transaction():
    """Create a sample transaction for testing"""
    return {
        "amount": 150.75,
        "timestamp": datetime.now().isoformat(),
        "merchant_name": "Test Online Store",
        "merchant_category": "retail",
        "is_online": True,
        "card_present": False,
        "country": "US",
        "unusual_location": False,
        "high_frequency": False
    }

def test_endpoints():
    """Test all main endpoints with simple requests"""
    endpoints = [
        {"name": "Health Check", "path": "/health", "method": "GET"},
        {"name": "Test Endpoint", "path": "/test", "method": "POST", "data": {}},
        {"name": "Raw Predict", "path": "/raw", "method": "POST", "data": create_test_transaction()},
        {"name": "Pipeline", "path": "/pipeline", "method": "POST", "data": create_test_transaction()}
    ]
    
    results = []
    
    for endpoint in endpoints:
        try:
            logger.info(f"Testing {endpoint['name']} ({endpoint['method']} {endpoint['path']})")
            start_time = time.time()
            
            if endpoint['method'] == 'GET':
                response = requests.get(f"{BASE_URL}{endpoint['path']}", timeout=30)
            else:  # POST
                response = requests.post(
                    f"{BASE_URL}{endpoint['path']}", 
                    json=endpoint.get('data', {}),
                    timeout=30
                )
            
            elapsed = time.time() - start_time
            status = "OK" if response.status_code == 200 else "FAILED"
            
            result = {
                "name": endpoint["name"],
                "path": endpoint["path"],
                "status": status,
                "status_code": response.status_code,
                "elapsed_seconds": elapsed
            }
            
            if response.status_code == 200:
                result["response"] = response.json()
                logger.info(f"✅ {endpoint['name']} - Success ({elapsed:.2f}s)")
            else:
                result["error"] = response.text
                logger.error(f"❌ {endpoint['name']} - Failed: {response.status_code} ({elapsed:.2f}s)")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"❌ {endpoint['name']} - Exception: {str(e)}")
            results.append({
                "name": endpoint["name"],
                "path": endpoint["path"],
                "status": "ERROR",
                "error": str(e)
            })
    
    return results

def main():
    """Main verification function"""
    logger.info(f"Verifying deployment at {BASE_URL}")
    
    # Check version compatibility
    compatible, health_data = check_version_compatibility()
    if not compatible:
        logger.error("Deployment verification failed: Version compatibility issues")
        return False
        
    # Test endpoints
    results = test_endpoints()
    
    # Check if all tests passed
    success = all(r.get("status") == "OK" for r in results)
    
    # Save results to file
    output = {
        "timestamp": datetime.now().isoformat(),
        "base_url": BASE_URL,
        "successful": success,
        "health": health_data,
        "endpoint_results": results
    }
    
    with open("deployment_verification.json", "w") as f:
        json.dump(output, f, indent=2)
    
    logger.info(f"Verification {'successful' if success else 'failed'}")
    logger.info(f"Results saved to deployment_verification.json")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
