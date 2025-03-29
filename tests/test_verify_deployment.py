import requests
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api_verifier")

# API configuration
BASE_URL = "https://model-91mn.onrender.com"

def check_endpoint(path: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
    """
    Check if an endpoint is available and working
    
    Args:
        path: API endpoint path
        method: HTTP method (GET or POST)
        data: Data to send for POST requests
        
    Returns:
        Dictionary with test results
    """
    url = f"{BASE_URL}{path}"
    logger.info(f"Testing {method} {url}")
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, timeout=10)
        else:  # POST
            response = requests.post(url, json=data, timeout=10)
        
        result = {
            "endpoint": path,
            "method": method,
            "status_code": response.status_code,
            "available": 200 <= response.status_code < 300,
            "response_time_ms": response.elapsed.total_seconds() * 1000
        }
        
        # Try to parse response as JSON
        try:
            result["response"] = response.json()
        except:
            result["response"] = response.text[:100] + "..." if len(response.text) > 100 else response.text
            
        return result
        
    except Exception as e:
        logger.error(f"Error testing {method} {url}: {str(e)}")
        return {
            "endpoint": path,
            "method": method,
            "available": False,
            "error": str(e)
        }

def create_test_transaction() -> Dict[str, Any]:
    """Create a sample transaction for testing"""
    return {
        "amount": 120.99,
        "timestamp": datetime.now().isoformat(),
        "merchant_name": "Test Online Store",
        "merchant_category": "retail",
        "is_online": True,
        "card_present": False,
        "country": "US"
    }

def verify_deployment():
    """Verify all API endpoints"""
    # List of endpoints to check
    endpoints = [
        {"path": "/", "method": "GET"},
        {"path": "/docs", "method": "GET"},
        {"path": "/health", "method": "GET"},
        {"path": "/test", "method": "POST", "data": {}},
        {"path": "/predict", "method": "POST", "data": {"features": [0] * 30}},
        {"path": "/raw", "method": "POST", "data": create_test_transaction()},
        {"path": "/transform", "method": "POST", "data": create_test_transaction()},
        {"path": "/transform/status", "method": "GET"},
        {"path": "/pipeline", "method": "POST", "data": create_test_transaction()}
    ]
    
    # Check each endpoint
    results = []
    for endpoint in endpoints:
        result = check_endpoint(
            endpoint["path"], 
            endpoint["method"], 
            endpoint.get("data")
        )
        results.append(result)
        
        # Print result
        status = "✅ PASSED" if result.get("available") else "❌ FAILED"
        logger.info(f"{status} - {endpoint['method']} {endpoint['path']}")
    
    # Print summary
    available = sum(1 for r in results if r.get("available", False))
    logger.info(f"Summary: {available}/{len(endpoints)} endpoints available")
    
    # Return results
    return results

if __name__ == "__main__":
    logger.info("Starting API deployment verification")
    results = verify_deployment()
    
    # Save results to file
    with open("api_verification_results.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to api_verification_results.json")
    
    # Exit with status code
    success = all(r.get("available", False) for r in results)
    sys.exit(0 if success else 1)
