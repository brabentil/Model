import requests
import json
import numpy as np
from datetime import datetime

# Load the request body for raw prediction
with open('predictRawRequestBody.json', 'r') as f:
    raw_request_body = json.load(f)

# URL of your API on Render
base_url = "https://model-91mn.onrender.com"

# Check if there are any docs or OpenAPI spec available
print("Checking API documentation...")
try:
    docs_response = requests.get(f"{base_url}/docs")
    if docs_response.ok:
        print(f"API docs available at: {base_url}/docs")
    else:
        print("No API docs found.")
except Exception as e:
    print(f"Error checking docs: {e}")

# Try different variations of the raw endpoint
raw_endpoints = [
    "/predict/raw", 
    "/raw/predict",
    "/predictraw", 
    "/raw_predict"
]

for endpoint in raw_endpoints:
    url = f"{base_url}{endpoint}"
    print(f"\nTrying endpoint: {url}")
    
    try:
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=raw_request_body, headers=headers)
        
        print(f"Status code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2) if response.ok else response.text)
        
        if response.ok:
            print(f"Success! The working endpoint is: {endpoint}")
            break
    except Exception as e:
        print(f"Error: {e}")

# If all raw endpoints fail, manually transform and use the /predict endpoint
print("\nManually transforming raw transaction data to features...")

# Transform raw transaction data to features (simplified version of what TransactionTransformer does)
def transform_raw_transaction(transaction_data):
    # Statistical values from TransactionTransformer
    time_mean = 94813.86
    time_std = 47488.15
    amount_mean = 88.35
    amount_std = 250.12
    
    # Extract data
    amount = transaction_data.get('amount', 0)
    
    # Calculate time feature
    time_feature = 0
    if 'timestamp' in transaction_data:
        try:
            timestamp = transaction_data['timestamp']
            if isinstance(timestamp, str):
                timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            time_feature = timestamp.timestamp() % 86400  # seconds since midnight
        except:
            time_feature = 0
    
    # Normalize time and amount
    time_feature = (time_feature - time_mean) / time_std
    amount_normalized = (amount - amount_mean) / amount_std
    
    # Generate synthetic V1-V28 features
    v_features = [0.0] * 28
    
    # Extract useful information
    is_online = transaction_data.get('is_online', False)
    merchant_category = transaction_data.get('merchant_category', '')
    unusual_location = transaction_data.get('unusual_location', False)
    high_frequency = transaction_data.get('high_frequency', False)
    
    # Set V1-V5 based on transaction characteristics (from codebase)
    v_features[0] = -1.2 if is_online else 0.5
    v_features[1] = -0.5 if amount > 200 else 0.3
    
    if merchant_category in ['jewelry', 'electronics', 'travel']:
        v_features[2] = -0.7  # Higher risk categories
    else:
        v_features[2] = 0.2
        
    v_features[3] = -1.0 if unusual_location else 0.1
    v_features[4] = -0.8 if high_frequency else 0.4
    
    # Combine all features
    features = [time_feature] + v_features + [amount_normalized]
    
    return features

# Transform the raw data
features = transform_raw_transaction(raw_request_body)

# Format for the /predict endpoint - it expects a list in "features" field
predict_request_body = {
    "features": features
}

print("\nTrying the /predict endpoint with transformed features...")
url = f"{base_url}/predict"
try:
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=predict_request_body, headers=headers)
    
    print(f"Status code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2) if response.ok else response.text)
except Exception as e:
    print(f"Error: {e}")

print("\nNote: This is using a simplified version of the transformation logic.")
print("For production use, you would want to implement the full transformation logic from TransactionTransformer.")
