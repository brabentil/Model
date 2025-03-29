# Fraud Detection API

This is a machine learning API for detecting fraudulent credit card transactions.

## Deployment Instructions

### Render Deployment

1. Sign up for a Render account at [render.com](https://render.com)
2. Create a new Web Service and connect your GitHub repository
3. Set the following configuration:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.index:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.9
   - **Environment Variables**: None required by default

Note: Render will automatically detect the `requirements.txt` file and install dependencies.

### Alternative Start Commands

#### For production environments with Gunicorn:
