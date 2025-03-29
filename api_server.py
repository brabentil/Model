"""
API server initialization script
This ensures the API is properly loaded with all endpoints
"""
import os
import logging
import uvicorn
from fastapi import FastAPI, HTTPException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api_server")

def start_server():
    """Start the API server with appropriate configuration"""
    try:
        # First try to import the full API
        logger.info("Attempting to load the main API...")
        from api import app
        logger.info("Main API loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load main API: {str(e)}")
        logger.info("Falling back to simplified API...")
        
        try:
            # If main API fails, load the fallback
            from api.fallback import app
            logger.info("Fallback API loaded successfully")
        except Exception as fallback_error:
            logger.error(f"Failed to load fallback API: {str(fallback_error)}")
            raise RuntimeError("Could not initialize any API version")
    
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8000))
    
    # Start the server
    logger.info(f"Starting API server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    start_server()
