import uvicorn
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("api")

if __name__ == "__main__":
    # Import the app here to avoid circular imports
    from api import app
    
    # Print all registered routes for debugging
    logger.info("Registered API routes:")
    for route in app.routes:
        logger.info(f"Route: {route.path}, methods: {route.methods}")
        
    # Set host and port from environment variables or use defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run("api:app", host=host, port=port, reload=True)
