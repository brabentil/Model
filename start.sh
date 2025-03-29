#!/bin/bash

# Set environment variables
export PYTHONPATH=$(pwd)
export API_PORT=8000

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Start the API server
python main.py
