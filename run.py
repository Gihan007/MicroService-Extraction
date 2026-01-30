#!/usr/bin/env python3
"""
Run script for the 10-K Data Ingestor Microservice.
Starts the FastAPI server.
"""

import uvicorn
from graphDB_builder_with_unstructured_data import app

if __name__ == "__main__":
    print("Starting 10-K Data Ingestor Microservice...")
    uvicorn.run(app, host="0.0.0.0", port=8000)