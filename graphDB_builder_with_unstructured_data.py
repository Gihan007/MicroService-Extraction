# Core
import os
import json
import re
import time
import asyncio
import logging
import traceback
import shutil
import csv
from datetime import datetime

# Config
from dotenv import load_dotenv
from config import get_config
config = get_config()

# LLM
from openai import OpenAI

# Utilities
from metric_extractor import MetricExtractor

# Web Framework
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# from langsmith import traceable


# FastAPI Models
class TenKRequest(BaseModel):
    ticker: str
    start_year: int
    end_year: int

# FastAPI App
app = FastAPI(title="10-K and Metics Data Ingestor Microservice", version="1.0.0")


def create_run_folder():
    """Create a timestamped folder for this run's JSON files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(os.getcwd(), "file_data", "metrics_storage", f"run_{timestamp}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def create_html_folder():
    """Create a timestamped folder for this run's HTML files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_folder = os.path.join(os.getcwd(), "file_data", "html_storage", f"run_{timestamp}")
    os.makedirs(html_folder, exist_ok=True)
    return html_folder


def create_csv_folder():
    """Create a timestamped folder for this run's CSV files."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_folder = os.path.join(os.getcwd(), "file_data", "csv_storage", f"run_{timestamp}")
    os.makedirs(csv_folder, exist_ok=True)
    return csv_folder


# @traceable(name="unstructured.tenK_data_injestor", metadata={"component": "data_ingestor"})
async def tenK_data_injestor(ticker=None, start_year=2011, end_year=2025):

    # Process ticker with year range
    if ticker:
        return await _process_ticker_year_range(ticker, start_year, end_year)
    
    # Neither provided
    response = {
        "status": "error",
        "message": "ticker must be provided.",
        "error": "Missing input",
        "http_status": 400
    }
    return response


async def _process_ticker_year_range(ticker, start_year, end_year):
    """Process 10-K data for a ticker across a year range by downloading from SEC and extracting metrics to JSON."""
    response = {
        "status": "success",
        "message": "",
        "error": None,
        "http_status": 200,
        "processed_years": [],
        "failed_years": [],
        "metrics_info": []
    }

    # Create run-specific folder for JSON files
    run_folder = create_run_folder()

    # Create run-specific folder for HTML files
    html_folder = create_html_folder()

    # Create run-specific folder for CSV files
    csv_folder = create_csv_folder()

    # Validate ticker
    if not re.match(r"^[A-Z]+$", ticker):
        response["status"] = "error"
        response["message"] = f"Invalid ticker '{ticker}'. Must contain only uppercase letters."
        response["error"] = "Invalid ticker format"
        response["http_status"] = 400
        return response

    # Validate year range
    if not (1900 <= start_year <= 2100 and 1900 <= end_year <= 2100):
        response["status"] = "error"
        response["message"] = f"Invalid year range {start_year}-{end_year}. Years must be between 1900-2100."
        response["error"] = "Invalid year range"
        response["http_status"] = 400
        return response

    if start_year > end_year:
        response["status"] = "error"
        response["message"] = f"Start year {start_year} cannot be greater than end year {end_year}."
        response["error"] = "Invalid year range"
        response["http_status"] = 400
        return response

    extractor = MetricExtractor()

    for year in range(start_year, end_year + 1):
        try:
            print(f"\n📅 Processing {ticker} {year}...")

            # Download and preprocess 10-K
            html_file = extractor.download_10k_html(ticker, str(year))
            cleaned_text = extractor.preprocess_text(html_file)

            # Store HTML file in html_folder
            html_path = f"{html_folder}/{ticker}_{year}_10k.html"
            shutil.copy(html_file, html_path)

            # Extract metrics in one LLM call
            print(f"📊 Extracting combined data for {ticker} {year}...")
            combined_result = extractor.extract_combined_data(cleaned_text)

            # Parse the results
            metrics_data = {"metrics": combined_result["data"]["metrics"]}

            # Save metrics JSON
            metrics_file = f"{run_folder}/{ticker}_{year}_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=4)

            # Convert to CSV and save
            csv_file = f"{csv_folder}/{ticker}_{year}_metrics.csv"
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                for key, value in metrics_data["metrics"].items():
                    csv_value = "NULL" if value is None else str(value)
                    writer.writerow([key, csv_value])

            print(f"✓ {ticker} {year} data extracted successfully")
            print(f"   Metrics: {len(metrics_data['metrics'])} | Cost: ${combined_result['cost_info']['total_cost_usd']:.6f}")

            response["processed_years"].append({
                "year": year,
                "status": "success",
                "metrics_file": metrics_file,
                "html_file": html_path,
                "csv_file": csv_file,
                "metrics_count": len(metrics_data["metrics"]),
                "extraction_cost": combined_result["cost_info"]["total_cost_usd"]
            })

            response["metrics_info"].append({
                "year": year,
                "json_file": metrics_file,
                "html_file": html_path,
                "csv_file": csv_file,
                "metrics_count": len(metrics_data["metrics"]),
                "extraction_cost": combined_result["cost_info"]["total_cost_usd"],
                "backend": combined_result["cost_info"]["backend"]
            })

            # Clean up downloaded file
            if os.path.exists(html_file):
                os.remove(html_file)

        except Exception as e:
            print(f"✗ Failed to process {ticker} {year}: {str(e)}")
            response["failed_years"].append({"year": year, "error": str(e)})
            continue

    response["message"] = f"Processed {len(response['processed_years'])} years for {ticker} ({start_year}-{end_year}) with metrics extraction"

    if response["failed_years"]:
        response["message"] += f", {len(response['failed_years'])} failed"
        if not response["processed_years"]:
            response["status"] = "error"
            response["http_status"] = 500

    return response


@app.post("/process-tenk")
async def process_tenk_endpoint(request: TenKRequest):
    """
    Endpoint to process 10-K data for a given ticker and year range.
    
    Request body:
    {
        "ticker": "AAPL",
        "start_year": 2020,
        "end_year": 2023
    }
    """
    try:
        # Validate input
        if not re.match(r"^[A-Z]+$", request.ticker):
            raise HTTPException(status_code=400, detail="Invalid ticker. Must contain only uppercase letters.")
        
        if not (1900 <= request.start_year <= 2100 and 1900 <= request.end_year <= 2100):
            raise HTTPException(status_code=400, detail="Invalid year range. Years must be between 1900-2100.")
        
        if request.start_year > request.end_year:
            raise HTTPException(status_code=400, detail="Start year cannot be greater than end year.")
        
        # Call the processing function
        result = await tenK_data_injestor(
            ticker=request.ticker,
            start_year=request.start_year,
            end_year=request.end_year
        )
        
        # Return the result with appropriate status code
        status_code = result.get("http_status", 200)
        if status_code >= 400:
            raise HTTPException(status_code=status_code, detail=result)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


if __name__ == "__main__":
    # Run the FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
