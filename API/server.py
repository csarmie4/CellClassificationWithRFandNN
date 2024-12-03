from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import os
import logging
import base64
from pydantic import BaseModel
import time

app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to the datasets root folder
DATASETS_PATH = "datasets_raw"

# Define request body model
class ImageRequestBody(BaseModel):
    dataset_name: str
    skip: int = 0
    limit: int = 10

# Rate Limiting (in-memory approach)
REQUEST_LIMIT = 100 # Max requests per IP
TIME_WINDOW = 60 # Time window in seconds (1 minute)
rate_limiter = {}

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for simplicity; you can specify a list of trusted domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def sanitize_dataset_name(dataset_name: str) -> str:
    if not dataset_name.isalnum() and "_" not in dataset_name:
        raise HTTPException(status_code=400, detail="Invalid dataset name")
    return dataset_name

def get_dataset_paths(dataset_name: str) -> Path:
    dataset_name = sanitize_dataset_name(dataset_name)
    dataset_path = Path(DATASETS_PATH) / dataset_name
    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail="Dataset not found")
    return dataset_path

def rate_limit(request: Request):
    # Rate Limiting: Track the IP address and request time
    ip = request.client.host
    current_time = time.time()

    if ip not in rate_limiter:
        rate_limiter[ip] = []

    # Filter out requests that are outside the time window
    rate_limiter[ip] = [timestamp for timestamp in rate_limiter[ip] if current_time - timestamp < TIME_WINDOW]

    # Check if the number of requests exceeds the limit
    if len(rate_limiter[ip]) >= REQUEST_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")

    # Record the current request time
    rate_limiter[ip].append(current_time)

@app.post("/get_all_images")
async def get_all_images(request: Request, request_body: ImageRequestBody):
    """
    Fetch a chunk of images from the dataset.
    """
    try:
        # Apply rate limiting
        rate_limit(request)

        dataset_name = request_body.dataset_name
        skip = request_body.skip
        limit = request_body.limit
        # Get the correct path to the 'images' directory
        folder_path = get_dataset_paths(dataset_name) / "malaria" / "images" # Add "malaria/images" to path
        logging.info(f"Fetching images from dataset {dataset_name}.")

        # Get all image files in the 'images' directory
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif"))]
        logging.info(f"Found {len(image_files)} image(s) in dataset {dataset_name}.") # Log the number of images
        # Apply pagination (skip/limit)
        images_chunk = image_files[skip: skip + limit]
        total_images = len(image_files)

        # Prepare the image data to send in the response
        images = []
        for img_file in images_chunk:
            img_path = folder_path / img_file
            with open(img_path, "rb") as img_file_handle:
                img_data = base64.b64encode(img_file_handle.read()).decode('utf-8')
                images.append({"image_id": img_file, "image_data": img_data})

        return JSONResponse(content={"total": total_images, "images": images})
    except HTTPException as e:
        logging.error(f"Error: {e.detail}")
        raise e
    except Exception as e:
        logging.exception("Unexpected error occurred")
        raise HTTPException(status_code=500, detail="Internal server error")