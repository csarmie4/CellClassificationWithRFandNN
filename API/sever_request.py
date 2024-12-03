import requests
import os
import base64
import logging
from time import sleep
from requests.exceptions import RequestException
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Set the API base URL and the directory to save images
API_BASE_URL = os.getenv("API_BASE_URL")
if not API_BASE_URL:
    raise ValueError("API_BASE_URL is not set. Please configure it in your .env file.")
SAVE_DIR = os.getenv("SAVE_DIR")  # Directory to save the fetched images
if not SAVE_DIR:
    raise ValueError("SAVE_DIR is not set. Please configure it in your .env file.")

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_and_save_all_images(dataset_name, limit=10):
    """
    Fetch all images from the FastAPI server and save them to the specified directory.
    
    Args:
        dataset_name (str): The name of the dataset to fetch images from.
        limit (int): Number of images to fetch per request.
    """
    url = f"{API_BASE_URL}/get_all_images"
    skip = 0
    total_images = 0

    while True:
        logging.info(f"Requesting images with skip={skip} and limit={limit}")  # Debugging statement
        
        # Send dataset_name, skip, and limit as JSON payload
        payload = {"dataset_name": dataset_name, "skip": skip, "limit": limit}
        
        try:
            response = requests.post(url, json=payload, timeout=30)  # Timeout added
            response.raise_for_status()
        except RequestException as e:
            logging.error(f"Request failed: {e}")
            sleep(2)  # Retry after waiting a few seconds
            continue
        
        logging.info(f"API Response Status: {response.status_code}")  # Log the status code
        if response.status_code != 200:
            logging.error(f"Failed to fetch images: {response.status_code} - {response.text}")
            break
        
        data = response.json()
        total_images = data.get("total", 0)
        images = data.get("images", [])

        logging.info(f"Total images in dataset: {total_images}, Images fetched in this call: {len(images)}")  # Debugging statement
        
        if not images:  # No more images to fetch
            logging.info("No more images to fetch.")
            break

        for img_data in images:
            image_id = img_data.get("image_id", "unknown")
            img_bytes = base64.b64decode(img_data["image_data"])
            image_path = os.path.join(SAVE_DIR, f"{image_id}.jpg")  # Add file extension for better management
            
            # Save the image to the specified path
            with open(image_path, "wb") as img_file:
                img_file.write(img_bytes)
                
            logging.info(f"Saved {image_id} to {image_path}")

        skip += limit  # Move to the next set of images

        # Break the loop if we have fetched all images
        if skip >= total_images:
            logging.info("All images have been fetched.")
            break

# Example usage
dataset_name = "BBBC041"  # Dataset name to fetch images from
fetch_and_save_all_images(dataset_name)
