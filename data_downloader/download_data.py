import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import zipfile
from io import BytesIO
import time
import hashlib
import logging
from tqdm import tqdm

# Set up logging for better visibility into script execution
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Security constants
TRUSTED_DOMAINS = ["bbbc.broadinstitute.org"]  # Specify trusted domains to prevent malicious URL requests

# List of keywords to skip (can be dynamically filtered in the future)
skip_keywords = [
    "BBBC005", "BBBC006", "BBBC025", "BBBC042", "BBBC046", "BBBC048",
    "BBBC051", "BBBC017", "BBBC021", "BBBC022", "BBBC036", "BBBC037",
    "BBBC047", "BBBC024", "BBBC027",
]

def is_trusted_url(url):
    """
    Ensure the URL belongs to a trusted domain.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if URL belongs to a trusted domain, False otherwise.
    """
    return any(domain in url for domain in TRUSTED_DOMAINS)

def get_file_hash(file_content):
    """
    Calculate SHA256 hash of the file content for integrity verification.

    Args:
        file_content (bytes): The content of the file to hash.

    Returns:
        str: The SHA256 hash of the file content.
    """
    return hashlib.sha256(file_content).hexdigest()

def find_zip_links(url):
    """
    Find all links containing ZIP files within a given URL.

    Args:
        url (str): The URL to search for ZIP files.

    Returns:
        tuple: A dictionary of BBBC links and their ZIP files, 
               the count of links with ZIPs, and the count without ZIPs.
    """
    # Sessions reuses TCP connection to improve performance since making repeated API calls
    session = requests.Session()

    def get_page_links(url):
        """
        Fetch the content of a webpage.

        Args:
            url (str): The URL of the webpage.

        Returns:
            Response: The response object if successful, None otherwise.
        """
        if not is_trusted_url(url):
            logging.warning(f"Untrusted URL: {url}")
            return None

        try:
            response = session.get(url, verify=True)  # Enable SSL certificate verification
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logging.error(f"Failed to retrieve {url}: {e}")
            return None

    def extract_zip_links(url):
        """
        Extract all ZIP file links from a given webpage.

        Args:
            url (str): The URL of the webpage.

        Returns:
            list: A list of ZIP file URLs.
        """
        response = get_page_links(url)
        if response:
            soup = BeautifulSoup(response.content, "html.parser")
            inner_links = soup.find_all("a", href=True)
            zip_files = [
                urljoin(url, link["href"])
                for link in inner_links
                if link["href"].endswith(".zip") or ".zip?" in link["href"]
            ]
            return zip_files if zip_files else None
        return None

    # Fetch the main page
    response = get_page_links(url)
    if not response:
        return {}, 0, 0

    # Find all BBBC links
    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", href=True)
    bbbc_links = [
        urljoin(url, link["href"]) for link in links if link["href"].startswith("/BBBC")
    ]

    # Initialize results
    zip_links = {}
    links_with_zip = 0
    links_without_zip = 0

    # Use ThreadPoolExecutor for parallel processing of links
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_bbbc = {
            executor.submit(extract_zip_links, bbbc_link): bbbc_link
            for bbbc_link in bbbc_links
        }
        for future in as_completed(future_to_bbbc):
            bbbc_link = future_to_bbbc[future]
            try:
                zip_files = future.result()
                if zip_files:
                    zip_links[bbbc_link] = zip_files
                    links_with_zip += 1
                else:
                    links_without_zip += 1
            except Exception as e:
                logging.error(f"Error processing {bbbc_link}: {e}")
                links_without_zip += 1

    return zip_links, links_with_zip, links_without_zip


def download_and_extract_zip_files(zip_links, base_path, skip_keywords):
    """
    Download and extract ZIP files from the provided links.

    Args:
        zip_links (dict): A dictionary of BBBC links and their associated ZIP files.
        base_path (str): The base directory for downloaded datasets.
        skip_keywords (list): A list of keywords to skip while downloading.
    """
    os.makedirs(base_path, exist_ok=True)
    total_data_downloaded = 0  # Track total data downloaded

    # Filter out links with skip keywords
    filtered_zip_links = {
        k: v
        for k, v in zip_links.items()
        if not any(keyword in k for keyword in skip_keywords)
    }

    # Dataset-level progress bar
    with tqdm(
        total=len(filtered_zip_links), desc="Processing datasets", unit="dataset"
    ) as pbar:
        for dataset_url, zips in filtered_zip_links.items():
            dataset_name = dataset_url.strip("/").split("/")[-1]
            dataset_path = os.path.join(base_path, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
            dataset_data_downloaded = 0
            start_time = time.time()

            # File-level progress bar
            with tqdm(
                total=len(zips),
                desc=f"Downloading {dataset_name}",
                leave=True,
                unit="file",
            ) as file_pbar:
                for zip_url in zips:
                    try:
                        response = requests.get(zip_url, stream=True, verify=True)
                        response.raise_for_status()
                        data_size = int(response.headers.get("Content-Length", 0))
                        dataset_data_downloaded += data_size
                        total_data_downloaded += data_size

                        # Verify the hash of the downloaded file before extracting
                        file_hash = get_file_hash(response.content)
                        logging.info(f"Downloaded {zip_url}, SHA256 hash: {file_hash}")

                        with zipfile.ZipFile(BytesIO(response.content)) as z:
                            # Ensure no directory traversal when extracting files
                            for zip_info in z.infolist():
                                extracted_path = os.path.join(dataset_path, zip_info.filename)
                                if os.path.commonprefix([dataset_path, extracted_path]) != dataset_path:
                                    raise ValueError(f"Unsafe file path detected: {zip_info.filename}")

                            z.extractall(dataset_path)
                        file_pbar.update(1)  # Update file-level progress bar
                    except Exception as e:
                        logging.error(f"Failed to download or extract {zip_url}: {e}")

            end_time = time.time()
            time_taken = end_time - start_time

            # Log completion information
            logging.info(f"Completed {dataset_name} in {time_taken:.2f} seconds, "
                         f"downloaded {dataset_data_downloaded / (1024 * 1024):.2f} MB.")

            pbar.update(1)  # Update dataset-level progress bar
            file_pbar.n = file_pbar.total  # Ensure inner progress bar reaches 100%
            file_pbar.refresh()  # Refresh the display to show 100%

    logging.info(f"Total data downloaded: {total_data_downloaded / (1024 * 1024):.2f} MB.")


# Example usage
site_url = "https://bbbc.broadinstitute.org/image_sets"  # Main page URL
base_download_path = "datasets"

# Find and process ZIP links
zip_files_by_bbbc, count_with_zip, count_without_zip = find_zip_links(site_url)
download_and_extract_zip_files(zip_files_by_bbbc, base_download_path, skip_keywords)

# Output summary
logging.info(f"Links with at least one ZIP file: {count_with_zip}")
logging.info(f"Links with no ZIP files: {count_without_zip}")
