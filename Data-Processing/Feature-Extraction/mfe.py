import os
import cv2
import numpy as np
import logging
from skimage import measure

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Default log level
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Outputs to console
        logging.FileHandler('image_processing.log')  # Also writes to a log file
    ]
)

def convert_images_to_grayscale(source_directory, output_directory):
    """Converts all images in the source directory to grayscale."""
    os.makedirs(output_directory, exist_ok=True)

    images_to_save = []
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image = cv2.imread(file_path)
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images_to_save.append((file_name, gray_image))
            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")

    # Write all processed grayscale images to disk at once
    for file_name, gray_image in images_to_save:
        gray_file_path = os.path.join(output_directory, file_name)
        cv2.imwrite(gray_file_path, gray_image)
        logging.info(f"Processed and saved grayscale image: {gray_file_path}")

    logging.info("Grayscale conversion completed.")

def rescale_intensity(source_directory, output_directory):
    """Rescales the intensity of grayscale images."""
    os.makedirs(output_directory, exist_ok=True)

    images_to_save = []
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                # Use numpy for rescaling
                rescaled_image = (image / 255.0).astype(np.float32)
                rescaled_image = (rescaled_image * 255).astype(np.uint8)
                images_to_save.append((file_name, rescaled_image))
            except Exception as e:
                logging.error(f"Failed to rescale {file_name}: {e}")

    # Write all processed rescaled images to disk at once
    for file_name, rescaled_image in images_to_save:
        rescaled_file_path = os.path.join(output_directory, file_name)
        cv2.imwrite(rescaled_file_path, rescaled_image)
        logging.info(f"Rescaled and saved image: {rescaled_file_path}")

    logging.info("Intensity rescaling completed.")

def smooth_and_enhance_edges(source_directory, output_directory, canny_factor=3):
    """Applies Gaussian smoothing followed by edge enhancement using the Canny method."""
    os.makedirs(output_directory, exist_ok=True)

    images_to_save = []
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Read grayscale image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Apply Gaussian smoothing using numpy
                smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

                # Use numpy for computing Canny thresholds
                median_intensity = np.median(smoothed_image)
                lower_threshold = int(max(0, (1.0 - canny_factor / 10.0) * median_intensity))
                upper_threshold = int(min(255, (1.0 + canny_factor / 10.0) * median_intensity))

                # Apply Canny edge detection
                edges = cv2.Canny(smoothed_image, lower_threshold, upper_threshold)

                images_to_save.append((file_name, edges))
            except Exception as e:
                logging.error(f"Failed to process {file_name}: {e}")

    # Write all processed edge-enhanced images to disk at once
    for file_name, edges in images_to_save:
        processed_file_path = os.path.join(output_directory, file_name)
        cv2.imwrite(processed_file_path, edges)
        logging.info(f"Smoothed and enhanced edges for: {processed_file_path}")

    logging.info("Smoothing and edge enhancement completed.")

def apply_min_cross_entropy_threshold(source_directory, output_directory, smoothing_scale=0.0, threshold_correction=0.4):
    """Applies global thresholding using the minimum cross-entropy method."""
    os.makedirs(output_directory, exist_ok=True)

    images_to_save = []
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Load the grayscale image
                image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Compute histogram using numpy
                hist = np.histogram(image, bins=256, range=(0, 256))[0]
                
                # Normalize histogram to probabilities
                hist_prob = hist / hist.sum()

                # Calculate cumulative sums
                cumulative_sum = np.cumsum(hist_prob)
                cumulative_mean = np.cumsum(hist_prob * np.arange(256))

                # Global mean
                global_mean = cumulative_mean[-1]

                # Compute minimum cross-entropy threshold using numpy operations
                cross_entropy = -np.nan_to_num(
                    cumulative_mean * np.log(cumulative_mean / (cumulative_sum + 1e-10))
                    + (global_mean - cumulative_mean) * np.log((global_mean - cumulative_mean) / (1 - cumulative_sum + 1e-10))
                )

                optimal_threshold = np.argmin(cross_entropy)
                corrected_threshold = optimal_threshold * threshold_correction

                # Apply threshold using numpy
                thresholded_image = np.where(image > corrected_threshold, 255, 0).astype(np.uint8)

                images_to_save.append((file_name, thresholded_image))
            except Exception as e:
                logging.error(f"Failed to threshold {file_name}: {e}")

    # Write all thresholded images to disk at once
    for file_name, thresholded_image in images_to_save:
        thresholded_file_path = os.path.join(output_directory, file_name)
        cv2.imwrite(thresholded_file_path, thresholded_image)
        logging.info(f"Applied threshold and saved image: {thresholded_file_path}")

    logging.info("Thresholding completed.")

def identify_objects5(
    source_directory,
    output_directory,
    min_diameter=200,  # Min diameter for objects
    max_diameter=250,  # Max diameter for objects
    smoothing_scale=1.3488,
    threshold_correction=1.0
):
    os.makedirs(output_directory, exist_ok=True)
    images_to_save = []
    for file_name in os.listdir(source_directory):
        file_path = os.path.join(source_directory, file_name)

        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            try:
                # Read the edged image (grayscale)
                edged_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

                # Convert to 3-channel image (RGB) for watershed
                edged_image_rgb = cv2.cvtColor(edged_image, cv2.COLOR_GRAY2BGR)

                # Apply Gaussian smoothing using numpy
                smoothed_image = cv2.GaussianBlur(edged_image, (5, 5), smoothing_scale)

                # Apply Otsu's thresholding method using numpy
                _, otsu_threshold = cv2.threshold(
                    smoothed_image, 0, 255, 
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )

                # Apply erosion followed by dilation to separate close objects and clean up noise
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                eroded_image = cv2.erode(otsu_threshold, kernel, iterations=1)
                dilated_image = cv2.dilate(eroded_image, kernel, iterations=5)

                # Apply Watershed segmentation to handle overlapping or touching cells
                dist_transform = cv2.distanceTransform(dilated_image, cv2.DIST_L2, 5)
                _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

                sure_bg = cv2.dilate(dilated_image, kernel, iterations=3)

                sure_fg = np.uint8(sure_fg)
                sure_bg = np.uint8(sure_bg)

                unknown = cv2.subtract(sure_bg, sure_fg)

                _, markers = cv2.connectedComponents(sure_fg)
                markers = markers + 1
                markers[unknown == 255] = 0
                markers = np.int32(markers)

                cv2.watershed(edged_image_rgb, markers)
                edged_image_rgb[markers == -1] = [255, 0, 0]  # Mark boundaries with red

                # Find connected components (objects)
                labeled_image = measure.label(dilated_image > 0, connectivity=2)
                properties = measure.regionprops(labeled_image)

                min_area = np.pi * (min_diameter / 2) ** 2
                max_area = np.pi * (max_diameter / 2) ** 2

                for region in properties:
                    if min_area <= region.area <= max_area:
                        minr, minc, maxr, maxc = region.bbox
                        cv2.rectangle(edged_image_rgb, (minc, minr), (maxc, maxr), (0, 255, 0), 2)
                    else:
                        logging.debug(f"Region area {region.area} is out of bounds (min_area: {min_area}, max_area: {max_area})")


                images_to_save.append((file_name, edged_image_rgb))
            except Exception as e:
                logging.error(f"Failed to identify objects in {file_name}: {e}")

    # Write all processed object-identified images to disk at once
    for file_name, output_image in images_to_save:
        output_file_path = os.path.join(output_directory, file_name)
        cv2.imwrite(output_file_path, output_image)
        logging.info(f"Processed and saved object-identified image: {output_file_path}")

    logging.info("Object identification completed.")
