# from manual_feature_extraction import (convert_images_to_grayscale, 
# rescale_intensity, apply_min_cross_entropy_threshold, smooth_and_enhance_edges,
# identify_objects5)
from mfe import (convert_images_to_grayscale, 
rescale_intensity, apply_min_cross_entropy_threshold, smooth_and_enhance_edges,
identify_objects5)

def main():
    # Directories for pipeline steps
    source_directory = r"test_images"
    grayscale_directory = r"test_output2/grayscale_images"
    rescaled_directory = r"test_output2/rescaled_images"
    thresholded_directory = r"test_output2/thresholded_images"
    smoothed_directory = r"test_output2/smoothed_images"
    objects_directory = r"test_output2/objects_image"

    # Step 1: Convert to grayscale
    print("Starting grayscale conversion...")
    convert_images_to_grayscale(source_directory, grayscale_directory)

    # Step 2: Rescale intensity
    print("Starting intensity rescaling...")
    rescale_intensity(grayscale_directory, rescaled_directory)

    # Step 3: Apply thresholding
    print("Starting thresholding with minimum cross-entropy...")
    apply_min_cross_entropy_threshold(rescaled_directory, thresholded_directory, smoothing_scale=0.0, threshold_correction=1.0)

    # Step 4: Smooth and enhance edges
    print("Starting Gaussian smoothing and edge enhancement...")
    smooth_and_enhance_edges(thresholded_directory, smoothed_directory, canny_factor=3)

    # Step 5: Identify objects and filter by size
    print("Starting object identification and filtering...")
    identify_objects5(smoothed_directory, objects_directory, min_diameter=50, max_diameter=120, smoothing_scale=1.3488, threshold_correction=1.0)
    
    print("Pipeline completed.")

if __name__ == "__main__":
    main()