import os
import pandas as pd

# Path to your image and noisy image directories
image_dir = '/data/home/bt241032/Diffusion_Models/data_Image/Image'  # Change this to your image directory
mask_dir = '/data/home/bt241032/Diffusion_Models/data_Image/Mask'    # Change this to your noisy image directory

# Lists to hold image paths and mask paths (noisy images)
image_paths = []
mask_paths = []

# Loop through images and masks (assuming they are named similarly)
for image_filename in os.listdir(image_dir):
    # Check if the noisy image (mask) exists with the same name
    mask_filename = image_filename  # Assuming mask has the same name as the image
    image_path = os.path.join(image_dir, image_filename)
    mask_path = os.path.join(mask_dir, mask_filename)  # Assuming same name

    # Check if the mask file exists
    if os.path.exists(mask_path):
        image_paths.append(image_path)
        mask_paths.append(mask_path)  # Noisy image is the mask for denoising task

# Create a DataFrame with paths for image and noisy (mask) images
meta_df = pd.DataFrame({
    'image_path': image_paths,
    'mask_path': mask_paths
})

# Save the dataframe to a CSV file
meta_df.to_csv('meta.csv', index=False)

print("meta.csv file has been created.")
