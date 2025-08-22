import pandas as pd

# Read the existing meta.csv
meta = pd.read_csv('meta.csv')

# Rename columns to match the expected ones
meta.rename(columns={'image_path': 'original_image', 'mask_path': 'mask_image'}, inplace=True)

# Save the modified meta.csv
meta.to_csv('meta.csv', index=False)

print("meta.csv has been updated with the correct column names.")



