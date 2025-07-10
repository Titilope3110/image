import os
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Import necessary decompress functions from your model script
from rle_dct_model import load_image, decompress_image, display_images  # ensure model.py has these exposed

IMAGE_FOLDER = "image_compression"
COMPRESSED_FOLDER = "compressed_results"

def list_images():
    images = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".png")]
    if not images:
        print(f"No PNG images found in {IMAGE_FOLDER}")
        return []
    print("\nAvailable Images:")
    for idx, img in enumerate(images):
        print(f"{idx + 1}: {img}")
    return images

def select_image(images):
    idx = int(input("\nEnter the image number to compare: ")) - 1
    if 0 <= idx < len(images):
        return images[idx]
    else:
        print("Invalid selection.")
        return None

def compare_selected_image(image_name):
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    compressed_path = os.path.join(COMPRESSED_FOLDER, f"compressed_{Path(image_name).stem}.npz")

    if not os.path.exists(compressed_path):
        print(f"Compressed file not found for {image_name}")
        return

    # Load original image
    original_img, _ = load_image(image_path)

    # Load compressed data
    compressed_data = dict(np.load(compressed_path, allow_pickle=True))
    # Fix structure to match what decompress_image expects
    compressed_data['channels'] = compressed_data['channels'].item()

    # Decompress
    reconstructed = decompress_image(compressed_data)

    # Display comparison
    display_images(original_img, reconstructed)

if __name__ == "__main__":
    images = list_images()
    if images:
        selected = select_image(images)
        if selected:
            compare_selected_image(selected)
