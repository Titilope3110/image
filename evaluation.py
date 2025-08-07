import os
from PIL import Image
import numpy as np
import pandas as pd
from rle_dct_model import load_image, apply_dct, apply_rle, save_image
from statistical_report import calculate_metrics

INPUT_FOLDER = "image_compression"
OUTPUT_DCT = "compressed_outputs/dct"
OUTPUT_RLE = "compressed_outputs/rle"
REPORT_FOLDER = "reports"
os.makedirs(REPORT_FOLDER, exist_ok=True)
# Ensure output folders exist
os.makedirs(OUTPUT_DCT, exist_ok=True)
os.makedirs(OUTPUT_RLE, exist_ok=True)


def is_grayscale(img):
    return len(img.shape) == 2 or (img.shape[2] == 3 and np.all(img[..., 0] == img[..., 1]) and np.all(img[..., 1] == img[..., 2]))

def process_all_images():
    results = []

    for filename in os.listdir(INPUT_FOLDER):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(INPUT_FOLDER, filename)
        img_array, _ = load_image(img_path)

        # Determine color mode
        mode = "Grayscale" if is_grayscale(img_array) else "Color"

        # Apply and save DCT
        dct_array = apply_dct(img_array)
        dct_path = os.path.join(OUTPUT_DCT, filename)
        save_image(dct_array, dct_path)
        dct_metrics = calculate_metrics(img_path, dct_path)
        dct_metrics["Method"] = "DCT"
        dct_metrics["Mode"] = mode
        results.append(dct_metrics)

        # Apply and save RLE
        rle_array = apply_rle(img_array)
        rle_path = os.path.join(OUTPUT_RLE, filename)
        save_image(rle_array, rle_path)
        rle_metrics = calculate_metrics(img_path, rle_path)
        rle_metrics["Method"] = "RLE"
        rle_metrics["Mode"] = mode
        results.append(rle_metrics)

    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(REPORT_FOLDER, "compression_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Report saved to: {csv_path}")

if __name__ == "__main__":
    process_all_images()
