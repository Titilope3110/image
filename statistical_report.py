import numpy as np
import os
import csv
from PIL import Image
from pathlib import Path
from rle_dct_model import load_image, preprocess_image, compress_image, decompress_image  # Adjust as needed

IMAGE_FOLDER = "image_compression"
OUTPUT_FOLDER = "compressed_results"
RESULTS_FILE = "compression_results.csv"

def mean_squared_error(original, reconstructed):
    return np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)

def peak_signal_to_noise_ratio(mse, max_pixel=255.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def compression_ratio_error(original_path, compressed_path):
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    return original_size / compressed_size

def evaluate():
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.png')]
    
    total_mse = 0
    total_psnr = 0
    total_cr = 0
    count = 0

    # Create results directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Open CSV file for writing
    with open(os.path.join(OUTPUT_FOLDER, RESULTS_FILE), 'w', newline='') as csvfile:
        fieldnames = ['Filename', 'MSE', 'PSNR (dB)', 'Compression Ratio']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for filename in image_files:
            print(f"\nEvaluating {filename}...")

            # Load image
            image_path = os.path.join(IMAGE_FOLDER, filename)
            img_array, mode = load_image(image_path)
            processed_img = preprocess_image(img_array, mode)

            # Compress
            compressed_data = compress_image(processed_img, mode)
            compressed_file = os.path.join(OUTPUT_FOLDER, f"compressed_{Path(filename).stem}.npz")
            np.savez_compressed(compressed_file, **compressed_data)

            # Decompress
            decompressed_img = decompress_image(compressed_data)

            # Resize for comparison if mode is RGB
            if mode == 'RGB':
                original = img_array.astype(np.uint8)
                reconstructed = decompressed_img.astype(np.uint8)
            else:
                original = img_array
                reconstructed = decompressed_img

            # Metrics
            mse = mean_squared_error(original, reconstructed)
            psnr = peak_signal_to_noise_ratio(mse)
            cr = compression_ratio_error(image_path, compressed_file)

            # Accumulate
            total_mse += mse
            total_psnr += psnr
            total_cr += cr
            count += 1

            # Write per-image result to CSV
            writer.writerow({
                'Filename': filename,
                'MSE': f"{mse:.2f}",
                'PSNR (dB)': f"{psnr:.2f}",
                'Compression Ratio': f"{cr:.2f}"
            })

    # Calculate and write averages if we processed any images
    if count > 0:
        avg_mse = total_mse / count
        avg_psnr = total_psnr / count
        avg_cr = total_cr / count
        
        # Append averages to the CSV file
        with open(os.path.join(OUTPUT_FOLDER, RESULTS_FILE), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Filename': 'AVERAGE',
                'MSE': f"{avg_mse:.2f}",
                'PSNR (dB)': f"{avg_psnr:.2f}",
                'Compression Ratio': f"{avg_cr:.2f}"
            })
    else:
        print("No PNG images found in the folder.")

if __name__ == "__main__":
    evaluate()