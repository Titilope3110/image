import numpy as np
import os
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from PIL import Image

def calculate_metrics(original_path, compressed_path):
    original = np.array(Image.open(original_path).convert('L'))
    compressed = np.array(Image.open(compressed_path).convert('L'))
    
    mse = mean_squared_error(original, compressed)
    psnr = peak_signal_noise_ratio(original, compressed)
    
    original_size = os.path.getsize(original_path)
    compressed_size = os.path.getsize(compressed_path)
    compression_ratio = original_size / compressed_size if compressed_size != 0 else 0

    return {
        "Image": os.path.basename(original_path),
        "MSE": mse,
        "PSNR": psnr,
        "Compression Ratio": compression_ratio
    }
