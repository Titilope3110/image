"""
Complete Image Compression for breast images using DCT and RLE
Handles both color and grayscale PNG images
"""

import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import matplotlib.pyplot as plt
import os
from pathlib import Path

# ========================
# 1. CONSTANTS & PARAMETERS
# ========================
BLOCK_SIZE = 8
QUALITY = 75  # Higher quality for medical images
IMAGE_FOLDER = "image_compression"
OUTPUT_FOLDER = "compressed_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Standard JPEG quantization matrices
Q_MATRIX_LUMA = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

Q_MATRIX_CHROMA = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
])

# ========================
# 2. CORE TRANSFORM FUNCTIONS
# ========================

def dct_2d(block):
    """Apply 2D DCT to an 8x8 block"""
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct_2d(block):
    """Apply 2D inverse DCT to an 8x8 block"""
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# ========================
# 3. IMAGE HANDLING FUNCTIONS
# ========================

def load_image(image_path):
    """Load image, handle both color and grayscale"""
    img = Image.open(image_path)
    if img.mode == 'L':
        return np.array(img), 'L'
    elif img.mode in ('RGB', 'RGBA'):
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        return np.array(img), 'RGB'
    else:
        raise ValueError(f"Unsupported image mode: {img.mode}")

def preprocess_image(image, mode):
    """Preprocess based on image mode (color/grayscale)"""
    if mode == 'L':
        image = image.astype(np.float32) - 128
        height, width = image.shape
    else:
        image = image.astype(np.float32)
        # Convert RGB to YCbCr
        ycbcr = np.zeros_like(image)
        ycbcr[..., 0] = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]  # Y
        ycbcr[..., 1] = 128 - 0.168736 * image[..., 0] - 0.331264 * image[..., 1] + 0.5 * image[..., 2]  # Cb
        ycbcr[..., 2] = 128 + 0.5 * image[..., 0] - 0.418688 * image[..., 1] - 0.081312 * image[..., 2]  # Cr
        ycbcr -= 128
        height, width, _ = ycbcr.shape
    
    # Pad image if needed
    pad_height = (BLOCK_SIZE - height % BLOCK_SIZE) % BLOCK_SIZE
    pad_width = (BLOCK_SIZE - width % BLOCK_SIZE) % BLOCK_SIZE
    
    if mode == 'L':
        if pad_height != 0 or pad_width != 0:
            image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='edge')
        return image
    else:
        if pad_height != 0 or pad_width != 0:
            ycbcr = np.pad(ycbcr, ((0, pad_height), (0, pad_width), (0, 0)), mode='edge')
        return ycbcr

# ========================
# 4. COMPRESSION FUNCTIONS
# ========================

def adjust_quantization_matrix(base_matrix, quality):
    """Adjust quantization matrix based on quality setting"""
    if quality <= 0:
        quality = 1
    if quality > 100:
        quality = 100
    
    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - 2 * quality
    
    adjusted_matrix = np.floor((base_matrix * scale + 50) / 100)
    adjusted_matrix = np.clip(adjusted_matrix, 1, 255)
    return adjusted_matrix

def zigzag_scan(block):
    """Convert 8x8 block to zigzag 1D array"""
    zigzag = np.zeros(BLOCK_SIZE * BLOCK_SIZE)
    rows, cols = block.shape
    i = 0
    
    for s in range(rows + cols - 1):
        if s % 2 == 0:  # Moving up
            row = min(s, rows - 1)
            col = max(0, s - row)
            while row >= 0 and col < cols:
                zigzag[i] = block[row, col]
                i += 1
                row -= 1
                col += 1
        else:  # Moving down
            col = min(s, cols - 1)
            row = max(0, s - col)
            while col >= 0 and row < rows:
                zigzag[i] = block[row, col]
                i += 1
                row += 1
                col -= 1
    return zigzag

def run_length_encode(data):
    """Compress data using run-length encoding"""
    encoded = []
    count = 1
    
    for i in range(1, len(data)):
        if data[i] == data[i-1]:
            count += 1
        else:
            encoded.append((data[i-1], count))
            count = 1
    
    encoded.append((data[-1], count))
    return encoded

def process_channel(channel, quality, is_luma=True):
    """Process a single channel (Y, Cb, or Cr)"""
    height, width = channel.shape
    compressed_blocks = []
    
    for i in range(0, height, BLOCK_SIZE):
        for j in range(0, width, BLOCK_SIZE):
            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            dct_block = dct_2d(block)
            q_matrix = Q_MATRIX_LUMA if is_luma else Q_MATRIX_CHROMA
            adjusted_q = adjust_quantization_matrix(q_matrix, quality)
            quantized_block = np.round(dct_block / adjusted_q)
            zigzag_data = zigzag_scan(quantized_block)
            rle_data = run_length_encode(zigzag_data)
            compressed_blocks.append(rle_data)
    
    return compressed_blocks

def compress_image(image, mode, quality=QUALITY):
    """Main compression pipeline"""
    if mode == 'L':
        compressed_data = {
            'mode': 'L',
            'quality': quality,
            'original_shape': image.shape,
            'channels': {
                'Y': process_channel(image, quality, is_luma=True)
            }
        }
    else:
        compressed_data = {
            'mode': 'RGB',
            'quality': quality,
            'original_shape': image.shape[:2],
            'channels': {
                'Y': process_channel(image[..., 0], quality, is_luma=True),
                'Cb': process_channel(image[..., 1], quality, is_luma=False),
                'Cr': process_channel(image[..., 2], quality, is_luma=False)
            }
        }
    return compressed_data

# ========================
# 5. DECOMPRESSION FUNCTIONS
# ========================

def run_length_decode(encoded_data):
    """Decode RLE data"""
    decoded = []
    for (value, count) in encoded_data:
        decoded.extend([value] * count)
    return np.array(decoded)

def inverse_zigzag(zigzag_data, block_size=BLOCK_SIZE):
    """Convert zigzag 1D array back to 2D block"""
    block = np.zeros((block_size, block_size))
    rows, cols = block.shape
    i = 0
    
    for s in range(rows + cols - 1):
        if s % 2 == 0:  # Moving up
            row = min(s, rows - 1)
            col = max(0, s - row)
            while row >= 0 and col < cols and i < len(zigzag_data):
                block[row, col] = zigzag_data[i]
                i += 1
                row -= 1
                col += 1
        else:  # Moving down
            col = min(s, cols - 1)
            row = max(0, s - col)
            while col >= 0 and row < rows and i < len(zigzag_data):
                block[row, col] = zigzag_data[i]
                i += 1
                row += 1
                col -= 1
    return block

def reconstruct_channel(compressed_blocks, original_shape, quality, is_luma=True):
    """Reconstruct a single channel from compressed data"""
    height, width = original_shape
    pad_height = (BLOCK_SIZE - height % BLOCK_SIZE) % BLOCK_SIZE
    pad_width = (BLOCK_SIZE - width % BLOCK_SIZE) % BLOCK_SIZE
    padded_height = height + pad_height
    padded_width = width + pad_width
    
    channel = np.zeros((padded_height, padded_width))
    block_index = 0
    
    q_matrix = Q_MATRIX_LUMA if is_luma else Q_MATRIX_CHROMA
    adjusted_q = adjust_quantization_matrix(q_matrix, quality)
    
    for i in range(0, padded_height, BLOCK_SIZE):
        for j in range(0, padded_width, BLOCK_SIZE):
            if block_index >= len(compressed_blocks):
                break
            
            rle_data = compressed_blocks[block_index]
            zigzag_data = run_length_decode(rle_data)
            quantized_block = inverse_zigzag(zigzag_data)
            dct_block = quantized_block * adjusted_q
            block = idct_2d(dct_block)
            channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = block
            block_index += 1
    
    return channel[:height, :width]

def decompress_image(compressed_data):
    """Main decompression pipeline"""
    mode = compressed_data['mode']
    quality = compressed_data['quality']
    original_shape = compressed_data['original_shape']
    
    if mode == 'L':
        # Grayscale reconstruction
        y_channel = reconstruct_channel(
            compressed_data['channels']['Y'],
            original_shape,
            quality,
            is_luma=True
        )
        reconstructed = y_channel + 128
        reconstructed = np.clip(reconstructed, 0, 255)
        return reconstructed.astype(np.uint8)
    else:
        # Color reconstruction (YCbCr to RGB)
        y_channel = reconstruct_channel(
            compressed_data['channels']['Y'],
            original_shape,
            quality,
            is_luma=True
        )
        cb_channel = reconstruct_channel(
            compressed_data['channels']['Cb'],
            original_shape,
            quality,
            is_luma=False
        )
        cr_channel = reconstruct_channel(
            compressed_data['channels']['Cr'],
            original_shape,
            quality,
            is_luma=False
        )
        
        # Combine channels and convert back to RGB
        ycbcr = np.stack([y_channel, cb_channel, cr_channel], axis=-1) + 128
        ycbcr = np.clip(ycbcr, 0, 255)
        
        # Convert YCbCr to RGB
        rgb = np.zeros_like(ycbcr)
        rgb[..., 0] = ycbcr[..., 0] + 1.402 * (ycbcr[..., 2] - 128)  # R
        rgb[..., 1] = ycbcr[..., 0] - 0.344136 * (ycbcr[..., 1] - 128) - 0.714136 * (ycbcr[..., 2] - 128)  # G
        rgb[..., 2] = ycbcr[..., 0] + 1.772 * (ycbcr[..., 1] - 128)  # B
        
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(np.uint8)

# ========================
# 6. VISUALIZATION & BATCH PROCESSING
# ========================

def display_images(original, compressed, original_title="Original", compressed_title="Compressed"):
    """Display original and compressed images side by side"""
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    if original.ndim == 2:
        plt.imshow(original, cmap='gray')
    else:
        plt.imshow(original)
    plt.title(original_title)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    if compressed.ndim == 2:
        plt.imshow(compressed, cmap='gray')
    else:
        plt.imshow(compressed)
    plt.title(compressed_title)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_image_folder():
    """Process all images in the folder"""
    image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith('.png')]
    
    if not image_files:
        print(f"No PNG images found in {IMAGE_FOLDER}")
        return
    
    for filename in image_files:
        try:
            print(f"\nProcessing {filename}...")
            image_path = os.path.join(IMAGE_FOLDER, filename)
            
            # Load and preprocess
            img_array, mode = load_image(image_path)
            processed_img = preprocess_image(img_array, mode)
            
            # Compress
            compressed_data = compress_image(processed_img, mode)
            
            # Save compressed data
            output_path = os.path.join(OUTPUT_FOLDER, f"compressed_{Path(filename).stem}.npz")
            np.savez_compressed(output_path, **compressed_data)
            print(f"Saved compressed data to {output_path}")
            
            # Decompress and show first image for verification
            if filename == image_files[0]:
                decompressed_img = decompress_image(compressed_data)
                display_images(img_array, decompressed_img)
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

if __name__ == "__main__":
    print("Starting breast image compression...")
    process_image_folder()
    print("\nProcessing complete!")