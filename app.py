# app.py
import streamlit as st
import os
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path

from rle_dct_model import load_image, preprocess_image, compress_image, decompress_image, display_images

IMAGE_FOLDER = "image_compression"
COMPRESSED_FOLDER = "compressed_results"
CSV_REPORT = os.path.join(COMPRESSED_FOLDER, "compression_results.csv")

st.set_page_config(page_title="RLE-DCT Image Compression", layout="wide")

st.title("📦 RLE + DCT Image Compression App")
st.markdown("Upload or select an image to compress using DCT and RLE, then view the results and stats.")

# =======================
# Helper Functions
# =======================

def compute_metrics(original, reconstructed, original_path, compressed_path):
    mse = np.mean((original.astype(np.float32) - reconstructed.astype(np.float32)) ** 2)
    psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse != 0 else float('inf')
    cr = os.path.getsize(original_path) / os.path.getsize(compressed_path)
    return mse, psnr, cr

def display_side_by_side(original, reconstructed):
    col1, col2 = st.columns(2)
    with col1:
        st.image(original, caption="Original Image", use_column_width=True)
    with col2:
        st.image(reconstructed, caption="Reconstructed Image", use_column_width=True)

# =======================
# Image Selection
# =======================

uploaded_image = st.file_uploader("📤 Upload an image (PNG only)", type=["png"])

# Show from local folder if no upload
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(".png")]
selected_image = None

if not uploaded_image:
    selected = st.selectbox("📁 Or select an image from folder:", image_files)
    if selected:
        selected_image = os.path.join(IMAGE_FOLDER, selected)
else:
    selected_image = uploaded_image

# =======================
# Compression Section
# =======================

if selected_image:
    st.subheader("🛠 Compression Result")
    
    # Save uploaded image temporarily
    if isinstance(selected_image, Path) or isinstance(selected_image, str):
        image_path = selected_image
    else:
        image_path = os.path.join(IMAGE_FOLDER, uploaded_image.name)
        with open(image_path, "wb") as f:
            f.write(uploaded_image.read())

    # Load and compress
    image_array, mode = load_image(image_path)
    processed_img = preprocess_image(image_array, mode)
    compressed = compress_image(processed_img, mode)
    
    compressed_file = os.path.join(COMPRESSED_FOLDER, f"compressed_{Path(image_path).stem}.npz")
    np.savez_compressed(compressed_file, **compressed)
    
    # Decompress
    compressed['channels'] = compressed['channels']  # Ensure structure matches
    reconstructed = decompress_image(compressed)

    # Display images
    display_side_by_side(image_array, reconstructed)

    # Show metrics
    st.subheader("📊 Compression Metrics")
    mse, psnr, cr = compute_metrics(image_array, reconstructed, image_path, compressed_file)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("PSNR (dB)", f"{psnr:.2f}")
    col3.metric("Compression Ratio", f"{cr:.2f}")

# =======================
# DataFrame Viewer
# =======================

st.subheader("📄 Statistical Report (CSV to DataFrame)")
if os.path.exists(CSV_REPORT):
    df = pd.read_csv(CSV_REPORT)
    st.dataframe(df)
else:
    st.warning("No report found. Run `statistical_report.py` first to generate the CSV.")
