import streamlit as st
from PIL import Image
import os
import numpy as np
import pandas as pd
from rle_dct_model import load_image, apply_dct, apply_rle, save_image
from statistical_report import calculate_metrics


# Paths
INPUT_FOLDER = "image_compression"
OUTPUT_DCT = "compressed_outputs/dct"
OUTPUT_RLE = "compressed_outputs/rle"
REPORT_CSV = "reports/compression_stats.csv"

# Ensure output folders exist
os.makedirs(OUTPUT_DCT, exist_ok=True)
os.makedirs(OUTPUT_RLE, exist_ok=True)

# Streamlit config
st.set_page_config(page_title="📦 Image Compression (RLE vs DCT)", layout="wide")
st.title("📦 Image Compression with RLE & DCT")

# Load image list
image_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
selected_image = st.selectbox("Select an image to compress", image_files)

if selected_image:
    # Load and display original
    path = os.path.join(INPUT_FOLDER, selected_image)
    img_array, mode = load_image(path)

    st.subheader("Original Image")
    st.image(img_array, caption="Original", use_column_width=True)

    # ---------- Apply DCT ----------
    dct_array = apply_dct(img_array)
    dct_path = os.path.join(OUTPUT_DCT, selected_image)
    save_image(dct_array, dct_path)

    # ---------- Apply RLE ----------
    rle_array = apply_rle(img_array)
    rle_path = os.path.join(OUTPUT_RLE, selected_image)
    save_image(rle_array, rle_path)

    # ---------- Display Results ----------
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("DCT Result")
        st.image(dct_array.astype(np.uint8), use_column_width=True)

    with col2:
        st.subheader("RLE Result")
        st.image(rle_array.astype(np.uint8), use_column_width=True)

    # ---------- Statistical Metrics ----------
    st.subheader("📊 Statistical Comparison")

    dct_metrics = calculate_metrics(path, dct_path)
    rle_metrics = calculate_metrics(path, rle_path)

    col3, col4 = st.columns(2)
    with col3:
        st.write("**DCT Metrics**")
        st.json(dct_metrics)

    with col4:
        st.write("**RLE Metrics**")
        st.json(rle_metrics)

# ------------------- Batch Evaluation Report -------------------
st.markdown("---")
st.subheader("📄 Batch Evaluation Report")

if os.path.exists(REPORT_CSV):
    df = pd.read_csv(REPORT_CSV)

    # Filters
    col5, col6 = st.columns(2)
    with col5:
        method_filter = st.selectbox("Filter by Method", ["All"] + sorted(df["Method"].unique().tolist()))
    with col6:
        mode_filter = st.selectbox("Filter by Image Mode", ["All"] + sorted(df["Mode"].unique().tolist()))

    # Apply filters
    filtered_df = df.copy()
    if method_filter != "All":
        filtered_df = filtered_df[filtered_df["Method"] == method_filter]
    if mode_filter != "All":
        filtered_df = filtered_df[filtered_df["Mode"] == mode_filter]

    st.dataframe(filtered_df, use_container_width=True)

    # Optional: Download CSV
    st.download_button(
        label="📥 Download CSV Report",
        data=filtered_df.to_csv(index=False).encode('utf-8'),
        file_name="compression_stats_filtered.csv",
        mime='text/csv'
    )
else:
    st.warning("No batch report found. Please run `batch_evaluation.py` to generate it.")
