# rle_dct_model.py
import os
import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct

def load_image(path):
    img = Image.open(path)
    img = img.convert('RGB') if img.mode != 'RGB' else img
    return np.array(img), img.mode

def save_image(array, path):
    img = Image.fromarray(np.uint8(array))
    img.save(path)

# ----------- DCT Compression ------------
def apply_dct(image_array):
    image_dct = dct(dct(image_array.astype(float), axis=0, norm='ortho'), axis=1, norm='ortho')
    image_idct = idct(idct(image_dct, axis=1, norm='ortho'), axis=0, norm='ortho')
    return np.clip(image_idct, 0, 255)

# ----------- RLE Compression ------------
def rle_encode(img_array):
    flat = img_array.flatten()
    encoding = []
    prev_pixel = flat[0]
    count = 1

    for pixel in flat[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoding.extend([prev_pixel, count])
            prev_pixel = pixel
            count = 1
    encoding.extend([prev_pixel, count])
    return encoding

def rle_decode(encoding, shape):
    flat = []
    for i in range(0, len(encoding), 2):
        flat.extend([encoding[i]] * encoding[i+1])
    return np.array(flat, dtype=np.uint8).reshape(shape)

def apply_rle(image_array):
    if len(image_array.shape) == 3:
        channels = [rle_decode(rle_encode(image_array[..., c]), image_array[..., c].shape) for c in range(3)]
        return np.stack(channels, axis=-1)
    else:
        return rle_decode(rle_encode(image_array), image_array.shape)
