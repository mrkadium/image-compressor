from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct

file = "test_compressed"
ext = ".jpg"
output = file + '_compressed' + ext

# Functions for DCT, IDCT, quantization, and dequantization
def apply_dct(image_block):
    return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

def apply_idct(dct_block):
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Quantization Matrix (same as before)
quant_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Padding function to ensure image dimensions are multiples of 8
def pad_image(image_array):
    rows, cols = image_array.shape
    padded_rows = rows if rows % 8 == 0 else rows + (8 - rows % 8)
    padded_cols = cols if cols % 8 == 0 else cols + (8 - cols % 8)
    
    # Create a new padded image with zeros
    padded_image = np.zeros((padded_rows, padded_cols))
    # Copy the original image into the padded one
    padded_image[:rows, :cols] = image_array
    
    return padded_image

# Step 1: Load the image and separate the RGB channels
image = Image.open(file + ext).convert('RGB')
image = np.array(image)

# Extract the red, green, and blue channels
red_channel = image[:, :, 0]
green_channel = image[:, :, 1]
blue_channel = image[:, :, 2]

# Step 2: Function to compress a single channel (like R, G, or B)
def compress_channel(channel):
    rows, cols = channel.shape
    padded_channel = pad_image(channel)
    compressed_channel = np.zeros_like(padded_channel)
    
    # Apply DCT and quantization on 8x8 blocks
    for i in range(0, padded_channel.shape[0], 8):
        for j in range(0, padded_channel.shape[1], 8):
            block = padded_channel[i:i+8, j:j+8]
            dct_block = apply_dct(block)
            quantized_block = quantize(dct_block, quant_matrix)
            compressed_channel[i:i+8, j:j+8] = quantized_block
    
    return compressed_channel

# Step 3: Compress each channel
compressed_red = compress_channel(red_channel)
compressed_green = compress_channel(green_channel)
compressed_blue = compress_channel(blue_channel)

# Step 4: Function to decompress a single channel
def decompress_channel(compressed_channel, original_shape):
    rows, cols = original_shape
    reconstructed_channel = np.zeros_like(compressed_channel)
    
    # Apply IDCT and dequantization on 8x8 blocks
    for i in range(0, compressed_channel.shape[0], 8):
        for j in range(0, compressed_channel.shape[1], 8):
            quantized_block = compressed_channel[i:i+8, j:j+8]
            dequantized_block = dequantize(quantized_block, quant_matrix)
            idct_block = apply_idct(dequantized_block)
            reconstructed_channel[i:i+8, j:j+8] = np.round(idct_block)
    
    # Crop the padded area to original size
    return reconstructed_channel[:rows, :cols]

# Step 5: Decompress each channel
decompressed_red = decompress_channel(compressed_red, red_channel.shape)
decompressed_green = decompress_channel(compressed_green, green_channel.shape)
decompressed_blue = decompress_channel(compressed_blue, blue_channel.shape)

# Step 6: Merge the decompressed channels back into an RGB image
reconstructed_image = np.stack((decompressed_red, decompressed_green, decompressed_blue), axis=2)
reconstructed_image = np.clip(reconstructed_image, 0, 255)  # Ensure valid pixel values

# Step 7: Convert the result back to an image and save it
reconstructed_image = Image.fromarray(reconstructed_image.astype(np.uint8))
reconstructed_image.save(file + '_compressed' + ext)

print("Color image compression complete and saved as '" + output + "'.")
