from PIL import Image
import numpy as np
from scipy.fftpack import dct, idct
print("1")

# Step 1: Load the image
image = Image.open('test.jpg')

# Step 2: Convert the image to grayscale
image = image.convert('L')  # 'L' mode is for grayscale

# Step 3: Convert the image to a NumPy array
image_array = np.array(image)

# Displaying some basic information about the image
print("Image shape:", image_array.shape)
print("Image array:")
print(image_array)

print(1)

# Functions for DCT and IDCT
def apply_dct(image_block):
    return dct(dct(image_block.T, norm='ortho').T, norm='ortho')

def apply_idct(dct_block):
    return idct(idct(dct_block.T, norm='ortho').T, norm='ortho')

def quantize(block, quant_matrix):
    return np.round(block / quant_matrix)

def dequantize(block, quant_matrix):
    return block * quant_matrix

# Quantization Matrix
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

# Step 1: Pad the image to ensure its dimensions are multiples of 8
def pad_image(image_array):
    rows, cols = image_array.shape
    padded_rows = rows if rows % 8 == 0 else rows + (8 - rows % 8)
    padded_cols = cols if cols % 8 == 0 else cols + (8 - cols % 8)
    
    padded_image = np.zeros((padded_rows, padded_cols))
    padded_image[:rows, :cols] = image_array  # Copy original image data into padded image
    
    return padded_image

padded_image = pad_image(image_array)
rows, cols = padded_image.shape

# Step 2: Now perform the DCT and compression on the padded image
compressed_image = np.zeros_like(padded_image)

for i in range(0, rows, 8):
    for j in range(0, cols, 8):
        block = padded_image[i:i+8, j:j+8]
        dct_block = apply_dct(block)
        quantized_block = quantize(dct_block, quant_matrix)
        compressed_image[i:i+8, j:j+8] = quantized_block

# Step 3: Reconstruct the image from the compressed data
reconstructed_image = np.zeros_like(padded_image)

for i in range(0, rows, 8):
    for j in range(0, cols, 8):
        quantized_block = compressed_image[i:i+8, j:j+8]
        dequantized_block = dequantize(quantized_block, quant_matrix)
        idct_block = apply_idct(dequantized_block)
        reconstructed_image[i:i+8, j:j+8] = np.round(idct_block)

# Step 4: Crop the padded areas to get back the original image size
reconstructed_image = reconstructed_image[:image_array.shape[0], :image_array.shape[1]]

# Step 5: Convert the result back to an image and save it
reconstructed_image = np.clip(reconstructed_image, 0, 255)
reconstructed_image = Image.fromarray(reconstructed_image.astype(np.uint8))
reconstructed_image.save('compressed_image.jpg')

print("Image compression complete and saved as 'compressed_image.jpg'.")
