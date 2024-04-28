import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.signal import convolve2d

# Implementation of the Perona-Malik paper's
# Anisotropic Diffusion!

def apply_canny_edge_detector(image, low_threshold=50, high_threshold=150):
    if image.dtype != np.uint8:
        image = np.uint8(image * 255)

    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges

def g(diff, k):
    # This function calculates the gradient modulus function, which controls diffusion rate
    return np.exp(-(np.abs(diff) ** 2) / (k ** 2)).astype(np.float32)

def aniso_diff_channel(channel, iter, l, k):
    # Convert the input channel to float32 to prevent type mismatch during calculations
    tmp = np.float32(channel)

    # Define kernels to find gradients in four directions
    n_ker = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]], dtype=np.float32)
    e_ker = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]], dtype=np.float32)
    w_ker = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]], dtype=np.float32)
    s_ker = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]], dtype=np.float32)

    for i in range(iter):
        n_diff = convolve2d(tmp, n_ker, mode='same', boundary='symm')
        e_diff = convolve2d(tmp, e_ker, mode='same', boundary='symm')
        s_diff = convolve2d(tmp, s_ker, mode='same', boundary='symm')
        w_diff = convolve2d(tmp, w_ker, mode='same', boundary='symm')

        c_n = g(n_diff, k)
        c_e = g(e_diff, k)
        c_s = g(s_diff, k)
        c_w = g(w_diff, k)
        
        
        # Ensure tmp remains float32 throughout all operations
        tmp = tmp + l * (c_n * n_diff + c_e * e_diff + c_s * s_diff + c_w * w_diff)

    # Convert the float32 image back to uint8 for proper image format
    return np.clip(tmp, 0, 255).astype(np.uint8)

def aniso_diff_color(in_img, iter, l, k):
    if len(in_img.shape) == 2:
        return aniso_diff_channel(in_img, iter, l, k)
    elif len(in_img.shape) == 3:
        channels = cv2.split(in_img)
        processed_channels = [aniso_diff_channel(ch, iter, l, k) for ch in channels]
        return cv2.merge(processed_channels)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    print(image.shape)

    # Parameters
    nIterations = 10  # Number of iterations
    LAMBDA = 0.25   # Lambda, the integration constant
    k = 15     # K, edge threshold parameter

    # Perform anisotropic diffusion on color image
    processed_image = aniso_diff_color(image, nIterations, LAMBDA, k)

    # Process the image with anisotropic diffusion from cv2
    #processed_image_check = cv2.ximgproc.anisotropicDiffusion(np.copy(image), LAMBDA, K, nIterations)

    # Plot the original and processed images
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(221)
    plt.title('Original')
    plt.imshow(image, cmap='gray')
    fig.add_subplot(222)
    plt.title(f'After {nIterations} Iterations')
    plt.imshow(processed_image, cmap='gray')

    # Apply Harris corner detection to both images
    image_with_edges = apply_canny_edge_detector(np.copy(image))
    processed_with_edges = apply_canny_edge_detector(np.copy(processed_image))

    # Plot images with Harris corners
    fig.add_subplot(223)
    plt.title('Original with Harris Corners')
    plt.imshow(image_with_edges, cmap='gray')
    fig.add_subplot(224)
    plt.title('Processed with Harris Corners')
    plt.imshow(processed_with_edges, cmap='gray')

    plt.show()

