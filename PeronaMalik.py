import sys
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Implementation of the Perona-Malik paper -
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
        n_diff = cv2.filter2D(tmp, -1, n_ker, borderType=cv2.BORDER_REFLECT)
        e_diff = cv2.filter2D(tmp, -1, e_ker, borderType=cv2.BORDER_REFLECT)
        s_diff = cv2.filter2D(tmp, -1, s_ker, borderType=cv2.BORDER_REFLECT)
        w_diff = cv2.filter2D(tmp, -1, w_ker, borderType=cv2.BORDER_REFLECT)

        c_n = g(n_diff, k)
        c_e = g(e_diff, k)
        c_s = g(s_diff, k)
        c_w = g(w_diff, k)
        
        # Update tmp with the diffusion calculated using the conductance terms
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
    nIterations = 50 # Number of iterations
    LAMBDA = 0.25 # Lambda, needs to be between 0 -> 1/4
    k = 15 # K, edge threshold parameter

    # Perform anisotropic diffusion on color image
    processed_image = aniso_diff_color(image, nIterations, LAMBDA, k)

    # Process the image with anisotropic diffusion from cv2
    #processed_image_check = cv2.ximgproc.anisotropicDiffusion(np.copy(image), LAMBDA, K, nIterations)

    # Plot the original and processed images
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].set_title('Original')
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 1].set_title(f'After {nIterations} Iterations')
    axs[0, 1].imshow(processed_image, cmap='gray')

    # Apply Canny (edge) detection to both images
    image_with_edges = apply_canny_edge_detector(np.copy(image))
    processed_with_edges = apply_canny_edge_detector(np.copy(processed_image))

    # Plot images with Canny
    axs[1, 0].set_title('Original with Canny')
    axs[1, 0].imshow(image_with_edges, cmap='gray')
    axs[1, 1].set_title('Processed with Canny')
    axs[1, 1].imshow(processed_with_edges, cmap='gray')

    plt.tight_layout()
    plt.show()