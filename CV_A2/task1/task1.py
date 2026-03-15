import os.path as osp
import cv2
import numpy as np

def gaussian_filter(img, kernel_size, sigma):
    """Returns the image after Gaussian filter.
    Args:
        img: the input image to be Gaussian filtered. (H, W, C)
        kernel_size: the kernel size in both the X and Y directions.
        sigma: the standard deviation in both the X and Y directions.
    Returns:
        res_img: the output image after Gaussian filter.
    """
    # Calculate the center coordinate of the kernel
    center = kernel_size // 2
    
    # Initialize a zero matrix for the kernel
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    
    # Compute the Gaussian weights for each position in the kernel
    for x in range(kernel_size):
        for y in range(kernel_size):
            dx = x - center
            dy = y - center
            # Apply the 2D Gaussian function
            kernel[x, y] = np.exp(-(dx**2 + dy**2) / (2 * sigma**2))
            
    # Normalize the kernel so the sum of all weights equals 1
    kernel = kernel / np.sum(kernel)
    
    # Expand the kernel dimensions to match the image channels for broadcasting
    kernel = np.expand_dims(kernel, axis=-1)

    # Pad the spatial dimensions of the image with zeros
    pad_width = kernel_size // 2
    padded_img = np.pad(img, ((pad_width, pad_width), (pad_width, pad_width), (0, 0)), mode='constant', constant_values=0)
    
    # Initialize the resulting image array
    res_img = np.zeros_like(img, dtype=np.float32)
    height, width, _ = img.shape
    
    # Perform pixel-wise convolution using a sliding window
    for i in range(height):
        for j in range(width):
            # Extract the local image window
            window = padded_img[i:i+kernel_size, j:j+kernel_size, :]
            # Multiply the window by the kernel and sum across the spatial dimensions
            res_img[i, j, :] = np.sum(window * kernel, axis=(0, 1))
            
    # Clip the values to the valid uint8 range and cast the array
    res_img = np.clip(res_img, 0, 255).astype(np.uint8)

    return res_img

if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "Lena-RGB.jpg"))
    kernel_size = 5
    sigma = 1
    res_img = gaussian_filter(img, kernel_size, sigma)

    cv2.imwrite(osp.join(root_dir, "gaussian_result.jpg"), res_img)