import os.path as osp
import cv2
import numpy as np

def histogram_equalization(img):
    """Returns the image after histogram equalization.
    Args:
        img: the input image to be executed for histogram equalization.
    Returns:
        res_img: the output image after histogram equalization.
    """
    # Compute the histogram of the image
    hist, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])

    # Compute the Cumulative Distribution Function (CDF)
    cdf = hist.cumsum()

    # Mask zero values in the CDF to prevent issues during normalization
    cdf_masked = np.ma.masked_equal(cdf, 0)

    # Normalize the CDF to map pixel intensities to the range [0, 255]
    cdf_normalized = (cdf_masked - cdf_masked.min()) * 255 / (cdf_masked.max() - cdf_masked.min())

    # Fill the masked values back with 0 and convert the lookup table to uint8
    cdf_final = np.ma.filled(cdf_normalized, 0).astype(np.uint8)

    # Apply the mapping to the original image
    res_img = cdf_final[img]

    return res_img

def local_histogram_equalization(img):
    """Returns the image after local histogram equalization.
    Args:
        img: the input image to be executed for local histogram equalization.
    Returns:
        res_img: the output image after local histogram equalization.
    """
    # Define the local square size. 
    # 33x33 is a reasonable size to capture local contrast features naturally.
    window_size = 33
    pad_size = window_size // 2

    # Pad the image using reflection to handle boundary pixels smoothly
    padded_img = np.pad(img, pad_size, mode='reflect')

    # Initialize the output image array
    res_img = np.zeros_like(img, dtype=np.uint8)
    height, width = img.shape
    window_area = window_size * window_size

    # Iterate over each pixel in the spatial dimensions
    for i in range(height):
        for j in range(width):
            # Extract the local square window around the current pixel
            window = padded_img[i:i+window_size, j:j+window_size]

            # Get the intensity of the center pixel
            center_intensity = img[i, j]

            # Compute the CDF value specifically for the center pixel's intensity
            # This is equivalent to counting how many pixels in the window are <= center_intensity
            count = np.sum(window <= center_intensity)

            # Map the local rank proportion to the [0, 255] range
            new_intensity = int((count / window_area) * 255)

            # Assign the equalized intensity
            res_img[i, j] = new_intensity

    return res_img

if __name__ == "__main__":
    root_dir = osp.dirname(osp.abspath(__file__))
    img = cv2.imread(osp.join(root_dir, "moon.png"), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res_hist_equalization = histogram_equalization(img)
    res_local_hist_equalization = local_histogram_equalization(img)

    cv2.imwrite(osp.join(root_dir, "HistEqualization.jpg"), res_hist_equalization)
    cv2.imwrite(
        osp.join(root_dir, "LocalHistEqualization.jpg"), res_local_hist_equalization
    )