import cv2
import numpy as np
from scipy.signal import convolve2d
from scipy.fftpack import fft2, ifft2, fftshift
from scipy.ndimage import gaussian_filter
from methods.clahe import apply_clahe


def hef_filtering(image, radius=1, cutoff_distance=10):
    # Separate the image into channels
    if image.ndim == 3 and image.shape[2] == 3:
        channels = []
        for channel in range(image.shape[2]):
            channels.append(hef_filtering(image[:, :, channel], radius, cutoff_distance))
        return np.stack(channels, axis=2)

    # Apply Gaussian high-pass filter
    filtered_image = image - gaussian_filter(image, radius)

    # Calculate the Fourier transform of the filtered image
    f_filtered_image = fft2(filtered_image)

    # Shift the zero-frequency component to the center of the spectrum
    f_filtered_image_shifted = fftshift(f_filtered_image)

    # Calculate the Gaussian high-pass filter in the frequency domain
    gaussian_filter_freq = np.exp(-0.5 * ((np.linspace(-image.shape[0]/2, image.shape[0]/2, image.shape[0])[:, None])**2 + (np.linspace(-image.shape[1]/2, image.shape[1]/2, image.shape[1])[None, :])**2) / (2 * cutoff_distance**2))
    gaussian_filter_freq[0, 0] = 1  # Preserve the DC component

    # Make sure the Gaussian high-pass filter has the same dimensions as the filtered image
    if image.ndim == 3:
        gaussian_filter_freq = np.repeat(gaussian_filter_freq[:, :, np.newaxis], image.shape[2], axis=2)

    # Apply the Gaussian high-pass filter in the frequency domain
    f_filtered_image_shifted *= gaussian_filter_freq

    # Shift the zero-frequency component back to its original position
    f_filtered_image_shifted = fftshift(f_filtered_image_shifted)

    # Calculate the inverse Fourier transform to get the filtered image
    filtered_image = np.real(ifft2(f_filtered_image_shifted))

    # Adjust the contrast using histogram equalization
    hef_sharpened = filtered_image + image
    hef_sharpened = (hef_sharpened - np.min(hef_sharpened)) / (np.max(hef_sharpened) - np.min(hef_sharpened))
    hef_sharpened *= 255

    return hef_sharpened.astype(np.uint8)
    
# Function to perform image fusion
def fusion_clahe_hef(image, alpha=0.5):
    """
    Perform image fusion using a linear combination of CLAHE and HEF.
    
    Parameters:
        - image: Input image (numpy array).
        - alpha: Weighting factor for blending (float).
    
    Returns:
        - Fused image (numpy array).
    """
    # Convert the input image to BGR format
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE to the input image
    image_clahe = apply_clahe(image_bgr)
    
    # Apply hef to the input image
    image_hef = hef_filtering(image_bgr)
    
    # Convert the enhanced images back to RGB format
    image_clahe_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
    image_hef_rgb = cv2.cvtColor(image_hef, cv2.COLOR_BGR2RGB)
    
    
    # Perform weighted blending to fuse the images
    fused_image = cv2.addWeighted(image_clahe_rgb, alpha, image_hef_rgb, 1 - alpha, 0)
    
    return fused_image.astype(np.uint8)