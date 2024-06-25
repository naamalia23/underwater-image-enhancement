import cv2
import numpy as np

def apply_clahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(channel)


def percentile_stretch(channel):
    prct_min, prct_max = np.percentile(channel, (1, 99))
    stretched = np.clip((channel - prct_min) * 255.0 / (prct_max - prct_min), 0, 255)
    return stretched.astype(np.uint8)

#===================
#this blending only uses clahe and percentile dr paper garg2018

def blending_clahe_percentile(image):
    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the image into its respective channels
    h, s, v = cv2.split(hsv_image)
    
    # Apply CLAHE to H, S, and V channels
    h_clahe = apply_clahe(h)
    s_clahe = apply_clahe(s)
    v_clahe = apply_clahe(v)
    
    # Merge the CLAHE enhanced channels back together
    hsv_clahe = cv2.merge((h_clahe, s_clahe, v_clahe))
    
    # Convert the HSV image back to RGB
    rgb_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    # Split the image into R, G, and B channels
    r, g, b = cv2.split(image)
    
    # Apply percentile stretching to R, G, and B channels
    r_stretch = percentile_stretch(r)
    g_stretch = percentile_stretch(g)
    b_stretch = percentile_stretch(b)
    
    # Merge the stretched channels back together
    rgb_stretch = cv2.merge((r_stretch, g_stretch, b_stretch))
    
    # Combine the CLAHE and percentile stretched images
    blended_image = cv2.addWeighted(rgb_clahe, 0.5, rgb_stretch, 0.5, 0)
    
    return blended_image


#====================================
#this blending uses clahe, percentile, gamma correction and sharpen image purposed

def gamma_correction(image, gamma=1.0):
    """
    Apply gamma correction to the image.
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def sharpen_image(image):
    """
    Apply sharpening to the image.
    """
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def blending_sharpen_clahe_percentile(image):
    """
    Enhance the underwater image using CLAHE, percentile stretching, gamma correction, and sharpening.
    """
    if image is None:
        raise ValueError("Image not loaded properly.")
    
    # Ensure the image is in RGB format
    if len(image.shape) == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif image.shape[2] != 3:
        raise ValueError("Unexpected number of channels in the image.")

    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Split the image into its respective channels
    h, s, v = cv2.split(hsv_image)
    
    # Apply CLAHE to H, S, and V channels
    h_clahe = apply_clahe(h)
    s_clahe = apply_clahe(s)
    v_clahe = apply_clahe(v)
    
    # Merge the CLAHE enhanced channels back together
    hsv_clahe = cv2.merge((h_clahe, s_clahe, v_clahe))
    
    # Convert the HSV image back to RGB
    rgb_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    # Split the original image into R, G, and B channels
    r, g, b = cv2.split(image)
    
    # Apply percentile stretching to R, G, and B channels
    r_stretch = percentile_stretch(r)
    g_stretch = percentile_stretch(g)
    b_stretch = percentile_stretch(b)
    
    # Merge the stretched channels back together
    rgb_stretch = cv2.merge((r_stretch, g_stretch, b_stretch))
    
    # Apply gamma correction to both CLAHE and percentile stretched images
    rgb_clahe_gamma = gamma_correction(rgb_clahe, gamma=1.2)
    rgb_stretch_gamma = gamma_correction(rgb_stretch, gamma=1.2)
    
    # Blend the CLAHE and percentile stretched images with adaptive weights
    weight_clahe = np.mean(cv2.cvtColor(rgb_clahe_gamma, cv2.COLOR_BGR2GRAY)) / 255.0
    weight_stretch = 1.0 - weight_clahe
    blended_image = cv2.addWeighted(rgb_clahe_gamma, weight_clahe, rgb_stretch_gamma, weight_stretch, 0)
    
    # Apply sharpening to the blended image
    final_image = sharpen_image(blended_image)
    
    return final_image