import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_clahe(channel):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(channel)

def blending_clahe_2(image):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Split the image into R, G, B components
    r, g, b = cv2.split(image_rgb)

    # Apply CLAHE to R, G, B components
    r_clahe = apply_clahe(r)
    g_clahe = apply_clahe(g)
    b_clahe = apply_clahe(b)

    # Merge R, G, B CLAHE components
    clahe_rgb = cv2.merge((r_clahe, g_clahe, b_clahe))

    # Convert the image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split the image into H, S, V components
    h, s, v = cv2.split(image_hsv)

    # Apply CLAHE to H, S, V components
    h_clahe = apply_clahe(h)
    s_clahe = apply_clahe(s)
    v_clahe = apply_clahe(v)

    # Merge H, S, V CLAHE components
    clahe_hsv = cv2.merge((h_clahe, s_clahe, v_clahe))

    # Convert CLAHE HSV image back to BGR
    clahe_hsv_bgr = cv2.cvtColor(clahe_hsv, cv2.COLOR_HSV2BGR)

    # Convert the RGB CLAHE image back to BGR
    clahe_rgb_bgr = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2BGR)

    # Blend the CLAHE images
    blended_image = cv2.addWeighted(clahe_rgb_bgr, 0.5, clahe_hsv_bgr, 0.5, 0)

    return blended_image 


def percentile_stretch(channel):
    prct_min, prct_max = np.percentile(channel, (1, 99))
    stretched = np.clip((channel - prct_min) * 255.0 / (prct_max - prct_min), 0, 255)
    return stretched.astype(np.uint8)

def blending_clahe(image):
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