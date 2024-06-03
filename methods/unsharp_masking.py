import cv2
import numpy as np
from methods.clahe import apply_clahe

# Function to perform Unsharp Masking
def unsharp_masking_2(image, radius=3, amount=1):
    """
    Return a sharpened version of the image using Unsharp Masking (UM).
    
    Parameters:
        - image: Input image (numpy array).
        - radius: Radius of the Gaussian kernel for blurring (int).
        - amount: Amount of contrast added to the edges (float).
    
    Returns:
        - Sharpened image (numpy array).
    """
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray_image, (radius, radius), 3)
    
    # Calculate the unsharp mask
    unsharp_mask = cv2.subtract(gray_image, blurred)
    
    # Apply the unsharp mask to enhance the original image
    enhanced_image = cv2.addWeighted(gray_image, 1 + amount, unsharp_mask, -amount, 0)
    
    # Convert the enhanced image back to BGR format
    enhanced_image_bgr = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    
    return enhanced_image_bgr.astype(np.uint8)

def unsharp_masking(image, radius=5, amount=2):
    # Step 1: Read the original image
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if original_image is None:
        raise ValueError("Image not found or invalid image path")

    # Step 2: Apply Gaussian Blur to the original image
    blurred_image = cv2.GaussianBlur(original_image, (0, 0), radius)

    # Step 3: Create the unsharp mask by subtracting the blurred image from the original image
    unsharp_mask = cv2.subtract(original_image, blurred_image)

    # Step 4: Combine the original image with the unsharp mask to enhance the image
    enhanced_image = cv2.addWeighted(original_image, 1, unsharp_mask, amount, 0)

    # Convert the enhanced image back to BGR format
    # enhanced_image_bgr= cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2BGR)
    enhanced_image_rgb= cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)

    # Step 5:  result
    return enhanced_image_rgb.astype(np.uint8)

def unsharp_masking_3(image, sigma=1.0, strength=1.5):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
    
    # Create an unsharp mask
    sharpened = cv2.addWeighted(gray, 1 + strength, blurred, -strength, 0)
    
    # Merge back with the original image
    usm_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    return usm_img

# Function to perform image fusion
def fusion_clahe_um(image, alpha=0.5):
    """
    Perform image fusion using a linear combination of CLAHE and Unsharp Masking.
    
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
    
    # Apply Unsharp Masking to the input image
    image_usm = unsharp_masking(image_bgr)
    
    # Convert the enhanced images back to RGB format
    image_clahe_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
    image_usm_rgb = cv2.cvtColor(image_usm, cv2.COLOR_BGR2RGB)
    
    
    # Perform weighted blending to fuse the images
    fused_image = cv2.addWeighted(image_clahe_rgb, alpha, image_usm_rgb, 1 - alpha, 0)
    
    return fused_image.astype(np.uint8)