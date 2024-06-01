import cv2
import numpy as np

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