import cv2

# Function to perform CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE-enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert LAB image back to RGB
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return enhanced_image