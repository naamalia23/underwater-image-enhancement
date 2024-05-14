import cv2
import numpy as np
from measurement.uiqm import UIQM
from measurement.uciqe import UCIQE
from measurement.uiqm_uciqe import nmetrics

# ===========================
# Enhancement Method
# * CLAHE
# * ...
# ===========================

# Function to perform CLAHE (Contrast Limited Adaptive Histogram Equalization)
def apply_clahe(image):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    
    # Merge the CLAHE-enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((l_clahe, a, b))
    
    # Convert LAB image back to RGB
    enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    
    return enhanced_image


# ===========================
# code that runs on this file
# * load image > calculate score
# * enchance the image > calculate score
# * enhance methode :
#   - CLAHE
# * show image: original and enhanced
# ===========================
# original image
image = cv2.imread("dataset/image-1.jpg", cv2.IMREAD_COLOR)
# measurement
uiqm_score = UIQM(image)
uciqe_score = UCIQE(image)
nuiqm,nuciqe = nmetrics(image)
print("\nOriginal Score")
score_text = "UIQM: "+str(uiqm_score)+"\n"
score_text += "UCIQE: "+str(uciqe_score)+"\n"
score_text += "nUIQM: "+str(nuiqm)+" - nUCIQE: "+str(nuciqe)
print(score_text)

# Apply CLAHE
clahe_img = apply_clahe(image)
# measurement
uiqm_score = UIQM(clahe_img)
uciqe_score = UCIQE(clahe_img)
nuiqm,nuciqe = nmetrics(clahe_img)
print("-------\nCLAHE Score")
score_text = "UIQM: "+str(uiqm_score)+"\n"
score_text += "UCIQE: "+str(uciqe_score)+"\n"
score_text += "nUIQM: "+str(nuiqm)+" - nUCIQE: "+str(nuciqe)
print(score_text)

# show image
cv2.imshow("Original", image)
cv2.imshow("Clahe", clahe_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
