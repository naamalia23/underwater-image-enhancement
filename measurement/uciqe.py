'''
UCIQE
Metrics for unferwater image quality evaluation.

reference to the following below: 
Code: https://github.com/bilityniu/underimage-fusion-enhancement/blob/master/UCIQE.m
Paper: 
    https://doi.org/10.1109/TIP.2015.2491020 (UCIQE)
    https://doi.org/10.1109/TIP.2017.2759252 (Image Enhancement)

'''

import numpy as np
import cv2
from skimage import color, exposure

def getUCIQE(I, Coe_Metric=None):
    # Set default coefficients if none are provided
    if Coe_Metric is None:
        Coe_Metric = [0.4680, 0.2745, 0.2576]
    
    # Transform to Lab color space
    Img_lab = cv2.cvtColor(I, cv2.COLOR_RGB2LAB)
    
    Img_lum = Img_lab[:, :, 0].astype(np.float64)
    Img_lum = Img_lum / 255 + np.finfo(float).eps

    Img_a = Img_lab[:, :, 1].astype(np.float64) / 255
    Img_b = Img_lab[:, :, 2].astype(np.float64) / 255

    # Chroma
    Img_Chr = np.sqrt(Img_a.flatten()**2 + Img_b.flatten()**2)
    
    # Saturation
    Img_Sat = Img_Chr / np.sqrt(Img_Chr**2 + Img_lum.flatten()**2)

    # Average of saturation
    Aver_Sat = np.mean(Img_Sat)
    
    # Average of Chroma
    Aver_Chr = np.mean(Img_Chr)
    
    # Variance of Chroma
    Var_Chr = np.sqrt(np.mean((np.abs(1 - (Aver_Chr / Img_Chr))**2)))
    
    # Contrast of luminance
    Tol = exposure.rescale_intensity(Img_lum)
    Con_lum = Tol.max() - Tol.min()
    
    # Get final quality value
    Qualty_Val = Coe_Metric[0] * Var_Chr + Coe_Metric[1] * Con_lum + Coe_Metric[2] * Aver_Sat
    
    return Qualty_Val

# Example usage
if __name__ == "__main__":
    # Load an example image (replace with your own image path)
    I = cv2.imread('dataset/image-6.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    # Calculate UCIQE
    uciqe_value = getUCIQE(I)
    print("UCIQE Value:", uciqe_value)
