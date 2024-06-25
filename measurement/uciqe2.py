'''
UCIQE
Metrics for unferwater image quality evaluation.

Source: https://github.com/TongJiayan/UCIQE-python/blob/main/UCIQE.py
Author: TongJiayan
Based on the paper: https://doi.org/10.1109/TIP.2015.2491020

'''

import numpy as np
from skimage import io, color

def getUCIQE(img):
    img_LAB = color.rgb2lab(img)
    img_LAB = np.array(img_LAB,dtype=np.float64)
    
    # Trained coefficients are c1=0.4680, c2=0.2745, c3=0.2576 according to paper.
    # coe_Metric = [0.4680, 0.2745, 0.2576]
    c1,c2,c3 = 0.4680, 0.2745, 0.2576
    
    img_lum = img_LAB[:,:,0]/255.0
    img_a = img_LAB[:,:,1]/255.0
    img_b = img_LAB[:,:,2]/255.0

    # item-1 : contrast of chroma
    chroma = np.sqrt(np.square(img_a)+np.square(img_b))
    sigma_c = np.std(chroma)

    # item-2 : contrast of luminance
    img_lum = img_lum.flatten()
    sorted_index = np.argsort(img_lum)
    top_index = sorted_index[int(len(img_lum)*0.99)]
    bottom_index = sorted_index[int(len(img_lum)*0.01)]
    con_lum = img_lum[top_index] - img_lum[bottom_index]

    # item-3 : average saturation
    chroma = chroma.flatten()
    sat = np.divide(chroma, img_lum, out=np.zeros_like(chroma, dtype=np.float64), where=img_lum!=0)
    avg_sat = np.mean(sat)

    # uciqe = sigma_c*coe_Metric[0] + con_lum*coe_Metric[1] + avg_sat*coe_Metric[2]
    uciqe = sigma_c*c1 + con_lum*c2 + avg_sat*c3

    return uciqe

if __name__ == '__main__':
    image_path = 'dataset/image-6.jpg'
    img = io.imread(image_path)
    uciqe = getUCIQE(img)
    print("UCIQE = ",uciqe)