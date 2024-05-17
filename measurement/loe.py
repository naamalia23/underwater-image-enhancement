import cv2
import numpy as np

#https://stackoverflow.com/questions/37596001/lightness-order-error-loe-for-images-quality-assessement-matlab
#bisa cari referensi lain untuk IQA. pak khahlil pake loe di papaer xray enhancement
def calculate_LOE(original_image, enhanced_image): #
    # Get the dimensions of the original image
    N, M, _ = original_image.shape

    # Calculate the lightness of the original and enhanced images
    L = np.max(original_image, axis=2)
    Le = np.max(enhanced_image, axis=2)

    # Calculate the downsampling factor
    r = 50 / min(M, N)

    # Perform downsampling
    Md = round(M * r)
    Nd = round(N * r)
    Ld = cv2.resize(L, (Md, Nd))
    Led = cv2.resize(Le, (Md, Nd))

    # Initialize RD matrix
    RD = np.zeros((Nd, Md))

    # Calculate RD values
    for y in range(Md):
        for x in range(Nd):
            E = np.bitwise_xor(Ld[x, y] >= Ld, Led[x, y] >= Led)
            RD[x, y] = np.sum(E)

    # Calculate LOE
    LOE = np.sum(RD) / (Md * Nd)
    return LOE
