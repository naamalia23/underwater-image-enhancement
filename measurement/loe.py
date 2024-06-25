import cv2
import numpy as np

#https://stackoverflow.com/questions/37596001/lightness-order-error-loe-for-images-quality-assessement-matlab
#bisa cari referensi lain untuk IQA. pak khahlil pake loe di papaer xray enhancement
def calculate_LOE_2(original_image, enhanced_image): #
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


def calculate_LOE(original_image, enhanced_image):
    # Ensure both images are in the same format (grayscale)
    if len(original_image.shape) == 3 and original_image.shape[2] == 3:
        original_image_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    else:
        original_image_gray = original_image

    if len(enhanced_image.shape) == 3 and enhanced_image.shape[2] == 3:
        enhanced_image_gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
    else:
        enhanced_image_gray = enhanced_image
    
    # Get the dimensions of the original image
    N, M = original_image_gray.shape

    # Calculate the lightness of the original and enhanced images
    L = original_image_gray
    Le = enhanced_image_gray

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