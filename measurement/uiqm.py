'''
UIQM
Metrics for unferwater image quality evaluation.

reference to the following below: 
Code: https://github.com/bilityniu/underimage-fusion-enhancement/blob/master/UIQM.m
Paper: 
    https://doi.org/10.1109/TIP.2015.2491020 (UIQM)
    https://doi.org/10.1109/TIP.2017.2759252 (Image Enhancement)

'''

import numpy as np
import cv2
from scipy.ndimage import convolve

def getUIQM(image):
    c1, c2, c3 = 0.0282, 0.2953, 3.5753
    uicm = UICM(image)
    uism = UISM(image)
    uiconm = UIConM(image)
    uiqm = c1 * uicm + c2 * uism + c3 * uiconm
    return uiqm

def UICM(image):
    R = image[:, :, 0].astype(float)
    G = image[:, :, 1].astype(float)
    B = image[:, :, 2].astype(float)
    RG = R - G
    YB = (R + G) / 2 - B

    K = R.size

    # For R-G channel
    RG1 = np.sort(RG.ravel())
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[int(alphaL * K) : int(K * (1 - alphaR))]
    N = K * (1 - alphaL - alphaR)
    meanRG = np.mean(RG1)
    deltaRG = np.sqrt(np.mean((RG1 - meanRG) ** 2))

    # For Y-B channel
    YB1 = np.sort(YB.ravel())
    YB1 = YB1[int(alphaL * K) : int(K * (1 - alphaR))]
    meanYB = np.mean(YB1)
    deltaYB = np.sqrt(np.mean((YB1 - meanYB) ** 2))

    # UICM
    uicm = -0.0268 * np.sqrt(meanRG ** 2 + meanYB ** 2) + 0.1586 * np.sqrt(deltaRG ** 2 + deltaYB ** 2)
    
    return uicm

def UISM(image):
    Ir = image[:, :, 0].astype(float)
    Ig = image[:, :, 1].astype(float)
    Ib = image[:, :, 2].astype(float)

    hx = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    hy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

    SobelR = np.abs(convolve(Ir, hx) + convolve(Ir, hy))
    SobelG = np.abs(convolve(Ig, hx) + convolve(Ig, hy))
    SobelB = np.abs(convolve(Ib, hx) + convolve(Ib, hy))

    patchsz = 5
    m, n = Ir.shape
    SobelR = cv2.resize(SobelR, (n - n % patchsz + patchsz, m - m % patchsz + patchsz))
    SobelG = cv2.resize(SobelG, (n - n % patchsz + patchsz, m - m % patchsz + patchsz))
    SobelB = cv2.resize(SobelB, (n - n % patchsz + patchsz, m - m % patchsz + patchsz))

    m, n = SobelR.shape
    k1 = m // patchsz
    k2 = n // patchsz

    def calc_eme(SobelC):
        EME = 0
        for i in range(0, m, patchsz):
            for j in range(0, n, patchsz):
                im = SobelC[i:i + patchsz, j:j + patchsz]
                if np.max(im) != 0 and np.min(im) != 0:
                    EME += np.log(np.max(im) / np.min(im))
        return 2 / (k1 * k2) * np.abs(EME)

    EMER = calc_eme(SobelR)
    EMEG = calc_eme(SobelG)
    EMEB = calc_eme(SobelB)

    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB

    return uism

def UIConM(image):
    R = image[:, :, 0].astype(float)
    G = image[:, :, 1].astype(float)
    B = image[:, :, 2].astype(float)

    patchsz = 5
    m, n = R.shape
    R = cv2.resize(R, (n - n % patchsz + patchsz, m - m % patchsz + patchsz))
    G = cv2.resize(G, (n - n % patchsz + patchsz, m - m % patchsz + patchsz))
    B = cv2.resize(B, (n - n % patchsz + patchsz, m - m % patchsz + patchsz))

    m, n = R.shape
    k1 = m // patchsz
    k2 = n // patchsz

    def calc_amee(C):
        AMEE = 0
        for i in range(0, m, patchsz):
            for j in range(0, n, patchsz):
                im = C[i:i + patchsz, j:j + patchsz]
                Max = np.max(im)
                Min = np.min(im)
                if (Max != 0 or Min != 0) and Max != Min:
                    AMEE += np.log((Max - Min) / (Max + Min)) * ((Max - Min) / (Max + Min))
        return 1 / (k1 * k2) * np.abs(AMEE)

    AMEER = calc_amee(R)
    AMEEG = calc_amee(G)
    AMEEB = calc_amee(B)

    uiconm = AMEER + AMEEG + AMEEB
    return uiconm


# Example usage
if __name__ == "__main__":
    # Load an example image (replace with your own image path)
    I = cv2.imread('dataset/image-6.jpg')
    I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)

    # Calculate UCIQE
    uciqe_value = getUIQM(I)
    print("UIQM Value:", uciqe_value)