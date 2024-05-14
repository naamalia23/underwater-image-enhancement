import numpy as np
import cv2

def calculate_contrast(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast

def calculate_saturation(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    saturation = np.mean(saturation)
    return saturation

def calculate_colorfulness(im):
    (B, G, R) = cv2.split(im.astype("float"))
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    rb_mean = np.mean(rg)
    yb_mean = np.mean(yb)
    colorfulness = np.sqrt((rg.std() ** 2) + (yb.std() ** 2)) + 0.3 * (rb_mean + yb_mean)
    return colorfulness

def calculate_naturalness(im):
    R, G, B = im[:,:,0], im[:,:,1], im[:,:,2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    naturalness = 1 - np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2) / np.sqrt(np.std(rg) ** 2 + np.std(yb) ** 2)
    return naturalness

def UCIQE(im):
    contrast = calculate_contrast(im)
    saturation = calculate_saturation(im)
    colorfulness = calculate_colorfulness(im)
    naturalness = calculate_naturalness(im)

    alpha = 0.0282
    beta = 0.2953
    gamma = 3.5753
    delta = 1.5068

    uciqe_score = np.exp(alpha * contrast) * np.exp(-beta * saturation) * np.exp(gamma * colorfulness) * np.exp(delta * naturalness)

    return uciqe_score

# # Example usage
# image_path = 'path_to_your_image.jpg'
# image = cv2.imread(image_path)

# uciqe_score = UCIQE(image)
# print("UCIQE Score:", uciqe_score)
