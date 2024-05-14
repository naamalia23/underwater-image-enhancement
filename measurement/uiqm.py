import numpy as np
import cv2

def calculate_contrast(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    contrast = np.std(gray)
    return contrast

def calculate_brightness(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness

def calculate_saturation(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    _, saturation, _ = cv2.split(hsv)
    saturation = np.mean(saturation)
    return saturation

def calculate_entropy(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    hist /= hist.sum()
    entropy = -np.sum(hist*np.log2(hist + 1e-10))
    return entropy

def UIQM(im):
    contrast = calculate_contrast(im)
    brightness = calculate_brightness(im)
    saturation = calculate_saturation(im)
    entropy = calculate_entropy(im)

    alpha = 0.1140
    beta = 0.5870
    gamma = 0.2989

    quality = (alpha * contrast) + (beta * saturation) + (gamma * brightness) - entropy

    return quality

# # Example usage
# image_path = 'path_to_your_image.jpg'
# image = cv2.imread(image_path)

# uiqm_score = UIQM(image)
# print("UIQM Score:", uiqm_score)
