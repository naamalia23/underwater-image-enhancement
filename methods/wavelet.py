import cv2
import numpy as np
import pywt

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def enhance_image(image, gamma_rgb=2.0, gamma_hsv=2.0):
    # Apply gamma correction in RGB space
    enhanced_rgb = gamma_correction(image, gamma_rgb)
    
    # Convert to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Apply gamma correction to the V (Value) channel
    h, s, v = cv2.split(hsv_image)
    v_enhanced = gamma_correction(v, gamma_hsv)
    
    # Merge the channels back and convert to RGB
    enhanced_hsv = cv2.merge([h, s, v_enhanced])
    enhanced_hsv = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    
    return enhanced_rgb, enhanced_hsv

def wavelet_decomposition(image, level=2):
    coeffs = pywt.wavedec2(image, 'db1', level=level)
    return coeffs

def wavelet_fusion(coeffs1, coeffs2):
    fused_coeffs = []
    for c1, c2 in zip(coeffs1, coeffs2):
        if isinstance(c1, tuple):
            fused_coeffs.append(tuple((c1[i] + c2[i]) / 2 for i in range(len(c1))))
        else:
            fused_coeffs.append((c1 + c2) / 2)
    return fused_coeffs

def wavelet_reconstruction(coeffs):
    image_reconstructed = pywt.waverec2(coeffs, 'db1')
    return np.clip(image_reconstructed, 0, 255).astype('uint8')

def wavelet_based_fusion(image, gamma_rgb=2.0, gamma_hsv=2.0, wavelet_level=2):
    # Enhance the image in RGB and HSV spaces
    enhanced_rgb, enhanced_hsv = enhance_image(image, gamma_rgb, gamma_hsv)
    
    # Wavelet decomposition for each channel separately
    channels_rgb = cv2.split(enhanced_rgb)
    channels_hsv = cv2.split(enhanced_hsv)
    
    fused_channels = []
    for ch_rgb, ch_hsv in zip(channels_rgb, channels_hsv):
        # Decompose each channel
        coeffs_rgb = wavelet_decomposition(ch_rgb, level=wavelet_level)
        coeffs_hsv = wavelet_decomposition(ch_hsv, level=wavelet_level)
        
        # Fuse the decomposed channels
        fused_coeffs = wavelet_fusion(coeffs_rgb, coeffs_hsv)
        
        # Reconstruct the fused channel
        fused_channel = wavelet_reconstruction(fused_coeffs)
        fused_channels.append(fused_channel)
    
    # Merge the fused channels back into an RGB image
    fused_image = cv2.merge(fused_channels)
    
    return fused_image