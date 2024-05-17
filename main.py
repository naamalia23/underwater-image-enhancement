import cv2
import numpy as np
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu



from methods.clahe import apply_clahe
from methods.unsharp_masking import unsharp_masking
from methods.hef import hef_filtering
from methods.wavelet import wavelet_based_fusion

from measurement.loe import calculate_LOE
from measurement.uiqm import UIQM
from measurement.uciqe import UCIQE
from measurement.uiqm_uciqe import nmetrics


# with st.sidebar:
#     selected = option_menu("Main Menu", ["Home", 'Settings'], 
#         icons=['house', 'gear'], menu_icon="cast", default_index=1)
#     selected

# Load custom CSS
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load custom CSS
load_css()


# Function to perform image fusion
def image_fusion(image, alpha=0.5):
    """
    Perform image fusion using a linear combination of CLAHE and Unsharp Masking.
    
    Parameters:
        - image: Input image (numpy array).
        - alpha: Weighting factor for blending (float).
    
    Returns:
        - Fused image (numpy array).
    """
    # Convert the input image to BGR format
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Apply CLAHE to the input image
    image_clahe = apply_clahe(image_bgr)
    
    # Apply Unsharp Masking to the input image
    image_usm = unsharp_masking(image_bgr)
    
    # Convert the enhanced images back to RGB format
    image_clahe_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
    image_usm_rgb = cv2.cvtColor(image_usm, cv2.COLOR_BGR2RGB)
    
    
    # Perform weighted blending to fuse the images
    fused_image = cv2.addWeighted(image_clahe_rgb, alpha, image_usm_rgb, 1 - alpha, 0)
    
    return fused_image.astype(np.uint8)


# Main function to run the Streamlit app
def main():
    st.title("Underwater Image Enhancement Methods Comparison")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image as numpy array
        image = np.array(Image.open(uploaded_file))
        
        # Show the "Proceed" button only if an image has been uploaded
        if st.button("Proceed"):
            # Apply enhancement methods
            clahe_img = apply_clahe(image)
            unsharp_mask_img = unsharp_masking(image)
            fused_img = image_fusion(image)
            hef_img = hef_filtering(image)
            wavelet_img = wavelet_based_fusion(image)

            # Calculate LOE for each method
            loe_original = calculate_LOE(image, image)
            loe_clahe = calculate_LOE(image, clahe_img)
            loe_unsharp_mask = calculate_LOE(image, unsharp_mask_img)
            loe_fused_image = calculate_LOE(image, fused_img)
            loe_hef_image = calculate_LOE(image, hef_img)
            loe_wavelet = calculate_LOE(image, wavelet_img)
            
            # Calculate UIQM for each method
            uiqm_original = UIQM(image)
            uiqm_clahe = UIQM(clahe_img)
            uiqm_unsharp_mask = UIQM(unsharp_mask_img)
            uiqm_fused = UIQM(fused_img)
            uiqm_hef = UIQM(hef_img)
            uiqm_wavelet = UIQM(wavelet_img)
            
            # Calculate UCIQE for each method
            uciqe_original = UCIQE(image)
            uciqe_clahe = UCIQE(clahe_img)
            uciqe_unsharp_mask = UCIQE(unsharp_mask_img)
            uciqe_fused = UCIQE(fused_img)
            uciqe_hef = UCIQE(hef_img)
            uciqe_wavelet = UCIQE(wavelet_img)

            # Calculate nmetrics for each method
            nmetrics_clahe = nmetrics(clahe_img) 
            nmetrics_unsharp_mask = nmetrics(unsharp_mask_img)
            nmetrics_fused = nmetrics(fused_img)
            nmetrics_hef = nmetrics(hef_img)


            col1, col2, col3, col4, col5, col6 = st.columns(6)
            with col1:
                st.image(image, use_column_width=True)
                st.markdown(f"**Original Image**\n\nLOE: {loe_original:.2f}\n\nUIQM: {uiqm_original:.2f}\n\nUCIQE: {uciqe_original:.2f}\n\n")
            with col2:
                st.image(clahe_img, use_column_width=True)
                st.markdown(f"**CLAHE Enhanced Image**\n\nLOE: {loe_clahe:.2f}\n\nUIQM: {uiqm_clahe:.2f}\n\nUCIQE: {uciqe_clahe:.2f}\n\n")
            with col3:
                st.image(unsharp_mask_img, use_column_width=True)
                st.markdown(f"**Unsharp Masking Enhanced Image**\n\nLOE: {loe_unsharp_mask:.2f}\n\nUIQM: {uiqm_unsharp_mask:.2f}\n\nUCIQE: {uciqe_unsharp_mask:.2f}\n")
            with col4:
                st.image(fused_img, use_column_width=True)
                st.markdown(f"**Fusion Method Enhanced Image**\n\nLOE: {loe_fused_image:.2f}\n\nUIQM: {uiqm_fused:.2f}\n\nUCIQE: {uciqe_fused:.2f}\n")
            with col5:
                st.image(hef_img, use_column_width=True)
                st.markdown(f"**HEF Method Enhanced Image**\n\nLOE: {loe_hef_image:.2f}\n\nUIQM: {uiqm_hef:.2f}\n\nUCIQE: {uciqe_hef:.2f}\n")
            with col6:
                st.image(wavelet_img, use_column_width=True)
                st.markdown(f"**Wavelet Method Enhanced Image**\n\nLOE: {loe_wavelet:.2f}\n\nUIQM: {uiqm_wavelet:.2f}\n\nUCIQE: {uciqe_wavelet:.2f}\n")


    else:
        st.warning("Please upload an image first.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
