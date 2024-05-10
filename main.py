import cv2
import numpy as np
import streamlit as st
from PIL import Image

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

# Function to perform Unsharp Masking
def unsharp_masking(image, radius=3, amount=0):
    """
    Return a sharpened version of the image using Unsharp Masking (UM).
    
    Parameters:
        - image: Input image (numpy array).
        - radius: Radius of the Gaussian kernel for blurring (int).
        - amount: Amount of contrast added to the edges (float).
    
    Returns:
        - Sharpened image (numpy array).
    """
    # Apply Gaussian blur to the input image
    blurred = cv2.GaussianBlur(image, (radius, radius), 50)
    
    # Calculate the unsharp mask
    unsharp_mask = cv2.subtract(image, blurred)
    
    # Apply the unsharp mask to enhance the original image
    enhanced_image = cv2.addWeighted(image, 1 +amount, unsharp_mask, -amount, 0)
    
    
    return enhanced_image.astype(np.uint8)

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


# Main function to run the Streamlit app
def main():
    st.title("Image Enhancement with CLAHE and Fusion Method")

    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image as numpy array
        image = np.array(Image.open(uploaded_file))
        
        # Show the "Proceed" button only if an image has been uploaded
        if st.button("Proceed"):
            # Apply CLAHE
            clahe_img = apply_clahe(image)
            
            # Apply Unsharp Masking
            unsharp_mask_img = unsharp_masking(image)

            # Apply image fusion
            fused_image = image_fusion(image)

            # Calculate LOE for each method
            loe_clahe = calculate_LOE(image, clahe_img)
            loe_unsharp_mask = calculate_LOE(image, unsharp_mask_img)
            loe_fused_image = calculate_LOE(image, fused_image)

            # Display the images and LOE values
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(clahe_img, caption=f"CLAHE Enhanced Image\nLOE: {loe_clahe:.2f}", use_column_width=True)
            with col3:
                st.image(unsharp_mask_img, caption=f"Unsharp Masking Enhanced Image\nLOE: {loe_unsharp_mask:.2f}", use_column_width=True)
            with col4:
                st.image(fused_image, caption=f"Fusion Method Enhanced Image\nLOE: {loe_fused_image:.2f}", use_column_width=True)
    else:
        st.warning("Please upload an image first.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
