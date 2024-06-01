import numpy as np

# I/O
import io
import zipfile
from pathlib import Path
from PIL import Image

import cv2
import streamlit as st
# from streamlit_option_menu import option_menu
import uuid

# methods and measurement
from methods.clahe import apply_clahe
from methods.unsharp_masking import unsharp_masking
from methods.hef import hef_filtering
from methods.wavelet import wavelet_based_fusion
from methods.blending_clahe import blending_clahe

from measurement.loe import calculate_LOE
from measurement.uiqm_uciqe import nmetrics


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

# Function to perform image fusion
def fusion_clahe_hef(image, alpha=0.5):
    """
    Perform image fusion using a linear combination of CLAHE and HEF.
    
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
    
    # Apply HEF Filtering to the input image
    image_hef = hef_filtering(image_bgr)
    
    # Convert the enhanced images back to RGB format
    image_clahe_rgb = cv2.cvtColor(image_clahe, cv2.COLOR_BGR2RGB)
    image_hef_rgb = cv2.cvtColor(image_hef, cv2.COLOR_BGR2RGB)
    
    
    # Perform weighted blending to fuse the images
    fused_image = cv2.addWeighted(image_clahe_rgb, alpha, image_hef_rgb, 1 - alpha, 0)
    
    return fused_image.astype(np.uint8)


# streamlit 
# file uploads setup
MAX_FILES = 5
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

def setup_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(page_title="Underwater Image Enhancement", page_icon=":star:") 
    st.title("Underwater Image Enhancement Methods Comparison")
    hide_streamlit_style()

def hide_streamlit_style():
    """Hides default Streamlit styling."""
    st.markdown(
        "<style>footer {visibility: hidden;} #MainMenu {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )

def initialize_session():
    """Initializes a unique session ID."""
    if "uploader_key" not in st.session_state:
        st.session_state["uploader_key"] = str(uuid.uuid4())

def display_ui():
    """Displays the user interface for file upload and returns uploaded files."""
    st.sidebar.markdown("### Underwater Image")

    uploaded_files = st.sidebar.file_uploader(
        "Choose images",
        type=ALLOWED_TYPES,
        accept_multiple_files=True,
        key=st.session_state.get("uploader_key", "file_uploader"),
    )

    display_footer()
    return uploaded_files

def display_footer():
    """Displays a custom footer."""
    footer = """<div style="position: fixed; bottom: 0; left: 20px;">
                <p>Developed with ❤ by <a href="https://github.com/naamalia23/underwater-image-enhancement" target="_blank">@Team</a></p>
                </div>"""
    st.sidebar.markdown(footer, unsafe_allow_html=True)

def process_and_display_images(uploaded_files):
    """Processes the uploaded files and displays the original and result images."""
    if not uploaded_files:
        st.warning("Please upload an image.")
        return

    if not st.sidebar.button("Proceed"):
        return

    if len(uploaded_files) > MAX_FILES:
        st.warning(f"Maximum {MAX_FILES} files will be processed.")
        uploaded_files = uploaded_files[:MAX_FILES]

    results = []

    with st.spinner("Enhancing Images..."):
        for uploaded_file in uploaded_files:
            # image
            image = np.array(Image.open(uploaded_file))

        # Apply enhancement methods
            clahe_img = apply_clahe(image)
            unsharp_mask_img = unsharp_masking(image)
            fused_img = image_fusion(image)
            hef_img = hef_filtering(image)
            wavelet_img = wavelet_based_fusion(image)
            clahe_hef_img = fusion_clahe_hef(image)
            blending_clahe_img = blending_clahe(image)

            # Calculate LOE for each method
            loe_original = calculate_LOE(image, image)
            loe_clahe = calculate_LOE(image, clahe_img)
            loe_unsharp_mask = calculate_LOE(image, unsharp_mask_img)
            loe_fused = calculate_LOE(image, fused_img)
            loe_hef = calculate_LOE(image, hef_img)
            loe_wavelet = calculate_LOE(image, wavelet_img)
            loe_clahe_hef = calculate_LOE(image, clahe_hef_img)
            loe_blending_clahe = calculate_LOE(image, blending_clahe_img)

            # Calculate UIQM,UCIQE using nmetrics for each method
            uiqm_original,uciqe_original = nmetrics(image) 
            uiqm_clahe,uciqe_clahe = nmetrics(clahe_img) 
            uiqm_unsharp_mask,uciqe_unsharp_mask = nmetrics(unsharp_mask_img)
            uiqm_fused,uciqe_fused = nmetrics(fused_img)
            uiqm_hef,uciqe_hef = nmetrics(hef_img)
            uiqm_wavelet,uciqe_wavelet = nmetrics(wavelet_img)  
            uiqm_clahe_hef,uciqe_clahe_hef = nmetrics(clahe_hef_img)  
            uiqm_blending_clahe, uciqe_blending_clahe = nmetrics(blending_clahe_img)
            # append image result
            results.append((image, clahe_img, unsharp_mask_img, fused_img, hef_img, wavelet_img, clahe_hef_img, blending_clahe_img, uploaded_file.name))  
            
        for image, clahe_img, unsharp_mask_img, fused_img, hef_img, wavelet_img, clahe_hef_img,blending_clahe_img, __name__ in results:
            # Display the images and measurement values
            # col1, col2, col3, col4, col5, col6 = st.columns(6)
            st.markdown(f"**{__name__}**\n\n")
            cols = st.columns(2) # number of columns in each row! = 2
            with cols[0]:
                st.image(image, use_column_width=True)
                st.markdown(f'''**Original Image**  
                    LOE: {loe_original:.2f}  
                    UIQM: {uiqm_original:.2f}  
                    UCIQE: {uciqe_original:.2f}''')
            with cols[1]:
                st.image(clahe_img, use_column_width=True)
                st.markdown(f'''**CLAHE Enhanced Image**  
                    LOE: {loe_clahe:.2f}  
                    UIQM: {uiqm_clahe:.2f}  
                    UCIQE: {uciqe_clahe:.2f}''')
            # UM
            with cols[0]:
                st.image(unsharp_mask_img, use_column_width=True)
                st.markdown(f'''**UM Enhanced Image**  
                LOE: {loe_unsharp_mask:.2f}  
                UIQM: {uiqm_unsharp_mask:.2f}  
                UCIQE: {uciqe_unsharp_mask:.2f}''')
            with cols[1]:
                st.image(fused_img, use_column_width=True)
                st.markdown(f'''**Fusion UM Enhanced Image**  
                LOE: {loe_fused:.2f}  
                UIQM: {uiqm_fused:.2f}  
                UCIQE: {uciqe_fused:.2f}''')
            # HEF
            with cols[0]:
                st.image(hef_img, use_column_width=True)
                st.markdown(f'''**HEF Enhanced Image**  
                LOE: {loe_hef:.2f}  
                UIQM: {uiqm_hef:.2f}  
                UCIQE: {uciqe_hef:.2f}''')
            with cols[1]:
                st.image(clahe_hef_img, use_column_width=True)
                st.markdown(f'''**Fusion HEF Enhanced Image**  
                LOE: {loe_clahe_hef:.2f}  
                UIQM: {uiqm_clahe_hef:.2f}  
                UCIQE: {uciqe_clahe_hef:.2f}''')
            # WEVELET
            with cols[0]:
                st.image(wavelet_img, use_column_width=True)
                st.markdown(f'''**Wavelet Enhanced Image**  
                LOE: {loe_wavelet:.2f}  
                UIQM: {uiqm_wavelet:.2f}  
                UCIQE: {uciqe_wavelet:.2f}''')
            with cols[1]:
                st.image(blending_clahe_img, use_column_width=True)
                st.markdown(f'''**Blending Clahe Image**  
                LOE: {loe_blending_clahe:.2f}  
                UIQM: {uiqm_blending_clahe:.2f}  
                UCIQE: {uciqe_blending_clahe:.2f}''')
            
            st.divider()  # 👈 Draws a horizontal rule

        # download image result 
        # download_zip(results)

# image processing
def img_to_bytes(img):
    """Converts an Image object to bytes."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def download_zip(images):
    """Allows the user to download results as a ZIP file."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for _, image, name in images:
            image_bytes = img_to_bytes(image)
            zip_file.writestr(f"{Path(name).stem}_nobg.png", image_bytes)

    st.download_button(
        label="Download All as ZIP",
        data=zip_buffer.getvalue(),
        file_name="background_removed_images.zip",
        mime="application/zip",
    )

# main function
def main():
    setup_page()
    initialize_session()
    uploaded_files = display_ui()
    process_and_display_images(uploaded_files)


if __name__ == "__main__":
    main()
