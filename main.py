import numpy as np

# I/O
import io
import zipfile
from pathlib import Path
from PIL import Image

import cv2
import streamlit as st
import uuid

# methods
from methods.clahe import apply_clahe
from methods.unsharp_masking import fusion_clahe_um
from methods.hef import fusion_clahe_hef
from methods.blending_clahe import blending_clahe_percentile

# measurement
from measurement.loe import calculate_LOE
from measurement.uiqm import getUIQM
from measurement.uciqe import getUCIQE

# streamlit 
# file uploads setup
MAX_FILES = 5
ALLOWED_TYPES = ["png", "jpg", "jpeg"]

def setup_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(page_title="Underwater Image Enhancement", page_icon=":star:", layout="wide") 
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

    # results = []

    with st.spinner("Enhancing Images..."):
        for uploaded_file in uploaded_files:
            __name__ = uploaded_file.name
            # image
            # image = np.array(Image.open(uploaded_file))
            image = np.array(Image.open(uploaded_file).resize((800,600)))

            # Apply enhancement methods
            clahe_img = apply_clahe(image)
            clahe_um_img = fusion_clahe_um(image)
            clahe_hef_img = fusion_clahe_hef(image)
            blending_cp_img = blending_clahe_percentile(image)

            # Calculate LOE for each method
            loe_original = calculate_LOE(image, image)
            loe_clahe = calculate_LOE(image, clahe_img)
            loe_clahe_um = calculate_LOE(image, clahe_um_img)
            loe_clahe_hef = calculate_LOE(image, clahe_hef_img)
            loe_blending_cp = calculate_LOE(image, blending_cp_img)

            # Calculate UIQM,UCIQE for each method
            uiqm_original,uciqe_original = getUIQM(image), getUCIQE(image)
            uiqm_clahe,uciqe_clahe = getUIQM(clahe_img), getUCIQE(clahe_img)
            uiqm_clahe_um,uciqe_clahe_um = getUIQM(clahe_um_img), getUCIQE(clahe_um_img)
            uiqm_clahe_hef,uciqe_clahe_hef = getUIQM(clahe_hef_img), getUCIQE(clahe_hef_img)
            uiqm_blending_cp, uciqe_blending_cp = getUIQM(blending_cp_img), getUCIQE(blending_cp_img)

        #     # append image result
        #     results.append((image, clahe_img,  clahe_um_img, clahe_hef_img, blending_cp_img, uploaded_file.name))  
            
        # for image, clahe_img,  clahe_um_img, clahe_hef_img, blending_cp_img, __name__ in results:
            # Display the images and measurement values
            st.markdown(f"**{__name__}**\n\n")
            cols = st.columns(5) # number of columns in each row! = 5
            # original
            with cols[0]:
                st.image(image, use_column_width=True)
                st.markdown(f'''**Original Image**   
                    LOE: {loe_original:.2f}  
                    UIQM: {uiqm_original:.2f}  
                    UCIQE: {uciqe_original:.2f}''')
            # clahe
            with cols[1]:
                st.image(clahe_img, use_column_width=True)
                st.markdown(f'''**CLAHE Image**  
                    LOE: {loe_clahe:.2f}  
                    UIQM: {uiqm_clahe:.2f}  
                    UCIQE: {uciqe_clahe:.2f}''')
            # fusion CLAHE UM
            with cols[2]:
                st.image(clahe_hef_img, use_column_width=True)
                st.markdown(f'''**Fusion CLAHE-UM Image**  
                    LOE: {loe_clahe_um:.2f}  
                    UIQM: {uiqm_clahe_um:.2f}  
                    UCIQE: {uciqe_clahe_um:.2f}''')
            # fusion CLAHE HEF
            with cols[3]:
                st.image(clahe_um_img, use_column_width=True)
                st.markdown(f'''**Fusion CLAHE-HEF Image**  
                    LOE: {loe_clahe_hef:.2f}  
                    UIQM: {uiqm_clahe_hef:.2f}  
                    UCIQE: {uciqe_clahe_hef:.2f}''')
            # belending clahe
            with cols[4]:
                st.image(blending_cp_img, use_column_width=True)
                st.markdown(f'''**Blending CLAHE Percentile Image**  
                    LOE: {loe_blending_cp:.2f}  
                    UIQM: {uiqm_blending_cp:.2f}  
                    UCIQE: {uciqe_blending_cp:.2f}''')
            
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
