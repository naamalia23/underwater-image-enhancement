import io
import cv2
import zipfile
from pathlib import Path
import streamlit as st
from PIL import Image
import numpy as np
import uuid

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
#LEO Measurement
def calculate_LOE(original_image, enhanced_image): 
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


MAX_FILES = 5
ALLOWED_TYPES = ["png", "jpg", "jpeg"]


def setup_page():
    """Sets up the Streamlit page configuration."""
    st.set_page_config(page_title="Image Enhancement", page_icon=":star:")
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
    st.sidebar.markdown("### Image Enhancement with CLAHE and Fusion Method")

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
                <p>Developed with ❤ by <a href="https://github.com/naamalia23" target="_blank">@Team</a></p>
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
            original_image = np.array(Image.open(uploaded_file))
            # Apply CLAHE
            clahe_img = apply_clahe(original_image)
            
            # Apply Unsharp Masking
            unsharp_mask_img = unsharp_masking(original_image)

            # Apply image fusion
            fused_image = image_fusion(original_image)

            # Calculate LOE for each method
            loe_clahe = calculate_LOE(original_image, clahe_img)
            loe_unsharp_mask = calculate_LOE(original_image, unsharp_mask_img)
            loe_fused_image = calculate_LOE(original_image, fused_image)
            results.append((original_image, clahe_img, unsharp_mask_img, fused_image, uploaded_file.name))

        for original_image, clahe_img, unsharp_mask_img, fused_image, __name__ in results:
            # Display the images and LOE values
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.image(original_image, caption="Original Image", use_column_width=True)
            with col2:
                st.image(clahe_img, caption=f"CLAHE Enhanced Image\nLOE: {loe_clahe:.2f}", use_column_width=True)
            with col3:
                st.image(unsharp_mask_img, caption=f"Unsharp Masking Enhanced Image\nLOE: {loe_unsharp_mask:.2f}", use_column_width=True)
            with col4:
                st.image(fused_image, caption=f"Fusion Method Enhanced Image\nLOE: {loe_fused_image:.2f}", use_column_width=True)

        ##download_zip(results)


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


def main():
    setup_page()
    initialize_session()
    uploaded_files = display_ui()
    process_and_display_images(uploaded_files)


if __name__ == "__main__":
    main()
