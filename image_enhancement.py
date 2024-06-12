import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# methods
from methods.clahe import apply_clahe
from methods.unsharp_masking import fusion_clahe_um
from methods.hef import fusion_clahe_hef
from methods.blending_clahe import blending_clahe_percentile

# measurement
from measurement.loe import calculate_LOE
from measurement.uiqm import getUIQM
from measurement.uciqe import getUCIQE

images = ['dataset/image-1.png', 'dataset/image-3.png', 'dataset/image-15.png']
results = []
scores = []

for img in images :
    # image = np.array(Image.open(img))
    image = np.array(Image.open(img).resize((800,600)))

    # Apply enhancement methods
    clahe_img = apply_clahe(image)
    clahe_um_img = fusion_clahe_um(image)
    clahe_hef_img = fusion_clahe_hef(image)
    blending_cp_img = blending_clahe_percentile(image)

    # results.append((image, clahe_img, clahe_um_img, clahe_hef_img, blending_cp_img))
    results.append(image)
    results.append(clahe_img)
    results.append(clahe_um_img)
    results.append(clahe_hef_img)
    results.append(blending_cp_img)

    # Calculate LOE, UIQM,UCIQE for each method
    loe_original, uiqm_original, uciqe_original = calculate_LOE(image,image), getUIQM(image), getUCIQE(image)
    loe_clahe, uiqm_clahe, uciqe_clahe = calculate_LOE(image,clahe_img), getUIQM(clahe_img), getUCIQE(clahe_img)
    loe_clahe_um, uiqm_clahe_um, uciqe_clahe_um = calculate_LOE(image,clahe_um_img), getUIQM(clahe_um_img), getUCIQE(clahe_um_img)
    loe_clahe_hef, uiqm_clahe_hef, uciqe_clahe_hef = calculate_LOE(image,clahe_hef_img), getUIQM(clahe_hef_img), getUCIQE(clahe_hef_img)
    loe_blending_cp, uiqm_blending_cp, uciqe_blending_cp = calculate_LOE(image,blending_cp_img), getUIQM(blending_cp_img), getUCIQE(blending_cp_img)

    # score ="LOE: 0.0\nUIQM: 0.0\nUCIQE: 0.0"
    scores.append(f"LOE: {loe_original:.2f}\nUIQM: {uiqm_original:.2f}\nUCIQE: {uciqe_original:.2f}")
    scores.append(f"LOE: {loe_clahe:.2f}\nUIQM: {uiqm_clahe:.2f}\nUCIQE: {uciqe_clahe:.2f}")
    scores.append(f"LOE: {loe_clahe_um:.2f}\nUIQM: {uiqm_clahe_um:.2f}\nUCIQE: {uciqe_clahe_um:.2f}")
    scores.append(f"LOE: {loe_clahe_hef:.2f}\nUIQM: {uiqm_clahe_hef:.2f}\nUCIQE: {uciqe_clahe_hef:.2f}")
    scores.append(f"LOE: {loe_blending_cp:.2f}\nUIQM: {uiqm_blending_cp:.2f}\nUCIQE: {uciqe_blending_cp:.2f}")

# Number of images per row
num_images = 15
images_per_row = 5

# Define headers
# column_headers = ['Original Image', 'CLAHE', 'Fusion CLAHE-UM', 'Fusion CLAHE-HEF', 'Blending CLAHE Percentile']
column_headers = ['(a)', '(b)', '(c)', '(d)', '(e)']
row_headers = ['Image 1', 'Image 2', 'Image 3']

# Calculate the number of rows needed
num_rows = (num_images + images_per_row - 1) // images_per_row

# Create a figure to display the images
fig, axes = plt.subplots(nrows=num_rows, ncols=images_per_row, figsize=(12, 10))

# Flatten axes array for easy iteration
axes = axes.flatten()

# Set up column headers
for ax, col in zip(axes[:images_per_row], column_headers):
    ax.set_title(col, size=10, weight='bold')

# Set up row headers
for i, row in enumerate(row_headers):
    fig.text(0.05, 1 - (i / num_rows) - 0.5 / num_rows, row, ha='center', va='center', size=10, weight='bold', rotation=90)

# scores ="LOE: 0.0\nUIQM: 0.0\nUCIQE: 0.0"
# Display images and hide axes
for i, ax in enumerate(axes):
    if i < num_images:
        ax.imshow(results[i])
        # ax.text(0.03, 0.15, scores[i], ha='left', va='center', transform=ax.transAxes,
        #         style ='italic', 
        #         fontsize = 7, 
        #         bbox ={'facecolor':'white', 'alpha':0.8, 'pad':2, 'edgecolor':'none'})
    ax.axis('off')  # Hide the axis

plt.tight_layout()
plt.show()
