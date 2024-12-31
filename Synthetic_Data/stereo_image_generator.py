import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('Synthetic_Data/SNR_2/Set_0/Gaussian_Grid_Image_Set_0.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Define the source points (corners of the original image)
height, width = image.shape[:2]
src_pts = np.float32([
    [0, 0],
    [width, 0],
    [width, height],
    [0, height]
])

dst_pts = np.float32([
    [0, height * 0.2],          # Top-left corner (moved upward slightly)
    [width, 0],                 # Top-right corner (kept at top edge)
    [width, height],            # Bottom-right corner (kept at bottom edge)
    [0, height * 0.8]           # Bottom-left corner (moved downward slightly)
])

# Calculate the perspective transformation matrix
H = cv2.getPerspectiveTransform(src_pts, dst_pts)

# Apply the transformation to the image
transformed_image = cv2.warpPerspective(image, H, (width, height))

# Display the original and transformed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(transformed_image)
plt.axis('off')
plt.show()

