import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize

def skeletonize_img(image, blur_window=(5, 5), min_size=10):
    """
    Function to skeletonize an image.

    :param image: Input image (assumed grayscale or single channel)
    :param blur_window: Tuple representing the Gaussian blur kernel size
    :return: Skeletonized image
    """
    blur = cv2.GaussianBlur(image, blur_window, 0).astype('uint8')

    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, ot = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to Eliminate Noises
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh = cv2.erode(adaptive_thresh, kernel=kernel)
    adaptive_thresh = cv2.dilate(adaptive_thresh, kernel=kernel)

    # Create a gradient mask to isolate specific regions
    grad_x = cv2.Sobel(ot, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(ot, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_mask = (grad_magnitude == 0)
    dark_mask = (ot == 0)
    mask = gradient_mask & dark_mask

    adaptive_thresh[mask] = 0
    skeleton = skeletonize(adaptive_thresh).astype(np.uint8)
    return adaptive_thresh, skeleton


def find_mode_pixel_value(image):
    """
    Finds the pixel value with the highest frequency in a grayscale image using a histogram.

    :param image: Input grayscale image
    :return: Pixel value with the highest frequency
    """
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    mode_pixel = np.argmax(hist)  # Pixel value with the highest frequency
    return mode_pixel


# Load the image
image = cv2.imread("Experimental_Data/Source/frame_1.png", cv2.IMREAD_GRAYSCALE)

# Define the source and destination points
height, width = image.shape[:2]
src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
dst_pts = np.float32([[0, height * 0.2], [width, 0], [width, height], [0, height * 0.8]])

# Perspective transformation
H = cv2.getPerspectiveTransform(src_pts, dst_pts)
transformed_image = cv2.warpPerspective(image, H, (width, height), flags=cv2.INTER_LINEAR)

# Create a mask of black regions in the transformed image
mask = (transformed_image == 0)

# Determine the background color using the mode of pixel values
background_color = find_mode_pixel_value(image)
transformed_image[mask] = background_color

# Skeletonize the image
thresh, skeleton = skeletonize_img(transformed_image)

# Visualize the original, transformed, and skeletonized images
plt.figure(figsize=(15, 5))

plt.subplot(1, 4, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 4, 2)
plt.title("Transformed Image (Blended)")
plt.imshow(transformed_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 3)
plt.title("Transformed Image (Blended)")
plt.imshow(thresh, cmap='gray')
plt.axis('off')

plt.subplot(1, 4, 4)
plt.title("Skeletonized Image")
plt.imshow(skeleton, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
