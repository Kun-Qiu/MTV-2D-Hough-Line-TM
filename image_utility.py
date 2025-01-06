import cv2
from skimage.morphology import skeletonize
import numpy as np


def skeletonize_img(image, blur_window=(5,5)):
    """
    Function takes in an image read in through cv2.imread, returns a skeletonized image of the
    grid

    :param image        :   Input image
    :param blur_window  :   Window size for the gaussian blur
    :return             :   Skeletonized image
    """
    blur = cv2.GaussianBlur(image, blur_window, 0).astype('uint8')

    # Apply adaptive thresholding and Otsu's thresholding
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
    return adaptive_thresh, skeletonize(adaptive_thresh).astype(np.uint8)