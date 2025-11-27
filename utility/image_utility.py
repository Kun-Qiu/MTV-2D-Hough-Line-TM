import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
from typing import Tuple
from skimage.morphology import skeletonize, thin


def save_plt(img: np.ndarray, filename: str, cmap: str='gray') -> None:
        plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.savefig(filename, 
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


def skeletonize_img(image: np.ndarray, blur_window: Tuple[int, int]=(5,5), 
                    method: str = 'thin') -> Tuple[np.ndarray, np.ndarray]:
    """
    Function takes in an image read in through cv2.imread, returns a skeletonized image of the
    grid

    :param image        :   Input image
    :param blur_window  :   Window size for the gaussian blur
    :return             :   Skeletonized image
    """
    blur = cv2.GaussianBlur(image, blur_window, 0).astype('uint8')

    # Apply adaptive thresholding and Otsu's thresholding
    _, ot = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to eliminate Noises
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.erode(ot, kernel=kernel)
    thresh = cv2.dilate(thresh, kernel=kernel)

    # Create a gradient mask to isolate specific regions
    grad_x = cv2.Sobel(ot, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(ot, cv2.CV_64F, 0, 1, ksize=5)   
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_mask = (grad_magnitude == 0)
    dark_mask = (ot == 0)
    mask = gradient_mask & dark_mask

    thresh[mask] = 0
    if method == 'thin':
        skeleton = thin(thresh).astype(np.uint8)
    elif method == 'zhang':
        skeleton = skeletonize(thresh, method="zhang").astype(np.uint8)
    elif method == 'lee':
        skeleton = skeletonize(thresh, method="lee").astype(np.uint8)
    else:
        raise ValueError(f"Unknown skeletonization method: {method}")
    
    return thresh, skeleton


def stereo_transform(im: np.ndarray) -> np.ndarray:

    """
    Manually transform image using compression on the left hand side and expansion on the
    right hand side to simulate stereoscopic effects.

    :param im  :    Input Image
    :return    :    Distorted Image   
    """
    
    height, width = im.shape[:2]
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


    def find_mode_pixel_value(image: np.ndarray) -> int:
        """
        Finds the pixel value with the highest frequency in a grayscale image using a histogram.

        :param image    :   Input grayscale image
        :return         :   Pixel value with the highest frequency
        """
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        mode_pixel = np.argmax(hist)
        return mode_pixel

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    transformed_image = cv2.warpPerspective(im, H, (width, height), flags=cv2.INTER_LINEAR)
    mask = (transformed_image == 0)
    background_color = find_mode_pixel_value(im)
    transformed_image[mask] = background_color
    return transformed_image


def transform_image(image: np.ndarray, dx: int=0, dy: int=0) -> np.ndarray:
    (h, w) = image.shape[:2]

    # Define the translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    # Apply the affine transformation (translation only)
    translated_img = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=0)
    return translated_img


if __name__ == "__main__":
    img_path = r"C:\Users\Kun Qiu\Projects\MTV-2D-Hough-Line-TM\data\synthetic_data\Image\SNR_2\5\src.png"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    thresh, skeleton = skeletonize_img(img)

    # Save original image
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('original_image.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save combined mask
    plt.figure(figsize=(5, 5))
    plt.imshow(thresh, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('combined_mask.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Save skeletonized result
    plt.figure(figsize=(5, 5))
    plt.imshow(skeleton, cmap='gray')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('skeleton_result.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()