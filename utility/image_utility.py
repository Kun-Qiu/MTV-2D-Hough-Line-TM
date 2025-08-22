from utility.py_import import os, cv2, np, plt, cm
from skimage.morphology import skeletonize


def save_plt(img, filename, cmap='gray'):
        plt.figure()
        plt.imshow(img, cmap=cmap)
        plt.axis('off')
        plt.savefig(filename, 
                   bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()


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
    adaptive_thresh = cv2.erode(ot, kernel=kernel)
    adaptive_thresh = cv2.dilate(ot, kernel=kernel)

    # Create a gradient mask to isolate specific regions
    grad_x = cv2.Sobel(ot, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(ot, cv2.CV_64F, 0, 1, ksize=5)   
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_mask = (grad_magnitude == 0)
    dark_mask = (ot == 0)
    mask = gradient_mask & dark_mask

    adaptive_thresh[mask] = 0
    skeleton = skeletonize(adaptive_thresh).astype(np.uint8)

    # plt.figure(figsize=(15, 10))
    
    # plt.subplot(1, 3, 1)
    # plt.imshow(image, cmap='gray')
    # plt.title('Original Image')
    # plt.axis('off')
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Combined Mask')
    # plt.axis('off')
    
    # plt.subplot(1, 3, 3)
    # plt.imshow(skeleton, cmap='gray')
    # plt.title('Skeletonized Result')
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    return adaptive_thresh, skeleton


def stereo_transform(im):
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


    def find_mode_pixel_value(image):
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


def transform_image(image, dx=0, dy=0):
    """
    Translate an image by a given displacement in x and y.

    :param image: Input image (template)
    :param dx: Displacement in x
    :param dy: Displacement in y
    :return: Translated image
    """
    (h, w) = image.shape[:2]

    # Define the translation matrix
    M = np.float32([[1, 0, dx], [0, 1, dy]])

    # Apply the affine transformation (translation only)
    translated_img = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_LINEAR, 
                                    borderMode=cv2.BORDER_CONSTANT, 
                                    borderValue=0)
    return translated_img