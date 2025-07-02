import numpy as np
import cv2
import matplotlib.pyplot as plt
# from data.Synthetic_Data.grid_image_generator import create_centered_grid
# import data.Synthetic_Data.flow_pattern as fp 
from grid_image_generator import create_centered_grid
import flow_pattern as fp 
import os
import random as rd


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


def displace_image(image, flow_field):
    """
    Displace an image based on the provided flow field while retaining its original shape.

    :param image        :   2D or 3D numpy array representing the image
    :param flow_field   :   Displacement field of shape (height, width, 2) with x and y translations
    :return             :   Displaced image
    """
    height, width = image.shape[:2]
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = (X - flow_field[..., 0]).astype(np.float32)
    map_y = (Y - flow_field[..., 1]).astype(np.float32)
    displaced_image = cv2.remap(
        image, map_x, map_y, 
        interpolation=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT
        )

    return displaced_image


if __name__ == "__main__":
    fwhm       = 10             # Full width at half maximum for the Gaussian lines
    spacing    = 30             # Reduced spacing for denser lines
    angle      = 60             # Angle for intersecting lines
    image_size = (512, 512)     # Size of the image
    num_lines  = 10             # Number of lines
    snrs       = [1, 2, 4, 8, 16]  # SNR value
    num_sets   = 20

    pwd = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(pwd, "Image")
    os.makedirs(base_dir, exist_ok=True)

    for snr in snrs:
        snr_dir = os.path.join(base_dir, f"SNR_{snr}")
        os.makedirs(snr_dir, exist_ok=True)
        
        for i in range(num_sets):
            iter_dir = os.path.join(snr_dir, f"{i}")
            os.makedirs(iter_dir, exist_ok=True)
        
            src = create_centered_grid(
                image_size, fwhm, spacing, angle, 
                line_intensity=0.5, num_lines=num_lines, 
                snr=snr
            )

            src_path = os.path.join(iter_dir, "src.png")
            plt.imsave(src_path, src, cmap='gray')

            src_stereo_path = os.path.join(iter_dir, "stereo_src.png")
            src_stereo = stereo_transform(src)
            plt.imsave(src_stereo_path, src_stereo, cmap='gray')

            flow_fields = {
                "poiseuille": fp.poiseuille_flow(
                    rd.uniform(-15, 15), shape=src.shape, 
                    filename=os.path.join(iter_dir, "poiseuille_flow"), 
                    show=False
                ),
                "uniform": fp.uniform_flow(
                    rd.uniform(-10, 10), rd.uniform(-10, 10), shape=src.shape, 
                    filename=os.path.join(iter_dir, "uniform_flow"), 
                    show=False
                ),
                "lamb_oseen": fp.lamb_oseen_vortex(
                    scale=rd.uniform(100, 200), shape=src.shape, 
                    filename=os.path.join(iter_dir, "lamb_oseen_flow"), 
                    show=False
                )
            }
            
            for flow_name, flow_field in flow_fields.items():
                displaced = displace_image(src, flow_field)
                displaced_stereo = displace_image(src_stereo, flow_field)

                displaced_path = os.path.join(iter_dir, f"displaced_{flow_name}.png")
                displaced_stereo_path = os.path.join(iter_dir, f"displaced_stereo_{flow_name}.png")
                plt.imsave(displaced_path, displaced, cmap='gray')
                plt.imsave(displaced_stereo_path, displaced_stereo, cmap='gray')

    print("All images have been saved successfully.")
