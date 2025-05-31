import numpy as np
import cv2
import matplotlib.pyplot as plt
from data.Synthetic_Data.grid_image_generator import create_centered_grid
import data.Synthetic_Data.flow_pattern as fp 
import os
import random as rd


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
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
        )

    return displaced_image


if __name__ == "__main__":
    fwhm       = 4          # Full width at half maximum for the Gaussian lines
    spacing    = 20         # Reduced spacing for denser lines
    angle      = 60         # Angle for intersecting lines
    image_size = (256, 256) # Size of the image
    num_lines  = 10         # Number of lines
    snrs       = [1, 2, 4, 8, 16]          # SNR value

    pwd = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(pwd, "Image")
    os.makedirs(base_dir, exist_ok=True)

    for snr in snrs:
        snr_dir = os.path.join(base_dir, f"SNR_{snr}")
        os.makedirs(snr_dir, exist_ok=True)
        
        for i in range(10):
            iter_dir = os.path.join(snr_dir, f"{i}")
            os.makedirs(iter_dir, exist_ok=True)
        
            src = create_centered_grid(
                image_size, fwhm, spacing, angle, 
                line_intensity=0.5, num_lines=num_lines, snr=snr
            )

            # Save the original image
            src_path = os.path.join(iter_dir, "src.png")
            plt.imsave(src_path, src, cmap='gray')

            # Define flow fields (Poiseuille, uniform, Lamb-Oseen)
            flow_fields = {
                "poiseuille": fp.poiseuille_flow(
                    rd.uniform(-20, 20), shape=src.shape, 
                    filename=os.path.join(iter_dir, "poiseuille_flow.npy"), 
                    show=False
                ),
                "uniform": fp.uniform_flow(
                    rd.uniform(-10, 10), rd.uniform(-10, 10), shape=src.shape, 
                    filename=os.path.join(iter_dir, "uniform_flow.npy"), 
                    show=False
                ),
                "lamb_oseen": fp.lamb_oseen_vortex(
                    scale=rd.uniform(100, 200), shape=src.shape, 
                    filename=os.path.join(iter_dir, "lamb_oseen_flow.npy"), 
                    show=False
                )
            }
            
            for flow_name, flow_field in flow_fields.items():
                displaced = displace_image(src, flow_field)
                displaced_path = os.path.join(iter_dir, f"displaced_{flow_name}.png")
                plt.imsave(displaced_path, displaced, cmap='gray')

    print("All images have been saved successfully.")
