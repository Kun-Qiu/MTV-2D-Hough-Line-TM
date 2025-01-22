import numpy as np
import cv2
import matplotlib.pyplot as plt
from grid_image_generator import create_centered_grid
import flow_pattern as fp 

def displace_image(image, flow_field):
    """
    Displace an image based on the provided flow field while retaining its original shape.

    :param image        :   2D or 3D NumPy array representing the image.
    :param flow_field   :   Displacement field of shape (height, width, 2) with x and y translations.
    :return:                Displaced image.
    """
    height, width = image.shape[:2]

    # Create meshgrid of original coordinates
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    # Compute new coordinates by adding displacement
    map_x = (X + flow_field[..., 0]).astype(np.float32)
    map_y = (Y + flow_field[..., 1]).astype(np.float32)

    # Remap image to displaced coordinates using interpolation
    displaced_image = cv2.remap(image, map_x, map_y, 
                                interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)

    return displaced_image


if __name__ == "__main__":
    fwhm        = 4             # Full width at half maximum for the Gaussian lines
    spacing     = 20            # Reduced spacing for denser lines
    angle       = 30            # Angle for intersecting lines
    image_size  = (256, 256)    # Size of the image
    num_lines   = 11            # Number of lines
    snr         = 2             # SNR value

    # Generate the grid image
    image = create_centered_grid(image_size, fwhm, spacing, angle, 
                                line_intensity=2, num_lines=num_lines, snr=snr)

    # Define a uniform flow field (shifts 10 pixels right and 5 pixels down)
    # flow_field = fp.poiseuille_flow(10, image.shape[0:2])
    # flow_field = fp.uniform_flow(10, 10, image.shape[0:2])
    flow_field = fp.lamb_oseen_vortex()
    displaced_image = displace_image(image, flow_field)

    # Plot original and displaced images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(1, 2, 2)
    plt.imshow(displaced_image, cmap='gray')
    plt.title('Displaced Image')

    plt.show()
