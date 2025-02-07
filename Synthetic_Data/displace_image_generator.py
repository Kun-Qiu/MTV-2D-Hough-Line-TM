import numpy as np
import cv2
import matplotlib.pyplot as plt
from grid_image_generator import create_centered_grid
import flow_pattern as fp 
import os


def displace_image(image, flow_field):
    """
    Displace an image based on the provided flow field while retaining its original shape.

    :param image        :   2D or 3D numpy array representing the image
    :param flow_field   :   Displacement field of shape (height, width, 2) with x and y translations
    :return             :   Displaced image
    """
    height, width = image.shape[:2]
    X, Y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = (X + flow_field[..., 0]).astype(np.float32)
    map_y = (Y + flow_field[..., 1]).astype(np.float32)
    displaced_image = cv2.remap(image, map_x, map_y, 
                                interpolation=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT)

    return displaced_image


if __name__ == "__main__":
    # Laser / Setup Parameters
    fwhm        = 3             # Full width at half maximum for the Gaussian lines
    spacing     = 25            # Reduced spacing for denser lines
    angle       = 30            # Angle for intersecting lines
    image_size  = (256, 256)    # Size of the image
    num_lines   = 11            # Number of lines
    snr         = 2             # SNR value

    src = create_centered_grid(image_size, fwhm, spacing, angle, 
                               line_intensity=2, num_lines=num_lines, 
                               snr=snr)

    flow_fields = {
        "poiseuille": fp.poiseuille_flow(8, src.shape[0:2]),
        "uniform": fp.uniform_flow(5, 5, src.shape[0:2]),
        "lamb_oseen": fp.lamb_oseen_vortex(scale=200, shape=src.shape[0:2])
    }
    
    save_dir = "Synthetic_Data/Image"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the original image after ensuring the directory exists
    plt.imsave(os.path.join(save_dir, "src.png"), src, cmap='gray')

    # Process and save displaced images
    for flow_name, flow_field in flow_fields.items():
        displaced_image = displace_image(src, flow_field)
        plt.imsave(os.path.join(save_dir,f"displaced_{flow_name}.png"), displaced_image, cmap='gray')

    # Plot original and displaced images for visualization
    plt.figure(figsize=(10, 10))

    plt.subplot(2, 2, 1)
    plt.imshow(src, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    for i, (flow_name, flow_field) in enumerate(flow_fields.items(), start=2):
        displaced_image = displace_image(src, flow_field)
        plt.subplot(2, 2, i)
        plt.imshow(displaced_image, cmap='gray')
        plt.title(f'Displaced Image ({flow_name})')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("All images have been saved successfully.")
