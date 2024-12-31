import numpy as np
import cv2
import random
import os
import matplotlib.pyplot as plt

def gaussian_kernel(size, fwhm):
    """
    Generates a 1D Gaussian kernel for simulating the laser lines

    :param size:    Width of the laser line
    :param fwhm:    Full-width half-max width of the laser line
    :return:        Normalized gaussian mask
    """
    x = np.linspace(-size // 2, size // 2, size)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-x ** 2 / (2 * sigma ** 2))
    return gaussian / gaussian.sum()

def draw_gaussian_line(image, start_point, end_point, fwhm, intensity=3.0):
    """
    Draws a Gaussian line on the image with specified intensity.

    :param image:           The image where laser lines are drawn
    :param start_point:     The start point of the laser lines
    :param end_point:       The end point of the laser lines
    :param fwhm:            Full-width half-max width of the laser line
    :param intensity:       Intensity of the laser lines
    :return:                None
    """
    line_img = np.zeros_like(image)
    cv2.line(line_img, start_point, end_point, 255, 1)
    size = int(4 * fwhm)
    kernel = gaussian_kernel(size, fwhm)[:, None]
    gaussian_line_img = cv2.filter2D(line_img, -1, kernel)
    gaussian_line_img = cv2.filter2D(gaussian_line_img, -1, kernel.T)
    np.clip(gaussian_line_img * intensity, 0, 255, out=gaussian_line_img)
    np.add(image, gaussian_line_img, out=image, casting="unsafe")

def create_laser_grid(image_size, fwhm, spacing, angle, line_intensity=1.0):
    """
    Creates the grid image with Gaussian lines resembling a laser grid.

    :param image_size:          The size of the image (Length X Width)
    :param fwhm:                Full-width half-max width of the laser line
    :param spacing:             Spacing between the parallel laser lines
    :param angle:               Angle between two intersecting laser lines
    :param line_intensity:      Intensity of the laser lines
    :return:                    An image with the simulated laser grid
    """
    image = np.zeros(image_size, dtype=np.float32)
    h, w = image_size
    radians = np.deg2rad(angle)

    # Draw lines at +angle degrees
    for x in range(-w, 2 * w, spacing):
        y1 = 0
        x1 = x
        y2 = h
        x2 = x + int(h / np.tan(radians))
        draw_gaussian_line(image, (x1, y1), (x2, y2), fwhm, intensity=line_intensity)

    # Draw lines at -angle degrees
    for x in range(-w, 2 * w, spacing):
        y1 = 0
        x1 = x
        y2 = h
        x2 = x - int(h / np.tan(radians))
        draw_gaussian_line(image, (x1, y1), (x2, y2), fwhm, intensity=line_intensity)

    # Normalize the image to [0, 255] and convert to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image

def simulate_stereo_images(image, baseline, focal_length):
    """
    Simulate images captured by a two-camera stereoscopic setup.

    :param image:          The input image (base for both cameras).
    :param baseline:       The distance between the two camera centers (stereo baseline).
    :param focal_length:   Focal length of the cameras (in pixels).
    :return:               Left and right images, and their disparity map.
    """
    height, width = image.shape[:2]

    # Generate disparity map (simulated as horizontal parallax)
    disparity = np.zeros_like(image, dtype=np.float32)

    # Simulate parallax for objects at different depths
    depth_range = (50, 200)  # Simulated depth range in arbitrary units
    depths = np.random.uniform(*depth_range, size=(height, width))

    # Calculate pixel disparity based on depth (d = Bf / Z)
    disparity = (baseline * focal_length) / depths
    disparity = disparity.astype(np.int32)

    # Create left and right images with horizontal shifts
    left_image = np.zeros_like(image)
    right_image = np.zeros_like(image)

    for y in range(height):
        for x in range(width):
            shift = disparity[y, x]
            if x - shift >= 0:
                left_image[y, x - shift] = image[y, x]
            if x + shift < width:
                right_image[y, x + shift] = image[y, x]

    return left_image, right_image, disparity

# ------------------------------------- Main Loop -------------------------------------------------------------

# Define parameters for the laser grid simulation
fwhm = 4  # Full width at half maximum for the Gaussian lines
spacing = 25  # Spacing between the lines
angle = 120  # Angle in degrees for the intersecting lines
image_size = (256, 256)  # Size of the image

num_images = 1
output_dir = "LaserGridExperiment"
os.makedirs(output_dir, exist_ok=True)

baseline = 10  # Baseline (distance between cameras) in arbitrary units
focal_length = 100  # Focal length in pixels

for i in range(num_images):
    grid_folder = os.path.join(output_dir, f"Set_{i}")
    os.makedirs(grid_folder, exist_ok=True)  # Ensure the folder is created

    laser_grid_image = create_laser_grid(image_size, fwhm, spacing, angle, line_intensity=2)

    # Simulate stereo images
    left_image, right_image, disparity_map = simulate_stereo_images(laser_grid_image, baseline, focal_length)

    # Save laser grid and stereo images
    # cv2.imwrite(os.path.join(grid_folder, f"Laser_Grid_Image_Set_{i}.png"), laser_grid_image)
    # cv2.imwrite(os.path.join(grid_folder, f"Left_Image_Set_{i}.png"), left_image)
    # cv2.imwrite(os.path.join(grid_folder, f"Right_Image_Set_{i}.png"), right_image)

    # Display the left and right images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Left Image")
    plt.imshow(left_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Right Image")
    plt.imshow(right_image, cmap="gray")
    plt.axis("off")

    plt.show()
