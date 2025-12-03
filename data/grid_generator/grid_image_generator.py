import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_noise(image_array, snr):
    """
    Add Gaussian noise to the image based on the specified SNR.
    
    image_array : Input image array (float32)
    snr         : Desired signal-to-noise ratio (S_p / N_p, where N_p = 4σ)
    :return     : Noisy image (float32)
    """
    sp = np.max(image_array)

    # Calculate required standard deviation
    sigma = sp / (4 * snr)                                  
    noise = np.random.normal(0, sigma, image_array.shape)   # Zero-mean Gaussian noise
    noisy_image = image_array + noise
    return noisy_image


def gaussian_kernel(size, fwhm):
    """
    Generates a 1D Gaussian kernel for simulating the laser lines

    :param size :   Width of the laser line
    :param fwhm :   Full-width half-max width of the laser line
    :return     :   Normalized gaussian mask
    """

    x = np.linspace(-size // 2, size // 2, size)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gaussian = np.exp(-x ** 2 / (2 * sigma ** 2))
    return gaussian / gaussian.sum()


def draw_gaussian_line(image, start_point, end_point, fwhm, intensity=0.5):
    """
    Draws a Gaussian line on the image with specified intensity.

    :param image        :   The image where laser lines are drawn
    :param start_point  :   The start point of the laser lines
    :param end_point    :   The end point of the laser lines
    :param fwhm         :   Full-width half-max width of the laser line
    :param intensity    :   Intensity of the laser lines range(0, 1)
    :return             :   None
    """

    line_img = np.zeros_like(image)
    cv2.line(line_img, start_point, end_point, 255, 1)
    size = int(4 * fwhm)
    kernel = gaussian_kernel(size, fwhm)[:, None]
    gaussian_line_img = cv2.filter2D(line_img, -1, kernel)
    gaussian_line_img = cv2.filter2D(gaussian_line_img, -1, kernel.T)
    np.clip(gaussian_line_img * intensity, 0, 255, out=gaussian_line_img)
    np.add(image, gaussian_line_img, out=image, casting="unsafe")


def create_centered_grid(image_size, fwhm, spacing, angle, line_intensity=0.5, num_lines=10, snr=20):
    """
    Creates a grid image with Gaussian lines intersecting near the center.

    :param image_size       :   The size of the image (Height X Width)
    :param fwhm             :   Full-width half-max width of the laser line
    :param spacing          :   Reduced spacing between the parallel laser lines
    :param angle            :   Angle between intersecting laser lines in left quadrant (degrees)
    :param line_intensity   :   Intensity of the laser lines
    :param num_lines        :   Number of lines in each direction
    :param snr              :   Signal to Noise ratio
    :return                 :   An image with grid
    """

    image = np.zeros(image_size, dtype=np.float32)
    h, w = image_size
    center_x, center_y = w // 2, h // 2

    intersection_angle = np.deg2rad(angle)
    angle1 = intersection_angle/2    # First line angle 
    angle2 = -intersection_angle/2   # Second line angle 
    
    max_dist = int(np.sqrt(w**2 + h**2)) + 100
    for i in range(-num_lines // 2, num_lines // 2 + 1):
        perpendicular_offset = i * spacing
        dx1 = max_dist * np.cos(angle1)
        dy1 = max_dist * np.sin(angle1)
        
        perp_dx = -np.sin(angle1)
        perp_dy = np.cos(angle1)
        offset_x = perpendicular_offset * perp_dx
        offset_y = perpendicular_offset * perp_dy
        start_x = int(center_x + offset_x - dx1)
        start_y = int(center_y + offset_y - dy1)
        end_x = int(center_x + offset_x + dx1)
        end_y = int(center_y + offset_y + dy1)
        draw_gaussian_line(image, (start_x, start_y), (end_x, end_y), fwhm, line_intensity)
    
    # Lines of second angles
    for i in range(-num_lines // 2, num_lines // 2 + 1):
        perpendicular_offset = i * spacing
        dx2 = max_dist * np.cos(angle2)
        dy2 = max_dist * np.sin(angle2)
        perp_dx = -np.sin(angle2)
        perp_dy = np.cos(angle2)
        offset_x = perpendicular_offset * perp_dx
        offset_y = perpendicular_offset * perp_dy
        start_x = int(center_x + offset_x - dx2)
        start_y = int(center_y + offset_y - dy2)
        end_x = int(center_x + offset_x + dx2)
        end_y = int(center_y + offset_y + dy2)
        draw_gaussian_line(image, (start_x, start_y), (end_x, end_y), fwhm, line_intensity)

    noisy_image = add_noise(image, snr)
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    return noisy_image 


if __name__ == "__main__":

    fwhm        = 4            # Full width at half maximum for the Gaussian lines
    spacing     = 10           # Reduced spacing for denser lines
    angle       = 150          # Angle for intersecting lines
    image_size  = (256, 256)   # Size of the image
    num_lines   = 5            # Number of lines
    snr         = 8            # SNR value

    # Generate the grid image
    image = create_centered_grid(
        image_size, fwhm, spacing, angle, 
        line_intensity=0.5, num_lines=num_lines, snr=snr
        )

    plt.figure(figsize=(6, 6))
    plt.title("Centered Gaussian Grid Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()