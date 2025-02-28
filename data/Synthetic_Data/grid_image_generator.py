import numpy as np
import cv2
import matplotlib.pyplot as plt


def add_noise(image_array, snr):
    """
    Add noise to the image based on the specified SNR.

    image_array :   Input image array
    snr         :   The desired signal-to-noise ratio
    :return     :   Noise induced image
    """

    noise = np.random.random(image_array.size).reshape(*image_array.shape)
    current_snr = np.max(image_array) / (4 * np.std(noise))
    noise_img = image_array + (noise * (current_snr / snr))
    return noise_img


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


def draw_gaussian_line(image, start_point, end_point, fwhm, intensity=3.0):
    """
    Draws a Gaussian line on the image with specified intensity.

    :param image        :   The image where laser lines are drawn
    :param start_point  :   The start point of the laser lines
    :param end_point    :   The end point of the laser lines
    :param fwhm         :   Full-width half-max width of the laser line
    :param intensity    :   Intensity of the laser lines
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


def create_centered_grid(image_size, fwhm, spacing, angle, line_intensity=1.0, num_lines=10, snr=20):
    """
    Creates a grid image with Gaussian lines intersecting near the center.

    :param image_size       :   The size of the image (Length X Width)
    :param fwhm             :   Full-width half-max width of the laser line
    :param spacing          :   Reduced spacing between the parallel laser lines
    :param angle            :   Angle between intersecting laser lines
    :param line_intensity   :   Intensity of the laser lines
    :param num_lines        :   Number of lines
    :param snr              :   Signal to Noise ratio
    :return                 :   An image with grid
    """

    image = np.zeros(image_size, dtype=np.float32)
    h, w = image_size
    center_x, center_y = w // 2, h // 2 
    radians = np.deg2rad(angle)

    # Draw lines intersecting at the center of the image
    for i in range(-num_lines // 2, num_lines // 2):
        offset = i * spacing
        
        # Calculate line start and end points for proper centering
        x1, y1 = center_x + offset - int(center_y / np.tan(radians)), 0
        x2, y2 = center_x + offset + int(center_y / np.tan(radians)), h
        draw_gaussian_line(image, (x1, y1), (x2, y2), fwhm, intensity=line_intensity)

        x1, y1 = center_x + offset + int(center_y / np.tan(radians)), 0
        x2, y2 = center_x + offset - int(center_y / np.tan(radians)), h
        draw_gaussian_line(image, (x1, y1), (x2, y2), fwhm, intensity=line_intensity)

    image = add_noise(np.clip(image, 0, 255).astype(np.uint8), snr)
    return image


if __name__ == "__main__":

    fwhm        = 4             # Full width at half maximum for the Gaussian lines
    spacing     = 20            # Reduced spacing for denser lines
    angle       = 30            # Angle for intersecting lines
    image_size  = (256, 256)    # Size of the image
    num_lines   = 10            # Number of lines
    snr         = 8             # SNR value

    # Generate the grid image
    image = create_centered_grid(image_size, fwhm, spacing, angle, 
                                line_intensity=2, num_lines=num_lines, snr=snr)

    plt.figure(figsize=(6, 6))
    plt.title("Centered Gaussian Grid Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()