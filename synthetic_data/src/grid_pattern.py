import numpy as np


FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def to_uint8(image):
    """
    Round and clip a float image to the uint8 range.

    :param image    :   Input image (float)
    :return         :   uint8 image
    """
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def add_noise(image_array, snr, signal=None, rng=None):
    """
    Add zero-mean Gaussian noise to the image based on the specified SNR.

    :param image_array  :   Input image array (float)
    :param snr          :   Desired signal-to-noise ratio S_p / N_p, where S_p is the peak line
                            intensity above the background and N_p = 4σ
    :param signal       :   Peak line intensity above the background, S_p
                            (default: max - min of the image)
    :param rng          :   numpy random Generator (default: a new unseeded generator)
    :return             :   Noisy image (float32, not clipped)
    """
    rng = np.random.default_rng() if rng is None else rng
    if signal is None:
        signal = float(np.max(image_array) - np.min(image_array))

    sigma = signal / (4 * snr)
    noise = rng.normal(0.0, sigma, image_array.shape)
    return (image_array + noise).astype(np.float32)


def grid_intensity(X, Y, fwhm, spacing, angle, center, line_intensity=0.5, num_lines=10,
                   background=30.0):
    """
    Evaluates the noise-free grid intensity at arbitrary (sub-pixel) source coordinates.

    The intensity of each line is computed analytically from the perpendicular distance d of
    the point to the line, I = A * exp(-d^2 / (2 sigma^2)), so peak intensity and width do not
    depend on the line angle. Intensities of crossing lines add up (clipped to 255 on conversion).
    Passing displaced or perspective-mapped coordinates renders the warped grid without any
    interpolation error.

    :param X, Y             :   Source x and y coordinates of every output pixel
    :param fwhm             :   Full-width half-max width of the laser line [px]
    :param spacing          :   Perpendicular spacing between the parallel laser lines [px]
    :param angle            :   Angle between intersecting laser lines in left quadrant (degrees)
    :param center           :   (x, y) of the grid center
    :param line_intensity   :   Peak intensity of a single line above background, range(0, 1)
    :param num_lines        :   Number of lines in each direction
    :param background       :   Background intensity (0 - 255)
    :return                 :   Grid image (float32)
    """

    cx, cy = center
    sigma = fwhm * FWHM_TO_SIGMA
    amplitude = 255.0 * line_intensity

    X = np.asarray(X, dtype=np.float32) - cx
    Y = np.asarray(Y, dtype=np.float32) - cy

    # Offsets symmetric about the center for both odd and even num_lines
    offsets = (np.arange(num_lines) - (num_lines - 1) / 2) * spacing

    image = np.full(X.shape, background, dtype=np.float32)
    half_angle = np.deg2rad(angle) / 2
    for theta in (half_angle, -half_angle):
        # Signed distance from the center line, measured along the normal (-sin θ, cos θ)
        dist = -X * np.sin(theta) + Y * np.cos(theta)
        for offset in offsets:
            image += amplitude * np.exp(-(dist - offset) ** 2 / (2 * sigma ** 2))
    return image


def render_grid(image_size, fwhm, spacing, angle, line_intensity=0.5, num_lines=10,
                background=30.0, center=None):
    """
    Renders a noise-free grid of Gaussian laser lines with sub-pixel accuracy.

    :param image_size       :   The size of the image (Height X Width)
    :param fwhm             :   Full-width half-max width of the laser line [px]
    :param spacing          :   Perpendicular spacing between the parallel laser lines [px]
    :param angle            :   Angle between intersecting laser lines in left quadrant (degrees)
    :param line_intensity   :   Peak intensity of a single line above background, range(0, 1)
    :param num_lines        :   Number of lines in each direction
    :param background       :   Background intensity (0 - 255)
    :param center           :   (x, y) of the grid center (default: image center)
    :return                 :   Grid image (float32)
    """

    h, w = image_size
    if center is None:
        center = ((w - 1) / 2, (h - 1) / 2)
    Y, X = np.mgrid[0:h, 0:w]
    return grid_intensity(X, Y, fwhm, spacing, angle, center, line_intensity, num_lines, background)


def create_centered_grid(image_size, fwhm, spacing, angle, line_intensity=0.5, num_lines=10,
                         snr=20, background=30.0, rng=None):
    """
    Creates a grid image with Gaussian lines intersecting near the center.

    :param image_size       :   The size of the image (Height X Width)
    :param fwhm             :   Full-width half-max width of the laser line
    :param spacing          :   Perpendicular spacing between the parallel laser lines
    :param angle            :   Angle between intersecting laser lines in left quadrant (degrees)
    :param line_intensity   :   Peak intensity of a single line above background, range(0, 1)
    :param num_lines        :   Number of lines in each direction
    :param snr              :   Signal to Noise ratio (None for a noise-free image)
    :param background       :   Background intensity (0 - 255), keeps noise from being clipped
    :param rng              :   numpy random Generator for the noise
    :return                 :   An image with grid (uint8)
    """

    image = render_grid(image_size, fwhm, spacing, angle, line_intensity, num_lines, background)
    if snr is not None:
        image = add_noise(image, snr, signal=255.0 * line_intensity, rng=rng)
    return to_uint8(image)
