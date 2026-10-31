import numpy as np
import cv2


def find_mode_pixel_value(image):
    """
    Finds the pixel value with the highest frequency in an image.

    :param image    :   Input grayscale image
    :return         :   Pixel value with the highest frequency
    """
    values, counts = np.unique(image, return_counts=True)
    return values[np.argmax(counts)]


def stereo_homography(shape):
    """
    Perspective transform that compresses the left hand side of the image while keeping the
    right hand side, to simulate stereoscopic effects.

    :param shape    :   Image shape (height, width)
    :return         :   3 x 3 homography mapping source to stereo pixel coordinates
    """
    height, width = shape[:2]
    src_pts = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])

    dst_pts = np.float32([
        [0, height * 0.2],          # Top-left corner (moved down)
        [width, 0],                 # Top-right corner (kept at top edge)
        [width, height],            # Bottom-right corner (kept at bottom edge)
        [0, height * 0.8]           # Bottom-left corner (moved up)
    ])
    return cv2.getPerspectiveTransform(src_pts, dst_pts)


def stereo_source_coords(shape):
    """
    Source coordinates q = H^-1 p of every stereo pixel p.

    :param shape    :   Image shape (height, width)
    :return         :   qx, qy (float32)
    """
    h, w = shape[:2]
    H = stereo_homography(shape)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    p = np.stack((X, Y), axis=-1).reshape(-1, 1, 2)
    q = cv2.perspectiveTransform(p, np.linalg.inv(H)).reshape(h, w, 2).astype(np.float32)
    return q[..., 0], q[..., 1]


def stereo_transform(im, border_value=None):
    """
    Manually transform image using compression on the left hand side and expansion on the
    right hand side to simulate stereoscopic effects.

    :param im           :   Input Image
    :param border_value :   Value of pixels mapped from outside the image
                            (default: the most frequent pixel value of im)
    :return             :   Distorted Image
    """
    if border_value is None:
        border_value = find_mode_pixel_value(im)

    height, width = im.shape[:2]
    H = stereo_homography(im.shape)
    return cv2.warpPerspective(
        im, H, (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value)
        )


def inverse_map(flow_field, iterations=10, points=None):
    """
    Converts a forward displacement field into the backward sampling map used by cv2.remap.

    flow_field[y, x] is the displacement of the material at source pixel (x, y). A destination
    point x receives the material from x0, where x0 + u(x0) = x. This is solved by the fixed-point
    iteration x0 <- x - u(x0), which converges when |grad u| < 1. For flows where u is constant
    along its own direction (uniform, Poiseuille) the first iterate is already exact.

    :param flow_field   :   Displacement field of shape (height, width, 2) with x and y translations
    :param iterations   :   Number of fixed-point iterations
    :param points       :   (X, Y) destination coordinates (default: every pixel of the field)
    :return             :   map_x, map_y (float32)
    """
    if points is None:
        h, w = flow_field.shape[:2]
        X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    else:
        X, Y = (np.asarray(c, dtype=np.float32) for c in points)
    flow = np.ascontiguousarray(flow_field, dtype=np.float32)

    map_x, map_y = X, Y
    for _ in range(iterations):
        u0 = cv2.remap(flow, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        map_x, map_y = X - u0[..., 0], Y - u0[..., 1]
    return map_x, map_y


def displace_image(image, flow_field, border_value=0, iterations=10):
    """
    Displace an image based on the provided flow field while retaining its original shape.

    :param image        :   2D or 3D numpy array representing the image
    :param flow_field   :   Displacement field of shape (height, width, 2) with x and y translations
    :param border_value :   Value of pixels whose material comes from outside the image
    :param iterations   :   Fixed-point iterations used to invert the flow field
    :return             :   Displaced image
    """
    map_x, map_y = inverse_map(flow_field, iterations)
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float(border_value)
        )


def stereo_flow(flow_field):
    """
    Ground-truth displacement field as seen in the stereo (perspective-warped) images.

    The material at stereo pixel p sits at source point q = H^-1 p, moves to q + u(q) and is
    imaged at H (q + u(q)). Returns H (q + u(q)) - p on the stereo pixel grid, NaN where q lies
    outside the source image.

    :param flow_field   :   Displacement field of shape (height, width, 2) in source coordinates
    :return             :   Displacement field of shape (height, width, 2) in stereo coordinates
    """
    h, w = flow_field.shape[:2]
    H = stereo_homography(flow_field.shape)
    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    p = np.stack((X, Y), axis=-1)

    q = np.stack(stereo_source_coords(flow_field.shape), axis=-1)
    u = cv2.remap(
        np.ascontiguousarray(flow_field, dtype=np.float32), q[..., 0], q[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
    target = cv2.perspectiveTransform((q + u).reshape(-1, 1, 2), H).reshape(h, w, 2)

    field = (target - p).astype(np.float32)
    outside = (q[..., 0] < 0) | (q[..., 0] > w - 1) | (q[..., 1] < 0) | (q[..., 1] > h - 1)
    field[outside] = np.nan
    return field
