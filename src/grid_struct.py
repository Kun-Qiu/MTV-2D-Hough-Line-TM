
__author__ = "Kun Qiu"
__credits__ = ["Kun Qiu"]
__version__ = "1.01"
__maintainer__ = "Kun Qiu"
__email__ = "qiukun1234@gmail.com"
__status__ = "Production"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from src.image_utility import transform_image
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale 
from scipy.ndimage import rotate


class GridStruct:
    def __init__(self, pos_lines, neg_lines, ref_im, mov_im, temp_scale=0.67, 
                 window_scale=1.2, search_scale=2, down_scale=4, rotate_range=30,
                 opt_generations=3, opt_shrink_factor=2):
        """
        Default constructor

        :param pos_lines        : [theta, rho] of positively sloped lines  
        :param neg_lines        : [theta, rho] of negatively sloped lines
        :param ref_im           : Reference image (Before Transformation)
        :param mov_im           : Moving image (After Transformation)
        :param temp_scale       : Scale of the template
        :param window_scale     : Scale of window such that template is located within
        :param search_scale     : Scale of the search region
        :param down_scale       : Down scale size
        :param rotate_range     : Range of rotation for template matching
        """

        def sort_lines(lines):
            """
            Sorted by the rho value
            """
            return lines[np.argsort(lines[:, 1])]

        # Shape = (11, 11, 2)
        self.shape              = (len(pos_lines), len(neg_lines))   
        self.num_intersections  = 0
        
        self.reference_img    = ref_im
        self.moving_img       = mov_im
        self.rotation_range   = rotate_range

        # Obtain the coarse global linear shift in image
        self.shifts, _, _ = phase_cross_correlation(
            rescale(self.reference_img, 1 / down_scale, anti_aliasing=True), 
            rescale(self.moving_img, 1 / down_scale, anti_aliasing=True))
        self.shifts *= down_scale

        # Public variables
        self.t0_grid        = np.empty(self.shape, dtype=object)
        self.dt_grid        = np.empty(self.shape, dtype=object)
        self.template       = np.empty(self.shape, dtype=object)

        # self.opt_generations = opt_generations
        # self.opt_shrink_factor = opt_shrink_factor
        # self.param_radii = np.array([5.0, 5.0, 5.0])  # dx, dy, angle
        # self.param_nsteps = np.array([5, 5, 5])

        ### Immediately initialize and populate the data structure ###
        self.__populate_grid(sort_lines(pos_lines), sort_lines(neg_lines))
        self.__generate_template(scale=temp_scale)
        self.__solve_dt_grid(window_scale=window_scale, search_scale=search_scale)


    def __is_within_bounds(self, x, y):
        """
        Check if a point (x, y) lies within the image boundaries.

        :param x: x-coordinate
        :param y: y-coordinate

        :return: True if within bounds, False otherwise
        """
        height, width = np.shape(self.reference_img)
        return 0 <= x < width and 0 <= y < height
        
    @staticmethod
    def __find_intersection(line1, line2):
        """
        Find the intersection of two lines given in rho-theta representation.

        :param line1: [rho, theta] for the first line
        :param line2: [rho, theta] for the second line

        :return: (x, y) intersection coordinates or None if lines are parallel
        """

        theta1, rho1 = line1
        theta2, rho2 = line2

        # Line equations in Cartesian form: x * cos(theta) + y * sin(theta) = rho
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])

        try:
            x, y = np.linalg.solve(A, b)
            return x, y
        except np.linalg.LinAlgError:
            # Lines are parallel, no intersection
            return None


    def __populate_grid(self, pos_lines, neg_lines):
        """
        Populate the grid with the intersection points of positive and negative lines.

        :param pos_lines    :   Lines with postive slope
        :param neg_lines    :   Lines with negative slope
        :return             :   None
        """
        for i, pos_line in enumerate(pos_lines):
            for j, neg_line in enumerate(neg_lines):
                intersection = self.__find_intersection(pos_line, neg_line)
                if intersection is None or not self.__is_within_bounds(intersection[0], 
                                                                       intersection[1]):
                    intersection = (np.nan, np.nan)

                self.t0_grid[i, j]        = intersection
                self.num_intersections   += 1
    

    def __grid_img_bound(self, i, j):
        """
        Given the center of point, determine the maximum bounding box for the template
        using at max 4 adjacent points and at min 1 adjacent point

        :params i   :   Index of point in structure
        :params j   :   Index of point in structure
        :return     :   Half width and half height of crop region 
        """

        if not (0 <= i < self.t0_grid.shape[0] and 0 <= j < self.t0_grid.shape[1]):
            raise IndexError("Center index (i, j) is out of bounds.")

        x_c, y_c = self.t0_grid[i, j]

        min_half_width, min_half_height = 0, 0
        max_distance = 0

        if j + 1 < self.t0_grid.shape[1] and not np.isnan(self.t0_grid[i, j + 1]).any():
            # Bottom right node
            x_br, y_br = self.t0_grid[i, j + 1]
            dist = np.sqrt((x_br - x_c) ** 2 + (y_br - y_c) ** 2)
            if dist > max_distance:
                max_distance = dist
                min_half_width = x_br - x_c
                min_half_height = y_br - y_c

        if i + 1 < self.t0_grid.shape[0] and not np.isnan(self.t0_grid[i + 1, j]).any():
            # Bottom left node 
            x_bl, y_bl = self.t0_grid[i + 1, j]
            dist = np.sqrt((x_bl - x_c) ** 2 + (y_bl - y_c) ** 2)
            if dist > max_distance:
                max_distance = dist
                min_half_width = x_bl - x_c
                min_half_height = y_bl - y_c

        if i - 1 >= 0 and not np.isnan(self.t0_grid[i - 1, j]).any():
            # Top right Node
            x_tr, y_tr = self.t0_grid[i - 1, j]
            dist = np.sqrt((x_tr - x_c) ** 2 + (y_tr - y_c) ** 2)
            if dist > max_distance:
                max_distance = dist
                min_half_width = x_tr - x_c
                min_half_height = y_tr - y_c

        if j - 1 >= 0 and not np.isnan(self.t0_grid[i, j - 1]).any():
            # Top left Node
            x_tl, y_tl = self.t0_grid[i, j - 1]
            dist = np.sqrt((x_tl - x_c) ** 2 + (y_tl - y_c) ** 2)
            if dist > max_distance:
                max_distance = dist
                min_half_width = x_tl - x_c
                min_half_height = y_tl - y_c

        return np.array([abs(min_half_width), abs(min_half_height)])


    def __generate_template(self, scale=0.7):
        """
        Create template patches using the grid intersections and the search patch for
        the consecutive frame.

        :param image    :   The image to crop, as a NumPy array.
        :param scale    :   The width of crop scale --> scale=1: from intersection to intersection
        """
        height, width = np.shape(self.reference_img)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                
                # Crop based on bottom right node
                center = self.t0_grid[i, j]
                if (
                    np.isnan(center).any()
                ):
                    continue

                x_half, y_half = self.__grid_img_bound(i, j)

                rect_half_width     = scale * x_half
                rect_half_height    = scale * y_half
                x_center, y_center  = center
                
                # Ensure the coordinates are within the image boundaries
                x_min = int(x_center - rect_half_width)
                y_min = int(y_center - rect_half_height)
                x_max = int(x_center + rect_half_width)
                y_max = int(y_center + rect_half_height)

                if (x_min < 0 or y_min < 0 or x_max > width or y_max > height):
                    # Skip if the crop region is near the image boundary
                    continue

                self.template[i, j] = np.array([x_min, y_min, x_max, y_max])


    def __solve_dt_grid(self, window_scale=1.1, search_scale=2):
        """
        Create search patches for the template matching algorithm by maximizing
        similarity between self.reference_img and self.moving_img

        :param window_scale :   Window scaling constant
        :param search_scale :   Search region scaling constant
        """
        
        assert window_scale >= 1, "window_scale must be greater than or equal to 1"
        assert search_scale >= 2, "search_scale must be greater than or equal to 2"

        warped_search_im = transform_image(self.moving_img, self.shifts[1], self.shifts[0])
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                
                center = self.t0_grid[i, j]
                if np.isnan(center).any() or np.any((self.template[i, j]) is None):
                    continue

                x_min, y_min, x_max, y_max  = self.template[i, j]
                rect_width                  = abs((x_max - x_min))
                rect_height                 = abs((y_max - y_min))
                x_center, y_center          = center
                bound_y, bound_x            = np.shape(self.reference_img)

                def get_bound(x_c, y_c, width, height, scale, bound_width, bound_height):
                    bound_x_min = max(0, int(x_c - (scale * width) / 2))
                    bound_x_max = min(bound_width, int(x_c + (scale * width) / 2))
                    bound_y_min = max(0, int(y_c - (scale * height) / 2))
                    bound_y_max = min(bound_height, int(y_c + (scale * height) / 2))

                    return np.array([bound_x_min, bound_y_min, bound_x_max, bound_y_max])
                
                temp_x_min, temp_y_min, temp_x_max, temp_y_max          = get_bound(x_center, y_center, 
                                                                                    rect_width, rect_height, 
                                                                                    window_scale, bound_x,
                                                                                    bound_y)
                search_x_min, search_y_min, search_x_max, search_y_max  = get_bound(x_center, y_center, 
                                                                                    rect_width, rect_height, 
                                                                                    search_scale, bound_x,
                                                                                    bound_y)
                
                template                = self.reference_img[temp_y_min:temp_y_max, temp_x_min:temp_x_max]
                search_region_warped    = warped_search_im[search_y_min:search_y_max, search_x_min:search_x_max]

                opt_score = -np.inf
                opt_loc, opt_res = None, None

                for angle in range(-self.rotation_range, self.rotation_range, 5):
                    rotate_center   = (template.shape[1] // 2, template.shape[0] // 2)
                    rot_mat         = cv2.getRotationMatrix2D(rotate_center, angle, 1)
                    rotate_dst      = cv2.warpAffine(template, rot_mat, (template.shape[1], template.shape[0]))    
                    
                    match_result            = cv2.matchTemplate(search_region_warped, rotate_dst, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc  = cv2.minMaxLoc(match_result)
                    
                    if max_val > opt_score:
                        opt_score = max_val
                        opt_loc   = max_loc
                        opt_res   = match_result

                """
                Refine the subpixel location of the best match using a 2D quadratic fit.
                """
                x_opt, y_opt = opt_loc
                dx_sub, dy_sub = 0, 0
                if 1 <= x_opt < opt_res.shape[1] - 1 and 1 <= y_opt < opt_res.shape[0] - 1:
                    patch = opt_res[y_opt - 1:y_opt + 2, x_opt - 1:x_opt + 2]
                    X, Y = np.meshgrid(np.linspace(-1, 1, 3), np.linspace(-1, 1, 3))
                    X, Y, Z = X.flatten(), Y.flatten(), patch.flatten()
                    dx_sub, dy_sub, _ = self.__quad_opt2D(X, Y, Z)
                
                x_opt = search_x_min + x_opt + dx_sub + template.shape[1] / 2 - self.shifts[1]
                y_opt = search_y_min + y_opt + dy_sub + template.shape[0] / 2 - self.shifts[0]
                self.dt_grid[i, j] = np.array([x_opt, y_opt])


    @staticmethod
    def __quad_opt2D(X, Y, Z, x_lim=None, y_lim=None):
        """
        Finds the maximum or critical point of a 2D quadratic fit.

        Parameters:
            X, Y, Z : 1D numpy arrays of coordinates and corresponding values.
            x_lim, y_lim : Limits for valid peak selection.

        Returns:
            x_opt, y_opt, z_opt: Optimal peak coordinates.
        """
        X, Y, Z = np.array(X).flatten(), np.array(Y).flatten(), np.array(Z).flatten()

        if x_lim is None:
            x_lim = (np.min(X), np.max(X))
        if y_lim is None:
            y_lim = (np.min(Y), np.max(Y))

        mat = np.column_stack([X**2, Y**2, X*Y, X, Y, np.ones_like(X)])
        A, _, _, _ = np.linalg.lstsq(mat, Z, rcond=None)

        # Solve dz/dx = dz/dy = 0 for [x, y] (critical point)
        mat_critical    = np.array([[2 * A[0], A[2]], 
                                    [A[2], 2 * A[1]]], dtype=np.float32)
        rhs             = np.array([-A[3], -A[4]], dtype=np.float32)
        cpoint          = np.linalg.solve(mat_critical, rhs)

        # Compute discriminant: d = dzdx*dzdy - (dz2dxdy)^2
        d = 4 * A[0] * A[1] - A[2]**2

        # If a max exists in valid bounds, use that; otherwise, take the max Z
        if (cpoint[0] < x_lim[0] or cpoint[0] > x_lim[1]
            or cpoint[1] < y_lim[0] or cpoint[1] > y_lim[1]
            or d <= 0
        ):
            max_idx = np.argmax(Z)
            x_opt, y_opt, z_opt = X[max_idx], Y[max_idx], Z[max_idx]
        else:
            x_opt, y_opt = cpoint
            z_opt = A @ np.array([x_opt**2, y_opt**2, x_opt*y_opt, x_opt, y_opt, 1])

        return x_opt, y_opt, z_opt
