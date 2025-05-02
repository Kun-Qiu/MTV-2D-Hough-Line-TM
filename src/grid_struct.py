from utility.py_import import np, cv2

from utility.image_utility import transform_image
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale


class GridStruct:
    def __init__(self, pos_lines: np.ndarray, neg_lines: np.ndarray, ref_im: np.ndarray, 
                 mov_im: np.ndarray, temp_scale: float=0.67, 
                 window_scale: float=1.2, search_scale:float=2, down_scale:int=4, 
                 rotate_range:int=45):
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

        def sort_lines(lines: np.ndarray) -> np.ndarray:
            """
            Sorted by the rho value
            """
            return lines[np.argsort(lines[:, 1])]

        # Shape = (11, 11, 2)
        self.shape            = (len(pos_lines), len(neg_lines))
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
        self.template_param = np.empty(self.shape, dtype=object)

        ### Immediately initialize and populate the data structure ###
        self._populate_grid(sort_lines(pos_lines), sort_lines(neg_lines))
        self._generate_template(scale=temp_scale)
        self._solve_dt_grid(window_scale=window_scale, search_scale=search_scale)


    def _is_within_bounds(self, x: int, y: int) -> bool:
        """
        Check if a point (x, y) lies within the image boundaries.

        :param x: x-coordinate
        :param y: y-coordinate

        :return: True if within bounds, False otherwise
        """
        height, width = np.shape(self.reference_img)
        return 0 <= x < width and 0 <= y < height


    @staticmethod
    def _find_intersection(line1: np.ndarray, line2):
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
            return (x, y)
        except np.linalg.LinAlgError:
            # Lines are parallel, no intersection
            return None


    def _populate_grid(self, pos_lines: np.ndarray, neg_lines: np.ndarray) -> None:
        """
        Populate the grid with the intersection points of positive and negative lines.

        :param pos_lines    :   Lines with postive slope
        :param neg_lines    :   Lines with negative slope
        :return             :   None
        """
        for i, pos_line in enumerate(pos_lines):
            for j, neg_line in enumerate(neg_lines):
                intersection = self._find_intersection(pos_line, neg_line)
                if intersection is None or not self._is_within_bounds(intersection[0], 
                                                                      intersection[1]):
                    intersection = (np.nan, np.nan)

                self.t0_grid[i, j]  = intersection
    

    def _grid_img_bound(self, i: int, j: int) -> tuple:
        """
        Given the center of point, determine the maximum bounding box for the template
        using at max 4 adjacent points and at min 1 adjacent point
        """
        
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise IndexError("Center index (i, j) is out of bounds.")

        x_c, y_c = self.t0_grid[i, j]
        dx_max, dy_max = 0, 0
        max_distance = 0

        # Check all adjacent points
        directions = [
            (i, j + 1),  # Right
            (i + 1, j),  # Bottom
            (i - 1, j),  # Top
            (i, j - 1)   # Left
        ]

        for ni, nj in directions:
            if 0 <= ni < self.shape[0] and 0 <= nj < self.shape[1] and not np.isnan(self.t0_grid[ni, nj]).any():
                x_adj, y_adj = self.t0_grid[ni, nj]
                dx = x_adj - x_c
                dy = y_adj - y_c
                dist = np.hypot(dx, dy)
                if dist > max_distance:
                    max_distance = dist
                    dx_max, dy_max = dx, dy

        return (np.array([abs(dx_max), abs(dy_max)]))


    def _generate_template(self, scale=0.7):
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

                x_half, y_half = self._grid_img_bound(i, j)

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


    def _solve_dt_grid(self, window_scale=1.1, search_scale=2):
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
                opt_loc   = None

                for angle in range(-self.rotation_range, self.rotation_range, 5):
                    rotate_center   = (template.shape[1] // 2, template.shape[0] // 2)
                    rot_mat         = cv2.getRotationMatrix2D(rotate_center, angle, 1)
                    rotate_dst      = cv2.warpAffine(template, rot_mat, (template.shape[1], template.shape[0]))    
                    
                    match_result            = cv2.matchTemplate(search_region_warped, rotate_dst, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc  = cv2.minMaxLoc(match_result)
                    
                    if max_val > opt_score:
                        opt_score = max_val
                        opt_loc   = max_loc

                x_opt, y_opt = opt_loc
                x_opt = search_x_min + x_opt + template.shape[1] / 2 - self.shifts[1]
                y_opt = search_y_min + y_opt + template.shape[0] / 2 - self.shifts[0]
                self.dt_grid[i, j] = np.array([x_opt, y_opt])
