from utility.py_import import np, cv2, dataclass, field, Tuple
from utility.image_utility import skeletonize_img, stereo_transform
from skimage.transform import hough_line, hough_line_peaks


def sort_lines(lines: np.ndarray) -> np.ndarray:
    return lines[np.argsort(lines[:, 1])]


@dataclass
class T0GridStruct:
    shape       : Tuple[int, int]
    image_path  : str
    num_lines   : int
    solve_uncert: bool 

    threshold   : float = 0.2
    density     : int = 10
    temp_scale  : float = 0.67

    grid        : np.ndarray = field(init=False)
    test_angles : np.ndarray = field(init=False)
    image       : np.ndarray = field(init=False)
    image_skel  : np.ndarray = field(init=False)
    template    : np.ndarray = field(init=False)
    params      : np.ndarray = field(init=False)
    uncertainty : np.ndarray = field(init=False)


    def __post_init__(self):
        self.grid        = np.empty(self.shape, dtype=object)
        self.template    = np.empty(self.shape, dtype=object)
        self.params      = np.empty(self.shape, dtype=object)
        self.uncertainty = np.empty(self.shape, dtype=object)
        self.test_angles = np.linspace(
            -np.pi / 2, np.pi / 2, 
            self.density * 360, 
            endpoint=True
            )
        
        self.image       = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        # self.image     = stereo_transform(self.image)

        _, self.image_skel  = skeletonize_img(self.image)

        ############################
        ### Initialize the grids ###
        ############################
        self._populate_grid()
        self._generate_template(scale=self.temp_scale)
    

    def _is_within_bounds(self, x: int, y: int) -> bool:
        height, width = np.shape(self.image)
        return 0 <= x < width and 0 <= y < height


    @staticmethod
    def _find_intersection(line1: np.ndarray, line2: np.ndarray) -> Tuple[float, float]:
        theta1, rho1 = line1
        theta2, rho2 = line2

        A = np.array([
            # Line equations in Cartesian form: x * cos(theta) + y * sin(theta) = rho
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


    def _populate_grid(self) -> None:
        """
        Populate the grid with the intersection points of positive and negative lines
        if lines are given.
        """
        pos_lines, neg_lines = self._hough_line_transform(slope_thresh=0.1)

        for i, pos_line in enumerate(sort_lines(pos_lines)):
            for j, neg_line in enumerate(sort_lines(neg_lines)):
                intersection = self._find_intersection(pos_line, neg_line)
                if intersection is None or not self._is_within_bounds(intersection[0], 
                                                                      intersection[1]):
                    intersection = (np.nan, np.nan)

                self.grid[i, j]  = intersection

                if self.solve_uncert:
                    angular_bin = np.pi / (self.density * 360)
                    sigma_params = np.diag([
                        1,                  # 1 pixel resolution for scipy hough
                        (angular_bin) ** 2,
                        1,              
                        (angular_bin) ** 2
                        ])
                    J = self.__jacobian(pos_line, neg_line)

                    # Chi Square 95 % with 2 degree of freedom --> 5.991
                    sigma_xy = J @ sigma_params @ J.T
                    self.uncertainty[i, j] = np.sqrt(
                        np.array([sigma_xy[0, 0], sigma_xy[1,1]])
                        ) * np.sqrt(5.991)
        print("Grid populated with intersections of lines.")


    def _hough_line_transform(self, slope_thresh:float=0.1):
        lines_pos = np.empty((0, 2), dtype=float)  
        lines_neg = np.empty((0, 2), dtype=float)  

        h, theta, d = hough_line(self.image_skel, theta=self.test_angles)
        for _, angle, dist in zip(*hough_line_peaks(
                h, theta, d, 
                threshold=self.threshold*h.max(),
                num_peaks=np.sum(self.shape)
                )):

            slope = np.tan(angle + np.pi / 2) 

            if abs(slope) > slope_thresh: # Assume no horizontal lines
                if slope >= 0:
                    lines_pos = np.vstack((lines_pos, [angle, dist]))
                else:
                    lines_neg = np.vstack((lines_neg, [angle, dist]))
        return lines_pos, lines_neg
    

    def _grid_img_bound(self, i: int, j: int) -> tuple:
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            raise IndexError("Center index (i, j) is out of bounds.")

        x_c, y_c = self.grid[i, j]
        dx_max1, dy_max1 = 0, 0  # First dominant direction (e.g., positive slope)
        dx_max2, dy_max2 = 0, 0  # Second dominant direction (e.g., negative slope)
        max_dist1, max_dist2 = 0, 0

        directions = [
            (i, j + 1),   # Right
            (i + 1, j),   # Bottom
            (i, j - 1),   # Left
            (i - 1, j)    # Top
            ]

        for ni, nj in directions:
            if 0 <= ni < self.shape[0] and 0 <= nj < self.shape[1] and not np.isnan(self.grid[ni, nj]).any():
                x_adj, y_adj = self.grid[ni, nj]
                dx = x_adj - x_c
                dy = y_adj - y_c
                dist = np.hypot(dx, dy)

                if dx * dy >= 0:  # Positive slope (e.g., ↗ or ↙)
                    if dist > max_dist1:
                        max_dist1 = dist
                        dx_max1, dy_max1 = dx, dy
                else:  # Negative slope (e.g., ↘ or ↖)
                    if dist > max_dist2:
                        max_dist2 = dist
                        dx_max2, dy_max2 = dx, dy


        # Y and X distance are flipped in this implementation
        angle1 = np.arctan2(dx_max1, dy_max1) if max_dist1 > 0 else 0.0
        angle2 = np.arctan2(dx_max2, dy_max2) if max_dist2 > 0 else 0.0
        
        # Ensure angles are acute (0 ≤ angle ≤ π/2)
        angle1 = min(abs(angle1), np.pi - abs(angle1)) 
        angle2 = min(abs(angle2), np.pi - abs(angle2)) 
        lengths = [max_dist1, max_dist2]

        # Bounding box half-sizes (max of absolute values)
        half_width = min(abs(dx_max1), abs(dx_max2))
        half_height = min(abs(dy_max1), abs(dy_max2))
        half_sizes = np.array([half_width, half_height])
        return (half_sizes, [angle1, angle2], min(lengths))
    

    def _generate_template(self, scale: float=0.7):
        height, width = np.shape(self.image)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # Crop based on bottom right node
                center = self.grid[i, j]
                if (np.isnan(center).any()):
                    continue

                half_sizes, angles, length = self._grid_img_bound(i, j)
                
                x_half, y_half     = half_sizes
                ang1, ang2         = angles 
                rect_half_width    = scale * x_half
                rect_half_height   = scale * y_half
                x_center, y_center = center
                
                # Ensure the coordinates are within the image boundaries
                x_min = int(x_center - rect_half_width)
                y_min = int(y_center - rect_half_height)
                x_max = int(x_center + rect_half_width)
                y_max = int(y_center + rect_half_height)

                if (x_min < 0 or y_min < 0 or x_max > width or y_max > height):
                    # Skip if the crop region is near the image boundary
                    continue

                self.template[i, j] = np.array([x_min, y_min, x_max, y_max])
                self.params[i, j] = np.array([ang1, ang2, length])

        print("Templates generated for the grid intersections.")
        return None
    

    def __jacobian(self, line1: np.ndarray, line2: np.ndarray) -> np.ndarray:
        theta1, rho1 = line1
        theta2, rho2 = line2
        delta_theta = theta2 - theta1
        sin_dtheta = np.sin(delta_theta)
        cos_dtheta = np.cos(delta_theta)

        if np.abs(sin_dtheta) < 1e-10:
            raise ValueError("Lines are nearly parallel (sin(delta_theta) ≈ 0)")
        
        # Precompute terms for x and y derivatives
        x_num = rho1 * np.sin(theta2) - rho2 * np.sin(theta1)
        y_num = rho2 * np.cos(theta1) - rho1 * np.cos(theta2)
        
        # Partial derivatives for x
        dx_drho1 = np.sin(theta2) / sin_dtheta
        dx_drho2 = -np.sin(theta1) / sin_dtheta
        dx_dtheta1 = (-rho2 * np.cos(theta1) * sin_dtheta + x_num * cos_dtheta) / (sin_dtheta ** 2)
        dx_dtheta2 = (rho1 * np.cos(theta2) * sin_dtheta - x_num * cos_dtheta) / (sin_dtheta ** 2)
        
        # Partial derivatives for y
        dy_drho1 = -np.cos(theta2) / sin_dtheta
        dy_drho2 = np.cos(theta1) / sin_dtheta
        dy_dtheta1 = (-rho2 * np.sin(theta1) * sin_dtheta + y_num * cos_dtheta) / (sin_dtheta ** 2)
        dy_dtheta2 = (rho1 * np.sin(theta2) * sin_dtheta - y_num * cos_dtheta) / (sin_dtheta ** 2)
        
        return np.array([
            [dx_drho1, dx_dtheta1, dx_drho2, dx_dtheta2],
            [dy_drho1, dy_dtheta1, dy_drho2, dy_dtheta2]
        ])
    
