from dataclasses import dataclass, field
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np

from utility.image_utility import skeletonize_img
from skimage.transform import hough_line, hough_line_peaks
from src.img_enhance import SingleShotEnhancer


def sort_lines(lines: np.ndarray) -> np.ndarray:
    return lines[np.argsort(lines[:, 1])]


@dataclass
class T0GridStruct:
    image     : np.ndarray
    num_lines : Tuple[int, int] 
    
    # Typically two of follows: 
    # [vertical lines, horizontal lines, positive sloped line, negatively sloped line]
    slope_thresh: Tuple[int, int]  
    avg_image : np.ndarray = None

    threshold   : float = 0.2
    density     : int = 10
    temp_scale  : float = 0.67

    grid        : np.ndarray = field(init=False)
    test_angles : np.ndarray = field(init=False)
    image_skel  : np.ndarray = field(init=False)
    template    : np.ndarray = field(init=False)
    params      : np.ndarray = field(init=False)
    uncertainty : np.ndarray = field(init=False)


    def __post_init__(self):
        self.grid        = np.empty(self.num_lines, dtype=object)
        self.template    = np.empty(self.num_lines, dtype=object)
        self.params      = np.empty(self.num_lines, dtype=object)
        self.uncertainty = np.empty(self.num_lines, dtype=object)

        self.lines_a     = np.zeros((self.num_lines[0], 2), dtype=float)
        self.lines_b     = np.zeros((self.num_lines[1], 2), dtype=float)

        self.test_angles = np.linspace(
            -np.pi / 2, np.pi / 2, 
            self.density * 360, 
            endpoint=True
            )

        if self.avg_image is not None:
            enhancer_source = SingleShotEnhancer(avg_shot=self.avg_image, single_shot=self.image)
            self.image = enhancer_source.filter()

        _, self.image_skel = skeletonize_img(self.image)

        self._populate_grid()
        # self._generate_template(scale=self.temp_scale)
    
    # def solve(self):
    #     _, self.image_skel = skeletonize_img(self.image)
    #     self._populate_grid()


    def _is_within_bounds(self, x: int, y: int) -> bool:
        height, width = self.image.shape[:2]
        if 0 <= x <= width and 0 <= y <= height:
            return True

        return False


    @staticmethod
    def _find_intersection(line1: np.ndarray, line2: np.ndarray) -> Tuple[float, float]:
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
            return None


    def _populate_grid(self) -> None:
        # Populate grid with intersection pts of group a and group b lines
        a_mat, b_mat = self._hough_line_transform()

        for i, a_line in enumerate(sort_lines(a_mat)):
            if i >= self.num_lines[0]:
                continue
            for j, b_line in enumerate(sort_lines(b_mat)):
                if j >= self.num_lines[1]:
                    continue
                intersection = self._find_intersection(a_line, b_line)
                bounded = self._is_within_bounds(intersection[0], intersection[1])
                if intersection is not None and bounded:
                    self.grid[i, j] = intersection
        
        self.lines_a = a_mat
        self.lines_b = b_mat

        return


    def __line_intersection_check(self, candidate: np.ndarray, mat: np.ndarray, num_lines:int) -> bool:
        """
        Enforce that the lines in the same group do not intersect within the image bounds. 
        """

        for idx in range(num_lines):
            line = mat[idx]
            
            intersection = self._find_intersection(line, candidate)
            if intersection is not None:
                x, y = intersection
                if self._is_within_bounds(x, y):
                    return True
        return False


    def _hough_line_transform(self) -> Tuple[np.ndarray, np.ndarray]:
        # Lines in group a and group b
        lines_a, lines_b = self.num_lines
        max_thresh = np.max(self.slope_thresh)

        a_mat = np.zeros((lines_a, 2), dtype=float)  
        b_mat = np.zeros((lines_b, 2), dtype=float)  

        cur_a, cur_b = 0, 0

        h, theta, d = hough_line(self.image_skel, theta=self.test_angles)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=self.threshold*h.max())):
            slope = np.tan(angle + np.pi / 2) 
            candidate_line = np.array([angle, dist])
            if slope >= max_thresh or slope <= -max_thresh:
                if cur_a < lines_a:
                    if self.__line_intersection_check(candidate_line, a_mat, cur_a):
                        continue
                    a_mat[cur_a] = candidate_line
                    cur_a += 1
            else:
                if cur_b < lines_b:
                    if self.__line_intersection_check(candidate_line, b_mat, cur_b):
                        continue
                    b_mat[cur_b] = candidate_line
                    cur_b += 1

            if cur_a >= lines_a and cur_b >= lines_b:
                break

        return a_mat, b_mat
    

    def _grid_img_bound(self, i: int, j: int) -> tuple:
        if not (0 <= i < self.num_lines[0] and 0 <= j < self.num_lines[1]):
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
            if 0 <= ni < self.num_lines[0] and 0 <= nj < self.num_lines[1] and not np.isnan(self.grid[ni, nj]).any():
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

        for i in range(self.num_lines[0]):
            for j in range(self.num_lines[1]):
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

                if (x_min < 0 or y_min < 0 or x_max > width or y_max > height or length <= 0):
                    # Skip if the crop region is near the image boundary
                    continue

                self.template[i, j] = np.array([x_min, y_min, x_max, y_max])
                self.params[i, j] = np.array([ang1, ang2, length])
        return 
    

    def plot_hough_lines(self):
        """
        Plot a line given Hough parameters (angle, distance), 
        clipped to image boundaries
        """
        
        lines_pos, lines_neg = self.lines_a, self.lines_b
        height, width = self.image.shape[:2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(self.image_skel, cmap='gray')
        ax1.axis('off')

        ax2.imshow(self.image, cmap='gray')
        ax2.axis('off')
        
        def plot_line(angle, dist, color):    
            a = np.cos(angle)
            b = np.sin(angle)
            x0 = a * dist
            y0 = b * dist
        
            points = []
            # Intersection with left boundary (x = 0)
            if b != 0:
                y_left = (dist - 0 * a) / b
                if 0 <= y_left <= height:
                    points.append((0, y_left))
            
            # Intersection with right boundary (x = width)
            if b != 0:
                y_right = (dist - width * a) / b
                if 0 <= y_right <= height:
                    points.append((width, y_right))
            
            # Intersection with top boundary (y = 0)
            if a != 0:
                x_top = (dist - 0 * b) / a
                if 0 <= x_top <= width:
                    points.append((x_top, 0))
            
            # Intersection with bottom boundary (y = height)
            if a != 0:
                x_bottom = (dist - height * b) / a
                if 0 <= x_bottom <= width:
                    points.append((x_bottom, height))
            
            if len(points) == 2:
                x1, y1 = points[0]
                x2, y2 = points[1]
                ax2.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
            else:
                x1 = x0 + 1000 * (-b)
                y1 = y0 + 1000 * (a)
                x2 = x0 - 1000 * (-b)
                y2 = y0 - 1000 * (a)
                ax2.plot([x1, x2], [y1, y2], color=color, linewidth=2, alpha=0.7)
        
        for angle, dist in lines_pos:
            plot_line(angle, dist, 'red')
        
        for angle, dist in lines_neg:
            plot_line(angle, dist, 'blue')
        
        ax2.plot([], [], 'red', label=f'Group 1:{len(lines_pos)}')
        ax2.plot([], [], 'blue', label=f'Group 2: {len(lines_neg)}')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()