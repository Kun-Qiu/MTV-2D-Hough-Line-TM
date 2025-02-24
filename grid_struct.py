"""
Main data structure for grid detection 
"""

__author__ = "Kun Qiu"
__credits__ = ["Kun Qiu"]
__version__ = "1.0"
__maintainer__ = "Kun Qiu"
__email__ = "qiukun1234@gmail.com"
__status__ = "Production"

import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_utility import transform_image
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale 
from matplotlib import cm

class GridStruct:
    def __init__(self, pos_lines, neg_lines, ref_im, mov_im, temp_scale=0.7, 
                 window_scale=1.1, search_scale=1.5, down_scale=4):
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

        # Obtain the coarse global linear shift in image
        self.shifts, _, _ = phase_cross_correlation(
            rescale(self.reference_img, 1 / down_scale, anti_aliasing=True), 
            rescale(self.moving_img, 1 / down_scale, anti_aliasing=True), 
            normalization=None)
        self.shifts *= down_scale

        # Public variables
        self.t0_grid        = np.empty(self.shape, dtype=object)
        self.dt_grid        = np.empty(self.shape, dtype=object)
        self.template       = np.empty(self.shape, dtype=object)
        self.search_patch   = np.empty(self.shape, dtype=object)
        self.displace_arr   = np.empty(self.shape, dtype=object)

        ### Immediately initialize and populate the data structure ###
        self._populate_grid(sort_lines(pos_lines), sort_lines(neg_lines))
        self._generate_template(scale=temp_scale)
        self._generate_search_patch(window_scale=window_scale, search_scale=search_scale)


    def _is_within_bounds(self, x, y):
        """
        Check if a point (x, y) lies within the image boundaries.

        :param x: x-coordinate
        :param y: y-coordinate

        :return: True if within bounds, False otherwise
        """
        height, width = np.shape(self.reference_img)
        return 0 <= x < width and 0 <= y < height
        

    def _find_intersection(self, line1, line2):
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


    def _populate_grid(self, pos_lines, neg_lines):
        """
        Populate the grid with the intersection points of positive and negative lines.

        :param pos_lines    :   Lines with postive slope
        :param neg_lines    :   Lines with negative slope
        :return             :   None
        """
        for i, pos_line in enumerate(pos_lines):
            for j, neg_line in enumerate(neg_lines):
                intersection = self._find_intersection(pos_line, neg_line)
                if intersection is not None:
                    x, y = intersection
                    if not self._is_within_bounds(x, y):
                        intersection = (np.nan, np.nan)
                else:
                    intersection = (np.nan, np.nan)
                
                self.t0_grid[i, j] = intersection
                self.num_intersections += 1
    

    def _grid_img_bound(self, i, j):
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
                x_min = max(0, int(x_center - rect_half_width))
                x_max = min(width, int(x_center + rect_half_width))
                y_min = max(0, int(y_center - rect_half_height))
                y_max = min(height, int(y_center + rect_half_height))

                self.template[i, j] = np.array([x_min, y_min, x_max, y_max])


    def _generate_search_patch(self, window_scale=1.1, search_scale=2):
        """
        Create search patches for the template matching algorithm by maximizing
        similarity between self.reference_img and self.moving_img

        :param window_scale :   Window scaling constant
        :param search_scale :   Search region scaling constant
        """
        
        assert window_scale >= 1, "window_scale must be greater than or equal to 1"
        assert search_scale >= 1, "search_scale must be greater than or equal to 1"

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
                warped_search_im        = transform_image(self.moving_img, self.shifts[1], self.shifts[0])
                search_region_warped    = warped_search_im[search_y_min:search_y_max, search_x_min:search_x_max]

                match_result              = cv2.matchTemplate(search_region_warped, template, cv2.TM_CCOEFF_NORMED)
                max_val, _, _, max_loc    = cv2.minMaxLoc(match_result)

                # confidence_threshold = 0.5
                # while max_val < confidence_threshold:
                #     search_x_min, search_y_min, search_x_max, search_y_max  = get_bound(x_center, y_center, 
                #                                                                     rect_width, rect_height, 
                #                                                                     search_scale * 1.1, 
                #                                                                     bound_x, bound_y)
                #     search_region_warped = warped_search_im[search_y_min:search_y_max, search_x_min:search_x_max]
                #     result = cv2.matchTemplate(search_region_warped, search_region_warped, cv2.TM_CCOEFF_NORMED)
                #     max_val, _, max_loc, _ = cv2.minMaxLoc(result)

                best_center = (search_x_min + max_loc[0] + template.shape[1] // 2,
                               search_y_min + max_loc[1] + template.shape[0] // 2)

                self.displace_arr[i, j] = [self.shifts[1], self.shifts[0]]
                warped_x = best_center[0] - self.displace_arr[i,j][0]
                warped_y = best_center[1] - self.displace_arr[i,j][1]
                
                if best_center:
                    self.search_patch[i, j] = np.array([
                        warped_x - template.shape[1] // 2,
                        warped_y - template.shape[0] // 2,
                        warped_x + template.shape[1] // 2,
                        warped_y + template.shape[0] // 2
                    ])

                    self.dt_grid[i, j] = np.array([
                        search_x_min + max_loc[0] + template.shape[1] // 2,
                        search_y_min + max_loc[1] + template.shape[0] // 2
                    ])

                # fig, axes = plt.subplots(2, 3, figsize=(20, 6))
                # ax = axes.ravel()
                # ax[0].imshow(template, cmap=cm.gray)
                # ax[0].set_title('Template')
                # ax[0].axis("off")

                # ax[1].imshow(self.reference_img[search_y_min:search_y_max, 
                #                                 search_x_min:search_x_max], cmap=cm.gray)
                # ax[1].set_title('Search Region')
                # ax[1].axis("off")

                # ax[2].imshow(search_region_warped, cmap=cm.gray)
                # ax[2].scatter(max_loc[0] + template.shape[1] // 2, 
                #               max_loc[1] + template.shape[0] // 2, 
                #               color='red', marker='x', s=100, label='Best Match')
                # ax[2].set_title('Cross Correlation')
                # ax[2].axis("off")

                # ax[3].imshow(self.reference_img, cmap=cm.gray)
                # ax[3].set_title('Source Image')
                # ax[3].plot([temp_x_min, temp_x_max, temp_x_max, temp_x_min, temp_x_min],
                #             [temp_y_min, temp_y_min, temp_y_max, temp_y_max, temp_y_min],
                #             color='red', linewidth=2, label='temp Region')
                # ax[3].axis("off")

                # x_min, y_min, x_max, y_max = self.search_patch[i,j]
                # ax[4].imshow(warped_search_im, cmap=cm.gray)
                # ax[4].set_title('Warped Image')
                # ax[4].plot([x_min, x_max, x_max, x_min, x_min],
                #             [y_min, y_min, y_max, y_max, y_min],
                #             color='red', linewidth=2, label='Search Region')
                # ax[4].axis("off")

                # ax[5].imshow(match_result, cmap='hot')
                # ax[5].set_title('Template Matching')
                # ax[5].axis("off")

                # plt.tight_layout()
                # plt.show()
    

    def get_template(self, i, j):
        """
        Get the template img corresponding with the index i, j

        :param i, j :   Index of the template 
        """
        x_min, y_min, x_max, y_max = self.template[i, j]
        
        return x_min, y_min, self.reference_img[int(y_min):int(y_max), int(x_min):int(x_max)]
    

    def get_search(self, i, j):
        """
        Get the search region img corresponding with the index i, j

        :param i, j :   Index of the template 
        """
        x_min, y_min, x_max, y_max = self.search_patch[i, j]
        
        return x_min, y_min, self.moving_img[int(y_min):int(y_max), int(x_min):int(x_max)]