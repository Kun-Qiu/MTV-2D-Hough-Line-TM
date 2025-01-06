import cv2
import numpy as np


class GridStruct:
    def __init__(self, pos_lines, neg_lines, img, img2, temp_scale=0.7, window_scale=1.1, search_scale=1.5):
        """
        Default constructor

        :param pos_lines    :   [rho, theta] of positively sloped lines  
        :param neg_lines    :   [rho, theta] of negatively sloped lines
        :param img          :   Input image (Before Transformation)
        :param img2         :   Input image (After Transformation)
        :param temp_scale   :   kjk
        :param window_scale :
        :param search_scale :   
        """

        def sort_lines(lines):
            """
            Sorted by the rho value
            """
            return lines[np.argsort(lines[:, 1])]
 
        self.shape = (len(pos_lines), len(neg_lines))   # Shape = (11, 11, 2) 
        self.img = img
        self.img2 = img2

        self.grid = np.empty(self.shape, dtype=object)
        self.template = np.empty(self.shape, dtype=object)
        self.search_patch = np.empty(self.shape, dtype=object)
        self.num_intersections = 0

        ### Immediately initialize and populate the data structure ###
        self.populate_grid(sort_lines(pos_lines), sort_lines(neg_lines))
        self.generate_template(scale=temp_scale)
        self.generate_search_patch(window_scale=window_scale,
                                   search_scale=search_scale)


    def _is_within_bounds(self, x, y):
        """
        Check if a point (x, y) lies within the image boundaries.

        :param x: x-coordinate
        :param y: y-coordinate

        :return: True if within bounds, False otherwise
        """
        height, width = np.shape(self.img)
        return 0 <= x < width and 0 <= y < height
    

    def find_intersection(self, line1, line2):
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


    def populate_grid(self, pos_lines, neg_lines):
        """
        Populate the grid with the intersection points of positive and negative lines.

        :param pos_lines    :   
        :param neg_lines    :
        :return             :   N/A
        """
        for i, pos_line in enumerate(pos_lines):
            for j, neg_line in enumerate(neg_lines):
                intersection = self.find_intersection(pos_line, neg_line)
                if intersection is not None:
                    x, y = intersection
                    if not self._is_within_bounds(x, y):
                        intersection = (np.nan, np.nan)
                else:
                    intersection = (np.nan, np.nan)
                
                self.grid[i, j] = intersection
                self.num_intersections += 1
    

    def generate_template(self, scale=0.7):
        """
        Create template patches using the grid intersections and the search patch for
        the consecutive frame.

        :param image    :   The image to crop, as a NumPy array.
        :param scale    :   The width of crop scale --> scale=1: from intersection to intersection
        """
        height, width = np.shape(self.img)

        for i in range(self.shape[0] - 1):
            for j in range(self.shape[1] - 1):
                
                # Crop based on bottom right node
                center = self.grid[i, j]
                bottom_right = self.grid[i, j + 1]

                # If any corner is NaN, skip cropping this region
                if (
                    np.isnan(center).any() or
                    np.isnan(bottom_right).any()
                ):
                    continue

                rect_half_width     = scale * abs(bottom_right[0] - center[0])
                rect_half_height    = scale * abs(bottom_right[1] - center[1])
                x_center, y_center  = center
                
                # Ensure the coordinates are within the image boundaries
                x_min = max(0, int(x_center - rect_half_width))
                x_max = min(width, int(x_center + rect_half_width))
                y_min = max(0, int(y_center - rect_half_height))
                y_max = min(height, int(y_center + rect_half_height))

                self.template[i, j] = np.array([x_min, y_min, x_max, y_max])


    def generate_search_patch(self, window_scale=1.2, search_scale=3):
        """
        Create search patches for the template matching algorithm by maximizing
        similarity between self.img and self.img2

        :param window_scale :   Window scaling constant
        :param search_scale :   Search region scaling constant
        """

        assert window_scale >= 1, "window_scale must be greater than or equal to 1"

        for i in range(self.shape[0] - 1):
            for j in range(self.shape[1] - 1):
                
                center = self.grid[i, j]
                if np.isnan(center).any() or np.any((self.template[i, j]) == None):
                    continue

                x_min, y_min, x_max, y_max  = self.template[i, j]
                rect_width                  = abs((x_max - x_min))
                rect_height                 = abs((y_max - y_min))
                x_center, y_center          = center
                bound_x, bound_y      = np.shape(self.img)

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
                
                template = self.img[temp_y_min:temp_y_max, temp_x_min:temp_x_max]
                search_region = self.img2[search_y_min:search_y_max, search_x_min:search_x_max]
                
                match_result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
                
                best_score = -float('inf')
                best_center = None

                for y in range(match_result.shape[0]):
                    for x in range(match_result.shape[1]):
                        # Iterate over the match result to find the best match

                        score = match_result[y, x]
                        candidate_center = (search_x_min + x + template.shape[1] // 2,
                                            search_y_min + y + template.shape[0] // 2)

                        if score > best_score:
                            best_score  = score
                            best_center = candidate_center

                # Store the best matching center and the region bounds
                if best_center:
                    self.search_patch[i, j] = np.array([
                        best_center[0] - template.shape[1] // 2,
                        best_center[1] - template.shape[0] // 2,
                        best_center[0] + template.shape[1] // 2,
                        best_center[1] + template.shape[0] // 2
                    ])


    def get_template(self, i, j):
        """
        Get the template img corresponding with the index i, j

        :param i, j :   Index of the template 
        """
        x_min, y_min, x_max, y_max = self.template[i, j]
        
        return x_min, y_min, self.img[int(y_min):int(y_max), int(x_min):int(x_max)]
    
    def get_search(self, i, j):
        """
        Get the search region img corresponding with the index i, j

        :param i, j :   Index of the template 
        """
        x_min, y_min, x_max, y_max = self.search_patch[i, j]
        
        return x_min, y_min, self.img2[int(y_min):int(y_max), int(x_min):int(x_max)]


