import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from skimage.transform import hough_line, hough_line_peaks
from grid_struct import GridStruct
from image_utility import skeletonize_img


class HoughTM:
    def __init__(self, path_ref, path_mov, num_lines, ang_density=10, threshold=0.2):
        """
        Default Constructor

        :param path_ref        : String : Reference image (Before Transformation)  
        :param path_mov        : String : Moving image (After Transformation)
        :param num_lines       : int    : Number of lines (One sided only)
        :param ang_density     : int    : Angle density for Hough Line
        :param threshold       : float  : Threshold for line detection
        """

        self.t0_im            = cv2.imread(path_ref, cv2.IMREAD_GRAYSCALE)
        self.dt_im            = cv2.imread(path_mov, cv2.IMREAD_GRAYSCALE)
        _, self.t0_im_skel    = skeletonize_img(image=self.t0_im)
        _, self.dt_im_skel    = skeletonize_img(image=self.dt_im)
        self.num_lines        = num_lines
        self.threshold        = threshold
        self.grid_struct      = None
        self.solve_bool       = False

        assert np.shape(self.t0_im) == np.shape(self.dt_im), "Shape of images does not match."
        self.im_shape         = self.t0_im.shape[:2]
        self.test_angles      = np.linspace(-np.pi / 2, np.pi / 2, ang_density * 360, endpoint=True)

        self.lines_pos_arr, self.lines_neg_arr = self._hough_line_transform(slope_thresh=0.1)
        self.dt_grid          = np.empty((len(self.lines_pos_arr), len(self.lines_neg_arr)), dtype=object)    

    
    def _hough_line_transform(self, slope_thresh=0.1):
        """
        Perform Hough Line Transform to detect lines in a skeletonized image.
        This function applies the Hough Line Transform to the skeletonized image 
        (`self.t0_im_skel`) using a set of test angles (`self.test_angles`).

        :param slope_thresh : float         : The threshold for the slope to consider a 
                                              line as non-horizontal. 
        :return lines_arr   : numpy.ndarray : An array of detected lines with positive slopes. 
                                              Each row contains the angle and distance of a line.
        """
        
        lines_pos_arr = np.empty((0, 2), dtype=float)  
        lines_neg_arr = np.empty((0, 2), dtype=float)  

        h, theta, d = hough_line(self.t0_im_skel, theta=self.test_angles)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=self.threshold*h.max(),
                                                    num_peaks=self.num_lines * 2)):
            slope = np.tan(angle + np.pi / 2) 

            # Assume no horizontal lines
            if abs(slope) > slope_thresh:
                if slope >= 0:
                    lines_pos_arr = np.vstack((lines_pos_arr, [angle, dist]))
                else:
                    lines_neg_arr = np.vstack((lines_neg_arr, [angle, dist]))
        return lines_pos_arr, lines_neg_arr


    def solve(self):
        """
        Solves the grid structure by performing template matching on each grid cell.
        This method initializes the grid structure with the given parameters and iterates 
        through each cell in the grid. For each cell, if both the template and search patch 
        are not None, it performs template matching using OpenCV's `cv2.matchTemplate` 
        method. The method calculates the top-left corner of the matched region and 
        determines the center coordinates. These coordinates are then stored in the 
        `dt_grid` attribute.

        Attributes:
            grid_struct (GridStruct): The grid structure initialized with the given parameters.
            dt_grid (np.ndarray): Array to store the determined coordinates for each grid cell.
            solve_bool (bool): Boolean flag indicating whether the solve method has been executed.
        Raises:
            ValueError: If the template or search patch is None for any grid cell.
        """

        self.grid_struct = GridStruct(self.lines_pos_arr, self.lines_neg_arr, 
                                      self.t0_im_skel, self.dt_im_skel,
                                      temp_scale=0.7, window_scale=1.2, search_scale=2,
                                      down_scale=4)
        
        for i in range(self.grid_struct.shape[0]):
            for j in range(self.grid_struct.shape[1]):
                if self.grid_struct.template[i, j] is not None and self.grid_struct.search_patch[i, j] is not None:
                    _ , _ , template               = self.grid_struct.get_template(i, j)
                    x_min, y_min, search_region    = self.grid_struct.get_search(i, j)

                    w, h    = template.shape[::-1]
                    method  = cv2.TM_CCORR_NORMED

                    pad_top     = max((template.shape[0] - search_region.shape[0]) // 2, 0)
                    pad_bottom  = max(template.shape[0] - search_region.shape[0] - pad_top, 0)
                    pad_left    = max((template.shape[1] - search_region.shape[1]) // 2, 0)
                    pad_right   = max(template.shape[1] - search_region.shape[1] - pad_left, 0)

                    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
                        search_region = cv2.copyMakeBorder(
                        search_region, pad_top, pad_bottom, pad_left, pad_right, 
                        cv2.BORDER_REPLICATE
                    )
                    res                 = cv2.matchTemplate(search_region, template, method)
                    _, _, _, top_left   = cv2.minMaxLoc(res)

                    # Determine the top-left corner
                    center_x = top_left[0] + w // 2
                    center_y = top_left[1] + h // 2

                    absolute_x = x_min + center_x
                    absolute_y = y_min + center_y

                    self.dt_grid[i, j] = np.array([absolute_x, absolute_y])
        self.solve_bool = True
    
    def plot_velocity_field(self):
        if not self.solve_bool:
            raise ValueError("Solve the transformation first using solve().")

        rows, cols = self.dt_grid.shape

        x_coords = np.full((rows, cols), np.nan)
        y_coords = np.full((rows, cols), np.nan)
        dx = np.full((rows, cols), np.nan)
        dy = np.full((rows, cols), np.nan)
        magnitude = np.full((rows, cols), np.nan)

        for i in range(rows):
            for j in range(cols):
                if self.grid_struct.grid[i, j] is not None and self.dt_grid[i, j] is not None:
                    x, y = self.grid_struct.grid[i, j]
                    x_coords[i, j], y_coords[i, j] = x, y
                    
                    # Compute displacement
                    dx[i, j] = self.dt_grid[i, j][0] - x
                    dy[i, j] = self.dt_grid[i, j][1] - y
                    magnitude[i, j] = np.sqrt(dx[i, j]**2 + dy[i, j]**2)

        valid_mask       = ~np.isnan(x_coords) & ~np.isnan(y_coords) & ~np.isnan(magnitude)
        
        x_valid         = x_coords[valid_mask]
        y_valid         = y_coords[valid_mask]
        dx_valid        = dx[valid_mask]
        dy_valid        = dy[valid_mask]
        magnitude_valid = magnitude[valid_mask]

        # Normalize displacement to unit vectors
        unit_dx = dx_valid / magnitude_valid
        unit_dy = dy_valid / magnitude_valid
        triang  = tri.Triangulation(x_valid, y_valid)

        # Create a heatmap and overlay unit vectors
        plt.figure(figsize=(8, 8))
        plt.tricontourf(triang, magnitude_valid, cmap='viridis', levels=100)
        plt.colorbar(label="Displacement Magnitude")

        # Plot unit displacement vectors
        plt.quiver(x_valid, y_valid, unit_dx, unit_dy, angles='xy', scale_units='xy', scale=0.1, color='blue')
        plt.title("Displacement Field with Unit Vectors")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.axis('equal')
        plt.show()
