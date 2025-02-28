import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from skimage.transform import hough_line, hough_line_peaks
from src.grid_struct import GridStruct
from src.image_utility import skeletonize_img
from scipy.spatial.distance import cdist
 

def gauss_smooth(points, dx, dy, method="median", alpha=0.2):
    """
    Smooths the dx, dy values using a weighted Gaussian kernel.

    Parameters:
        points (ndarray): Nx2 array of (x, y) positions.
        dx (ndarray): Nx1 array of dx values.
        dy (ndarray): Nx1 array of dy values.
        method (str): Smoothing method ("mean" or "gaussian").
        alpha (float): Standard deviation of the Gaussian function.

    Returns:
        smoothed_dx (ndarray): Smoothed dx values.
        smoothed_dy (ndarray): Smoothed dy values.
    """
    
    distances = cdist(points, points, metric='euclidean')
    
    if method == "mean":
        sigma = alpha * np.mean(distances)
    elif method == "median":
        sigma = alpha * np.median(distances)
    elif method == "max":
        sigma = alpha * np.max(distances)
    else:
        raise ValueError("Invalid method for sigma selection. Use 'mean', 'median', or 'max'.")

    weights = np.exp(- (distances**2) / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)
    return np.dot(weights, dx), np.dot(weights, dy)


class HoughTM:
    def __init__(self, path_ref, path_mov, num_lines, ang_density=10, threshold=0.2,
                 temp_scale=0.67, window_scale=1.2, search_scale=2, down_scale=4):
        """
        Default Constructor

        :param path_ref        : String :   Reference image 
        :param path_mov        : String :   Moving image
        :param num_lines       : int    :   Number of lines
        :param ang_density     : int    :   Angle density for Hough Line
        :param threshold       : float  :   Threshold for line detection
        :param temp_scale      : float  :   Scale of the template
        :param window_scale    : float  :   Scale of window such that template is located within
        :param search_scale    : float  :   Scale of the search region
        :param down_scale      : int    :   Down scale size
        """

        self.t0_im            = cv2.imread(path_ref, cv2.IMREAD_GRAYSCALE)
        self.dt_im            = cv2.imread(path_mov, cv2.IMREAD_GRAYSCALE)
        _, self.t0_im_skel    = skeletonize_img(image=self.t0_im)
        _, self.dt_im_skel    = skeletonize_img(image=self.dt_im)
        self.num_lines        = num_lines
        self.threshold        = threshold

        assert np.shape(self.t0_im) == np.shape(self.dt_im), "Shape of images does not match."
        self.im_shape         = self.t0_im.shape[:2]
        self.test_angles      = np.linspace(-np.pi / 2, np.pi / 2, ang_density * 360, endpoint=True)
        lines_pos, lines_neg  = self._hough_line_transform(slope_thresh=0.1)

        # Grid structures for intersections 
        self.grid_struct = GridStruct(pos_lines=lines_pos, neg_lines=lines_neg, ref_im=self.t0_im_skel, 
                                      mov_im=self.dt_im_skel, temp_scale=temp_scale, window_scale=window_scale, 
                                      search_scale=search_scale, down_scale=down_scale)
        self.disp_field  = np.empty(self.grid_struct.shape, dtype=object)
        self.solve_bool  = False

    
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
        
        lines_pos = np.empty((0, 2), dtype=float)  
        lines_neg = np.empty((0, 2), dtype=float)  

        h, theta, d = hough_line(self.t0_im_skel, theta=self.test_angles)
        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=self.threshold*h.max(),
                                                    num_peaks=self.num_lines * 2)):
            slope = np.tan(angle + np.pi / 2) 

            # Assume no horizontal lines
            if abs(slope) > slope_thresh:
                if slope >= 0:
                    lines_pos = np.vstack((lines_pos, [angle, dist]))
                else:
                    lines_neg = np.vstack((lines_neg, [angle, dist]))
        return lines_pos, lines_neg    
    

    def solve(self, sigma=7):
        """
        Solve the correspondence between t0 image and dt img to obtainn the change in 
        displacement field
        """
        rows, cols = self.grid_struct.shape
        self.disp_field = np.full((rows, cols, 4), np.nan)  # (x, y, dx, dy)
        valid_points = []
        
        for i in range(rows):
            for j in range(cols):
                if self.grid_struct.t0_grid[i, j] is not None and self.grid_struct.dt_grid[i, j] is not None:
                    x0, y0 = self.grid_struct.t0_grid[i, j]
                    x1, y1 = self.grid_struct.dt_grid[i, j]

                    dx, dy = (x1 - x0), (y1 - y0)
                    self.disp_field[i, j] = [x0, y0, dx, dy]
                    valid_points.append([x0, y0, dx, dy])

        valid_points = np.array(valid_points)

        points  = valid_points[:, :2]
        dx_vals = valid_points[:, 2]  
        dy_vals = valid_points[:, 3]  

        dx_s, dy_s = gauss_smooth(points, dx_vals, dy_vals, method="mean", alpha=0.1)

        for i, (x, y, dx_new, dy_new) in enumerate(zip(points[:, 0], points[:, 1], dx_s, dy_s)):
            mask = (self.disp_field[..., 0] == x) & (self.disp_field[..., 1] == y)
            self.disp_field[mask, 2] = dx_new
            self.disp_field[mask, 3] = dy_new

        self.solve_bool = True


    def get_velocity(self, dt=1):
        """
        Returns velocity field computed as displacement / dt.
        
        :param dt   :   float   :   Time step (default=1)
        :return     :               Velocity field (x, y, vx, vy) where vx = dx/dt, vy = dy/dt
        """

        if not self.solve_bool:
            raise ValueError("solve() must be called before get_velocity().")

        vel_field = self.disp_field.copy()
        vel_field[..., 2:] /= dt
        return vel_field


    def get_vorticity(self, dt=1):
        """
        Computes vorticity ω_z = dv/dx - du/dy and returns (x, y, vorticity).
        
        :return: Tuple of (x, y, vorticity) where x, y are grid coordinates.
        """
        if not self.solve_bool:
            raise ValueError("Call solve() before computing vorticity.")
        
        rows, cols      = self.grid_struct.shape
        vort_field      = np.full((rows, cols, 3), np.nan)  # (x, y, w)
        vel_field       = self.get_velocity(dt=dt)

        x, y    = vel_field[..., 0], vel_field[..., 1]
        vx, vy  = vel_field[..., 2], vel_field[..., 3]
        vort_field[..., 0] = x
        vort_field[..., 1] = y

        #Todo, not uniform spacing
        # Compute spatial grid spacing (assumes uniform spacing)
        dx      = np.gradient(x, axis=1)  
        dy      = np.gradient(y, axis=0)
        dx[dx == 0], dy[dy == 0] = 1e-8, 1e-8  

        du_dy   = np.gradient(vx, axis=0) / dy  
        dv_dx   = np.gradient(vy, axis=1) / dx 
        ω_z     = dv_dx - du_dy

        vort_field[..., 2] = ω_z
        return vort_field 


    def plot_intersections(self):
        """
        Plots the intersection points from self.t0_grid and self.dt_grid as two different subfigures.
        Each subfigure overlays the points onto the reference image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Flatten the grids and filter out None values
        t0_points = np.array([p for row in self.grid_struct.t0_grid for p in row if p is not None])
        dt_points = np.array([p for row in self.grid_struct.dt_grid for p in row if p is not None])

        # Plot t0_grid intersections
        axes[0].imshow(self.t0_im, cmap='gray')  # Display reference image
        if t0_points.size > 0:
            axes[0].scatter(t0_points[:, 0], t0_points[:, 1], color='blue', marker='o', label='t0 Grid Points')
        axes[0].set_title("t0 Grid")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].grid(False)

        # Plot dt_grid intersections
        axes[1].imshow(self.dt_im, cmap='gray')  # Display reference image
        if dt_points.size > 0:
            axes[1].scatter(dt_points[:, 0], dt_points[:, 1], color='red', marker='x', label='dt Grid Points')
        axes[1].set_title("dt Grid")
        axes[1].set_xlabel("X")
        axes[1].set_ylabel("Y")
        axes[1].grid(False)

        plt.tight_layout()
        plt.show()


    def plot_fields(self, dt=1):
        """
        Plots velocity field, displacement magnitude, and vorticity field on the same figure.

        :param dt: Time step factor (default=1)
        """
        if not self.solve_bool:
            raise ValueError("Call solve() before plotting fields.")

        # Compute velocity and vorticity
        vorticity = self.get_vorticity(dt)
        vel_field = self.get_velocity(dt)
        
        x, y = vel_field[..., 0], vel_field[..., 1]
        vx, vy = vel_field[..., 2], vel_field[..., 3]
        dx, dy = self.disp_field[..., 2], self.disp_field[..., 3]
        vort = vorticity[..., 2]

        # Compute magnitudes
        disp_mag = np.sqrt(dx**2 + dy**2)
        vel_mag = np.sqrt(vx**2 + vy**2)

        # Normalize vectors for quiver plots (avoid division by zero)
        unit_dx = np.divide(dx, disp_mag, where=disp_mag != 0, out=np.zeros_like(dx))
        unit_dy = np.divide(dy, disp_mag, where=disp_mag != 0, out=np.zeros_like(dy))
        unit_vx = np.divide(vx, vel_mag, where=vel_mag != 0, out=np.zeros_like(vx))
        unit_vy = np.divide(vy, vel_mag, where=vel_mag != 0, out=np.zeros_like(vy))

        # Modify `valid_mask` to capture all valid data
        valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vort)

        # Debugging: Check how many valid points exist
        valid_count = np.sum(valid_mask)
        print(f"Valid points count: {valid_count}")
        
        if valid_count == 0:
            raise ValueError("All data points are invalid (NaN or Inf). Check the input data.")

        # Triangulation for contour plots
        triang = tri.Triangulation(x[valid_mask], y[valid_mask])

        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Displacement Magnitude
        disp_plot = axs[0].tricontourf(triang, disp_mag[valid_mask], cmap='viridis', levels=100)
        axs[0].quiver(x[valid_mask], y[valid_mask], unit_dx[valid_mask], unit_dy[valid_mask],
                    angles='xy', scale_units='xy', scale=0.1, color='black')
        fig.colorbar(disp_plot, ax=axs[0], label="Displacement Magnitude")
        axs[0].set_title("Displacement Magnitude")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].axis('equal')

        # Plot 2: Velocity Field
        vel_plot = axs[1].tricontourf(triang, vel_mag[valid_mask], cmap='viridis', levels=100)
        axs[1].quiver(x[valid_mask], y[valid_mask], unit_vx[valid_mask], unit_vy[valid_mask],
                    angles='xy', scale_units='xy', scale=0.1, color='blue')
        fig.colorbar(vel_plot, ax=axs[1], label="Velocity Magnitude")
        axs[1].set_title("Velocity Field")
        axs[1].set_xlabel("X")
        axs[1].set_ylabel("Y")
        axs[1].axis('equal')

        # Plot 3: Vorticity Field
        vort_plot = axs[2].tricontourf(triang, vort[valid_mask], cmap='coolwarm', levels=100)
        fig.colorbar(vort_plot, ax=axs[2], label="Vorticity")
        axs[2].set_title("Vorticity Field")
        axs[2].set_xlabel("X")
        axs[2].set_ylabel("Y")
        axs[2].axis('equal')

        # Display plots
        plt.tight_layout()
        plt.show()