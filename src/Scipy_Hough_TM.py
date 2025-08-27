from utility.py_import import np, plt, dataclass, field, Tuple
from src.T0_grid_struct import T0GridStruct
from src.dT_grid_struct import DTGridStruct
from cython_build.ParametricX import ParametricX
from src.parametric_opt import ParameterOptimizer
from src.interpolator import dim2Interpolator


@dataclass
class HoughTM:
    path_ref    : str
    path_mov    : str
    num_lines   : Tuple[int, int]
    slope_thresh: Tuple[float, float]
    optimize : bool = False
    verbose  : bool = False

    # Guided Images for Filtering
    path_ref_avg: str = None
    path_mov_avg: str = None

    # Template Matching Optimization Parameters
    fwhm        : float = field(init=False)
    uncertainty : float = field(init=False)
    num_interval: int = field(init=False)
    intensity   : float = field(init=False)
    temp_scale  : float = field(init=False)

    # Hough Line Transform Parameters
    density  : int = field(init=False)
    threshold: float = field(init=False)

    # Lucas Kanade Optical Flow Parameters
    win_size : Tuple[int, int] = field(init=False)
    max_level: int = field(init=False)
    iteration: int = field(init=False)
    epsilon  : float = field(init=False)

    solve_bool  : bool = field(init=False)
    valid_ij    : np.ndarray = field(init=False) 
    grid_T0     : T0GridStruct = field(init=False)
    grid_dT     : DTGridStruct = field(init=False)
    interpolator: dim2Interpolator = field(init=False)

    def __post_init__(self):
        shape = self.num_lines

        # Default values for parameters
        self.set_hough_params(density=10, threshold=0.2)
        self.set_template_params(
            fwhm=3, uncertainty=1, num_interval=30, 
            intensity=0.5, temp_scale=0.67
            )
        
        self.set_optical_flow_params(
            win_size=(62, 62), max_level=3, 
            iteration=10, epsilon=0.001
            )
        
        self.uncertainty = self.fwhm

        self.grid_T0 = T0GridStruct(
            shape, 
            self.path_ref, 
            self.path_ref_avg,
            num_lines=self.num_lines,
            slope_thresh=self.slope_thresh,
            threshold=self.threshold, 
            density=self.density,
            temp_scale=self.temp_scale
            )
        if self.optimize:
            self._template_optimize(self.grid_T0, False)

        self.grid_dT = DTGridStruct(
            self.grid_T0, 
            self.path_mov,
            self.path_mov_avg, 
            win_size=self.win_size,
            max_level=self.max_level,
            iteration=self.iteration,
            epsilon=self.epsilon
            )

        grid_T0_valid = np.array(
            [[cell is not None for cell in row] 
            for row in self.grid_T0.grid]
            )
        grid_dT_valid = np.array(
            [[cell is not None for cell in row] 
            for row in self.grid_dT.grid]
            )
        
        self.valid_ij = np.argwhere(grid_T0_valid & grid_dT_valid)
        lines_a, lines_b = self.num_lines
        self.disp_field = np.full((lines_a, lines_b, 4), np.nan)
        self.solve_bool = False


    def set_template_params(
            self, fwhm: float, uncertainty: float, num_interval: int, 
            intensity: float, temp_scale: float
            ) -> None:
        
        self.fwhm = fwhm
        self.uncertainty = uncertainty
        self.num_interval = num_interval
        self.intensity = intensity
        self.temp_scale = temp_scale
       
        return 
    

    def set_hough_params(
            self, density: int, threshold: float
            ) -> None:

        self.density = density
        self.threshold = threshold

        return
    

    def set_optical_flow_params(
            self, win_size: Tuple[int, int], max_level: int,
            iteration: int, epsilon: float
            ) -> None:

        self.win_size = win_size
        self.max_level = max_level
        self.iteration = iteration
        self.epsilon = epsilon

        return
    

    def _template_optimize(self, grid_obj: np.ndarray, v: bool = False) -> None:
        grid_valid = np.array(
            [[cell is not None for cell in row] 
            for row in grid_obj.grid]
            )
        params_valid = np.array(
            [[cell is not None for cell in row] 
            for row in grid_obj.params]
            )
    
        valid_mask = grid_valid & params_valid
        valid_indices = np.argwhere(valid_mask)

        for i, j in valid_indices:
            x, y  = grid_obj.grid[i, j]
            ang1, ang2, leg_len = grid_obj.params[i, j]

            parametricX_obj = ParametricX(
                center=(x, y), 
                shape=(ang1, ang2, self.intensity, self.fwhm, leg_len),
                image=grid_obj.image
                )
        
            optimizer = ParameterOptimizer(
                parametricX_obj, uncertainty=self.uncertainty, 
                num_interval=self.num_interval, verbose=self.verbose
                )

            # parameter_star = optimizer.quad_optimize()
            parameter_star = optimizer.quad_optimize_gradient()
            if v:
                optimizer.visualize()
            grid_obj.grid[i, j] = parameter_star[0:2]
        return


    def solve(self) -> None:
        for i, j in self.valid_ij:
            x0, y0 = self.grid_T0.grid[i, j]
            x1, y1 = self.grid_dT.grid[i, j]

            dx, dy = (x1 - x0), (y1 - y0)
            self.disp_field[i, j] = [x0, y0, dx, dy]

        self.solve_bool = True
        return
    

    def sequence_solve(self, single_sequence: list, avg_sequence: list) -> None:
        self.grid_dT.sequence_solver(single_sequence, avg_sequence)
        self.solve()


    def get_fields(self, dt:float=1, pix_to_world: float = 1, extrapolate:bool=False) -> np.ndarray:
        if not self.solve_bool:
            raise ValueError("Call solve() before get_fields().")
        
        # Set up interpolator for displacement field
        valid_mask = ~np.isnan(self.disp_field).any(axis=2)
        valid_points = self.disp_field[valid_mask][:, :2] 
        valid_displacements = self.disp_field[valid_mask][:, 2:] 
        
        if valid_points.size > 0:
            self.interpolator = dim2Interpolator(
                xy=valid_points,
                dxy=valid_displacements,
                extrapolate=extrapolate
            )

        # Interpolate the fields
        h, w = self.grid_T0.image.shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        points = np.column_stack([x.ravel(), y.ravel()])
        
        disp = self.interpolator.interpolate(points)
        vel = (disp.copy() / dt).reshape(h, w, 2)
        disp = disp.reshape(h, w, 2)

        vort = np.full((h, w), np.nan)
        vort[1:-1, 1:-1] = (
            vel[:-2, 1:-1, 0] - vel[2:, 1:-1, 0] +   # -vx[i+1, j]
            vel[1:-1, 2:, 1] - vel[1:-1, :-2, 1]     # -vy[i, j-1]
            ) / 2

        return np.dstack([
                x * pix_to_world, 
                y * pix_to_world, 
                disp[..., 0] * pix_to_world, 
                disp[..., 1] * pix_to_world, 
                vel[..., 0] * pix_to_world, 
                vel[..., 1] * pix_to_world, 
                vort * pix_to_world
                ])


    def evaluate(self, pts:np.ndarray) -> np.ndarray:
        return self.interpolator.interpolate(pts)
    

    ###########################
    ## Visualization Methods ##
    ###########################

    def plot_fields(self, dt: float = 1.0, pix_to_world: float = 1.0, 
                    extrapolate: bool = False, arrow_stride: int = 32) -> None:
        fields = self.get_fields(dt, pix_to_world, extrapolate)
        
        x = fields[..., 0]
        y = fields[..., 1]
        vx = fields[..., 4]
        vy = fields[..., 5]
        vort = fields[..., 6]
 
        vel_mag = np.sqrt(vx**2 + vy**2)

        h, w = fields.shape[:2]
        row_idx = np.arange(0, h, arrow_stride)
        col_idx = np.arange(0, w, arrow_stride)
        ii, jj = np.meshgrid(row_idx, col_idx, indexing='ij')
        
        x_sub = x[ii, jj]
        y_sub = y[ii, jj]
        vx_sub = vx[ii, jj]
        vy_sub = vy[ii, jj]

        vel_mag_sub = np.sqrt(vx_sub**2 + vy_sub**2)        
        unit_vx_sub = np.divide(vx_sub, vel_mag_sub, where=vel_mag_sub!=0, out=np.zeros_like(vx_sub))
        unit_vy_sub = np.divide(vy_sub, vel_mag_sub, where=vel_mag_sub!=0, out=np.zeros_like(vy_sub))

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity Magnitude Plot
        mag_plot = axs[0, 0].imshow(vel_mag, cmap='viridis', origin='upper')
        axs[0, 0].quiver(x_sub, y_sub, unit_vx_sub, unit_vy_sub,
                    angles='xy', scale_units='xy', scale=0.1, color='white')
        cbar0 = fig.colorbar(mag_plot, ax=axs[0, 0], format='%.1e')
        axs[0, 0].set_title("Magnitude (m/s)")
        
        # X Component
        u_plot = axs[0, 1].imshow(vx, cmap='RdBu_r', origin='upper')
        axs[0, 1].quiver(x_sub, y_sub, unit_vx_sub, unit_vy_sub,
                    angles='xy', scale_units='xy', scale=0.1, color='black')
        cbar1 = fig.colorbar(u_plot, ax=axs[0, 1], format='%.1e')
        axs[0, 1].set_title("u (m/s)")

        # Y Component
        v_plot = axs[1, 0].imshow(vy, cmap='RdBu_r', origin='upper')
        axs[1, 0].quiver(x_sub, y_sub, unit_vx_sub, unit_vy_sub,
                    angles='xy', scale_units='xy', scale=0.1, color='black')
        cbar2 = fig.colorbar(v_plot, ax=axs[1, 0], format='%.1e')
        axs[1, 0].set_title("v (m/s)")

        # Vorticity
        vort_plot = axs[1, 1].imshow(vort, cmap='coolwarm', origin='upper')
        cbar3 = fig.colorbar(vort_plot, ax=axs[1, 1], format='%.1e')
        axs[1, 1].set_title("Vorticity (rad/s)")

        plt.tight_layout()
        plt.show()

        return


    def plot_intersections(self) -> None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        t0_points = np.array([p for row in self.grid_T0.grid for p in row if p is not None])
        dt_points = np.array([p for row in self.grid_dT.grid for p in row if p is not None])

        axes[0].imshow(self.grid_T0.image, cmap='gray')  
        if t0_points.size > 0:
            axes[0].scatter(
                t0_points[:, 0], t0_points[:, 1], color='blue', 
                marker='o', label='t0 Grid Points'
                )
        axes[0].set_title("t0 Grid")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].grid(False)

        axes[1].imshow(self.grid_dT.image, cmap='gray')  # Display reference image
        if dt_points.size > 0:
            axes[1].scatter(
                dt_points[:, 0], dt_points[:, 1], color='red', 
                marker='x', label='dt Grid Points'
                )
        axes[1].set_title("dt Grid")
        axes[1].set_xlabel("X")
        axes[1].grid(False)

        plt.tight_layout()
        plt.show()

        return 
    

    def plot_lines(self) -> None:
        """
        Plot the Hough lines detected in the t0 grid.
        """
        self.grid_T0.plot_hough_lines()
        
        return 