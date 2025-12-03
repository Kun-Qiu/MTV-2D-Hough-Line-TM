from networkx import sigma
from utility.py_import import np, plt, dataclass, field, Tuple
from src.T0_grid_struct import T0GridStruct
from src.dT_grid_struct import DTGridStruct
# from cython_build.ParametricX import ParametricX
# from src.parametric_opt import ParameterOptimizer
from src.interpolator import dim2Interpolator


@dataclass
class HoughTM:
    ref    : np.ndarray
    mov    : np.ndarray
    num_lines   : Tuple[int, int]
    slope_thresh: Tuple[int, int]
    # optimize : bool = False
    interp: int = 0

    # Guided Images for Filtering
    ref_avg: np.ndarray = None
    mov_avg: np.ndarray = None

    # Template Matching Optimization Parameters
    # fwhm        : float = field(init=False)
    # uncertainty : float = field(init=False)
    # num_interval: int = field(init=False)
    # intensity   : float = field(init=False)
    # temp_scale  : float = field(init=False)

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
        # Default values for parameters
        self.set_hough_params(density=10, threshold=0.2)
        # self.set_template_params(
        #     fwhm=3, uncertainty=1, num_interval=30, 
        #     intensity=0.5, temp_scale=0.67
        #     )
        self.set_optical_flow_params(
            win_size=(31, 31), max_level=3, 
            iteration=10, epsilon=0.001
            )
        # self.uncertainty = self.fwhm

        self.grid_T0 = T0GridStruct( 
            self.ref, 
            avg_image=self.ref_avg,
            num_lines=self.num_lines,
            slope_thresh=self.slope_thresh,
            threshold=self.threshold, 
            density=self.density
            # temp_scale=self.temp_scale
            )
        # if self.optimize:
        #     self._template_optimize(self.grid_T0)

        self.grid_dT = DTGridStruct(
            self.grid_T0, 
            self.mov,
            avg_image=self.mov_avg, 
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


    # def set_template_params(
    #         self, fwhm: float, uncertainty: float, num_interval: int, 
    #         intensity: float, temp_scale: float
    #         ) -> None:
        
    #     self.fwhm = fwhm
    #     self.uncertainty = uncertainty
    #     self.num_interval = num_interval
    #     self.intensity = intensity
    #     self.temp_scale = temp_scale
    #     return 
    

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
    

    # def _template_optimize(self, grid_obj: np.ndarray) -> None:
    #     grid_valid = np.array(
    #         [[cell is not None for cell in row] 
    #         for row in grid_obj.grid]
    #         )
    #     params_valid = np.array(
    #         [[cell is not None for cell in row] 
    #         for row in grid_obj.params]
    #         )
    
    #     valid_mask = grid_valid & params_valid
    #     valid_indices = np.argwhere(valid_mask)

    #     for i, j in valid_indices:
    #         x, y  = grid_obj.grid[i, j]
    #         ang1, ang2, leg_len = grid_obj.params[i, j]

    #         parametricX_obj = ParametricX(
    #             center=(x, y), 
    #             shape=(ang1, ang2, self.intensity, self.fwhm, leg_len),
    #             image=grid_obj.image
    #             )
        
    #         optimizer = ParameterOptimizer(
    #             parametricX_obj, uncertainty=self.uncertainty, 
    #             num_interval=self.num_interval
    #             )

    #         # parameter_star = optimizer.quad_optimize()
    #         parameter_star = optimizer.quad_optimize_gradient()
    #         grid_obj.grid[i, j] = parameter_star[0:2]
    #     return


    def solve(self) -> None:
        for i, j in self.valid_ij:
            x0, y0 = self.grid_T0.grid[i, j]
            x1, y1 = self.grid_dT.grid[i, j]

            dx, dy = (x1 - x0), (y1 - y0)
            self.disp_field[i, j] = [x0, y0, dx, dy]

        self.solve_bool = True
        return
    

    def sequence_solver(self, single_sequence: list[np.ndarray], avg_sequence: list[np.ndarray]) -> None:
        """
        Solve a sequence of images to update the displacement grid over time.
        """
        self.grid_dT.sequence_solver(single_sequence, avg_sequence)
        return 


    def get_fields(self, dt:float=1, pix_to_world: float = 1, extrapolate:bool=False) -> np.ndarray:
        """
        Get the interpolated displacement, velocity, and vorticity fields.
        """
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
                method=self.interp,
                extrapolate=extrapolate
            )
        else:
            raise ValueError("No valid displacement points available for interpolation.")

        # Interpolate the fields
        h, w = self.grid_T0.image.shape[:2]
        y, x = np.mgrid[0:h, 0:w]
        points = np.column_stack([x.ravel(), y.ravel()])
        
        disp = self.interpolator.interpolate(points)
        convex_hull_mask = self.interpolator.is_inside_bounds(points)
        disp[~convex_hull_mask] = np.nan

        vel = (disp.copy() / dt).reshape(h, w, 2)
        disp = disp.reshape(h, w, 2)

        vort = np.full((h, w), np.nan)
        dvx_dy, _ = np.gradient(vel[..., 0])  # ∂v_x/∂y, ∂v_x/∂x
        _, dvy_dx = np.gradient(vel[..., 1])
        vort = dvy_dx - dvx_dy

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
    

    def set_ref(self, im: np.ndarray) -> None:
        self.ref = im
        return

    
    def set_mov(self, im: np.ndarray) -> None:
        self.mov = im
        return
    

    ###########################
    ## Visualization Methods ##
    ###########################

    def plot_fields(self, dt: float = 1.0, pix_to_world: float = 1.0, extrapolate: bool = False) -> None:
        
        fields = self.get_fields(dt, pix_to_world, extrapolate)
        vx, vy, vort = fields[..., 4], fields[..., 5], fields[..., 6]
        vel_mag = np.sqrt(vx**2 + vy**2)

        valid_points = np.array([self.disp_field[i, j][:2] for i, j in self.valid_ij])
        X = np.round(valid_points[:, 0]).astype(int)
        Y = np.round(valid_points[:, 1]).astype(int)
        
        Vx = vx[Y, X] 
        Vy = vy[Y, X] 

        Vel_mag = vel_mag[Y, X]
        unit_Vx = Vx / Vel_mag
        unit_Vy = Vy / Vel_mag

        fig, axs = plt.subplots(2, 2, figsize=(16, 10))

        # Velocity Magnitude Plot
        mag_plot = axs[0, 0].imshow(vel_mag, cmap='RdBu_r', origin='upper')
        axs[0, 0].quiver(X, Y, unit_Vx, unit_Vy, color='red', scale=20)
        cbar0 = fig.colorbar(mag_plot, ax=axs[0, 0], format='%.1e')
        axs[0, 0].set_title("Magnitude (m/s)")
        
        # X Component
        u_plot = axs[0, 1].imshow(vx, cmap='RdBu_r', origin='upper')
        cbar1 = fig.colorbar(u_plot, ax=axs[0, 1], format='%.1e')
        axs[0, 1].set_title("u (m/s)")

        # Y Component
        v_plot = axs[1, 0].imshow(vy, cmap='RdBu_r', origin='upper')
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