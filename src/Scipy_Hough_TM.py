from utility.py_import import np, plt, tri, dataclass, field, Tuple
from src.T0_grid_struct import T0GridStruct
from src.dT_grid_struct import DTGridStruct
from cython_build.ParametricX import ParametricX
from src.parametric_opt import ParameterOptimizer
from src.interpolator import dim2Interpolator


@dataclass
class HoughTM:
    path_ref : str
    path_mov : str
    num_lines: int
    optimize : bool = False
    verbose  : bool = False

    # Template Matching Optimization Parameters
    fwhm        : float = 3
    uncertainty : float = 1
    num_interval: int = 10
    intensity   : float = 0.5
    temp_scale  : float = 0.67

    # Hough Line Transform Parameters
    density  : int = 10
    threshold: float = 0.2

    # Lucas Kanade Optical Flow Parameters
    win_size : Tuple[int, int] = (31, 31)
    max_level: int = 5
    iteration: int = 10
    epsilon  : float = 0.001

    solve_bool  : bool = field(init=False)
    valid_ij    : np.ndarray = field(init=False) 
    grid_T0     : T0GridStruct = field(init=False)
    grid_dT     : DTGridStruct = field(init=False)
    interpolator: dim2Interpolator = field(init=False)

    def __post_init__(self):
        shape = (self.num_lines, self.num_lines)
        self.uncertainty = self.fwhm

        self.grid_T0 = T0GridStruct(
            shape, 
            self.path_ref, 
            num_lines=self.num_lines, 
            threshold=self.threshold, 
            density=self.density,
            temp_scale=self.temp_scale
            )
        if self.optimize:
            self._optimize(self.grid_T0, False)

        self.grid_dT = DTGridStruct(
            self.grid_T0, 
            self.path_mov, 
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
        self.disp_field = np.full((self.num_lines, self.num_lines, 4), np.nan)
        self.solve_bool = False
        

    def _optimize(self, grid_obj: np.ndarray, v: bool = False) -> None:
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
        shape = (self.num_lines, self.num_lines)
        self.uncertainty = self.fwhm

        self.grid_T0 = T0GridStruct(
            shape, 
            self.path_ref, 
            num_lines=self.num_lines, 
            threshold=self.threshold, 
            density=self.density,
            temp_scale=self.temp_scale
            )
        if self.optimize:
            self._optimize(self.grid_T0, False)

        self.grid_dT = DTGridStruct(
            self.grid_T0, 
            self.path_mov, 
            win_size=self.win_size,
            max_level=self.max_level,
            iteration=self.iteration,
            epsilon=self.epsilon
            )

        self.disp_field = np.empty(shape, dtype=object)
        self.solve_bool = False
        

    def _optimize(self, grid_obj: np.ndarray, v: bool = False) -> None:
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


    def get_fields(self, dt:float=1, extrapolate:bool=False) -> np.ndarray:
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

        return np.dstack([x, y, disp[..., 0], disp[..., 1], vel[..., 0], vel[..., 1], vort])


    def evaluate(self, pts:np.ndarray) -> np.ndarray:
        return self.interpolator.interpolate(pts)
    

    def plot_fields(self, dt: float = 1.0, extrapolate: bool = False, arrow_stride: int = 32) -> None:
        fields = self.get_fields(dt, extrapolate)
        
        x = fields[..., 0]
        y = fields[..., 1]
        dx = fields[..., 2]
        dy = fields[..., 3]
        vx = fields[..., 4]
        vy = fields[..., 5]
        vort = fields[..., 6]

        disp_mag = np.sqrt(dx**2 + dy**2)
        vel_mag = np.sqrt(vx**2 + vy**2)

        # Create subsampled indices for arrows
        h, w = fields.shape[:2]
        row_idx = np.arange(0, h, arrow_stride)
        col_idx = np.arange(0, w, arrow_stride)
        ii, jj = np.meshgrid(row_idx, col_idx, indexing='ij')
        
        # Subsampled data for quiver plots
        x_sub = x[ii, jj]
        y_sub = y[ii, jj]
        dx_sub = dx[ii, jj]
        dy_sub = dy[ii, jj]
        vx_sub = vx[ii, jj]
        vy_sub = vy[ii, jj]

        # Normalize vectors for quiver plots
        disp_mag_sub = np.sqrt(dx_sub**2 + dy_sub**2)
        vel_mag_sub = np.sqrt(vx_sub**2 + vy_sub**2)
        
        unit_dx_sub = np.divide(dx_sub, disp_mag_sub, where=disp_mag_sub!=0, out=np.zeros_like(dx_sub))
        unit_dy_sub = np.divide(dy_sub, disp_mag_sub, where=disp_mag_sub!=0, out=np.zeros_like(dy_sub))
        unit_vx_sub = np.divide(vx_sub, vel_mag_sub, where=vel_mag_sub!=0, out=np.zeros_like(vx_sub))
        unit_vy_sub = np.divide(vy_sub, vel_mag_sub, where=vel_mag_sub!=0, out=np.zeros_like(vy_sub))

        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Displacement Magnitude
        disp_plot = axs[0].imshow(disp_mag, cmap='viridis', origin='lower')
        axs[0].quiver(x_sub, y_sub, unit_dx_sub, unit_dy_sub,
                    angles='xy', scale_units='xy', scale=0.1, color='black')
        fig.colorbar(disp_plot, ax=axs[0])
        axs[0].set_title("Displacement")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")

        # Plot 2: Velocity Field
        vel_plot = axs[1].imshow(vel_mag, cmap='viridis', origin='lower')
        axs[1].quiver(x_sub, y_sub, unit_vx_sub, unit_vy_sub,
                    angles='xy', scale_units='xy', scale=0.1, color='blue')
        fig.colorbar(vel_plot, ax=axs[1])
        axs[1].set_title("Velocity")
        axs[1].set_xlabel("X")

        # Plot 3: Vorticity Field
        vort_plot = axs[2].imshow(vort, cmap='coolwarm', origin='lower')
        fig.colorbar(vort_plot, ax=axs[2])
        axs[2].set_title("Vorticity")
        axs[2].set_xlabel("X")

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