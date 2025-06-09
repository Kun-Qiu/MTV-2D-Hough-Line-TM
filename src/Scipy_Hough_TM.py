from utility.py_import import np, plt, tri, dataclass, field, Tuple
from src.T0_grid_struct import T0GridStruct
from src.dT_grid_struct import DTGridStruct
from cython_build.ParametricX import ParametricX
from src.parametric_opt import ParameterOptimizer


@dataclass
class HoughTM:
    path_ref    : str
    path_mov    : str
    num_lines   : int
    fwhm        : float
    uncertainty : float = None
    num_interval: int = 5

    intensity : float = 0.5
    density   : int = 10
    threshold : float = 0.2
    temp_scale: float = 0.67
    win_size  : Tuple[int, int] = (31, 31)
    max_level : int = 5
    iteration : int = 10
    epsilon   : float = 0.001
    verbose   : bool = False
    optimize  : bool = False

    displacement: np.ndarray = field(init=False)
    solve_bool  : bool = field(init=False)
    grid_T0     : T0GridStruct = field(init=False)
    grid_dT     : DTGridStruct = field(init=False)

    def __post_init__(self):
        shape = (self.num_lines, self.num_lines)
        if self.uncertainty is None:
            approx_uncert = True
        else:
            approx_uncert = False

        self.grid_T0 = T0GridStruct(
            shape, 
            self.path_ref, 
            solve_uncert=approx_uncert,
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
        if self.optimize:
            self._optimize(self.grid_dT, False)

        self.disp_field = np.empty(shape, dtype=object)
        self.solve_bool = False
        

    def _optimize(self, grid_obj, v) -> None:
        for i in range(grid_obj.shape[0]):
            for j in range(grid_obj.shape[1]):
                if grid_obj.grid[i, j] is not None and grid_obj.params[i, j] is not None:
                    x, y  = grid_obj.grid[i, j]
                    ang1, ang2, leg_len = grid_obj.params[i, j]

                    parametricX_obj = ParametricX(
                        center=(x, y), 
                        shape=(ang1, ang2, self.intensity, self.fwhm, leg_len),
                        image=grid_obj.image
                        )
                    
                    if self.uncertainty is None:
                        pred_uncertainty = self.grid_T0.uncertainty[i, j]
                        self.uncertainty = np.max([
                            pred_uncertainty[0], 
                            pred_uncertainty[1] 
                            ])
               
                    optimizer = ParameterOptimizer(
                        parametricX_obj, uncertainty=self.uncertainty, 
                        num_interval=self.num_interval, verbose=self.verbose
                        )

                    parameter_star = optimizer.quad_optimize()
                    if v:
                        optimizer.visualize()
                    grid_obj.grid[i, j] = parameter_star[0:2]
        return None


    def solve(self) -> None:
        """
        Solve the correspondence between t0 image and dt img to obtain the change in 
        displacement field
        """
        rows, cols = self.grid_T0.shape
        self.disp_field = np.full((rows, cols, 4), np.nan)  # (x, y, dx, dy)
        valid_points = []
        
        for i in range(rows):
            for j in range(cols):
                if self.grid_T0.grid[i, j] is not None and self.grid_dT.grid[i, j] is not None:
                    x0, y0 = self.grid_T0.grid[i, j]
                    x1, y1 = self.grid_dT.grid[i, j]

                    dx, dy = (x1 - x0), (y1 - y0)
                    self.disp_field[i, j] = [x0, y0, dx, dy]
                    valid_points.append([x0, y0, dx, dy])

        self.solve_bool = True
        return None


    def get_velocity(self, dt:float=1) -> np.ndarray:
        if not self.solve_bool:
            raise ValueError("solve() must be called before get_velocity().")

        vel_field = self.disp_field.copy()
        vel_field[..., 2:] /= dt
        return vel_field


    def get_vorticity(self, dt:float=1) -> np.ndarray:
        if not self.solve_bool:
            raise ValueError("Call solve() before get_vorticity().")
    
        rows, cols = self.grid_T0.shape
        field = np.full((rows, cols, 3), np.nan)
        vorticity = np.full((rows, cols), np.nan)

        vel = self.get_velocity(dt)
        vx = vel[..., 2] 
        vy = vel[..., 3]

        vorticity[1:-1, 1:-1] = (
            vx[:-2, 1:-1] -  # vx[i-1, j]
            vx[2:, 1:-1] +   # -vx[i+1, j]
            vy[1:-1, 2:] -   # vy[i, j+1]
            vy[1:-1, :-2]    # -vy[i, j-1]
        ) / 2

        field[..., 0] = self.disp_field[..., 0]  # x
        field[..., 1] = self.disp_field[..., 1]  # y
        field[..., 2] = vorticity  # vorticity
    
        return field 


    def plot_intersections(self):
        """
        Plots the intersection points from self.t0_grid and self.dt_grid as two different subfigures.
        Each subfigure overlays the points onto the reference image.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Flatten the grids and filter out None values
        t0_points = np.array([p for row in self.grid_T0.grid for p in row if p is not None])
        dt_points = np.array([p for row in self.grid_dT.grid for p in row if p is not None])

        # Plot t0_grid intersections
        axes[0].imshow(self.grid_T0.image, cmap='gray')  # Display reference image
        if t0_points.size > 0:
            axes[0].scatter(t0_points[:, 0], t0_points[:, 1], color='blue', marker='o', label='t0 Grid Points')
        axes[0].set_title("t0 Grid")
        axes[0].set_xlabel("X")
        axes[0].set_ylabel("Y")
        axes[0].grid(False)

        # Plot dt_grid intersections
        axes[1].imshow(self.grid_dT.image, cmap='gray')  # Display reference image
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

        vorticity = self.get_vorticity(dt)
        vel_field = self.get_velocity(dt)
        
        x, y = vel_field[..., 0], vel_field[..., 1]
        vx, vy = vel_field[..., 2], vel_field[..., 3]
        dx, dy = self.disp_field[..., 2], self.disp_field[..., 3]
        vort = vorticity[..., 2]

        disp_mag = np.sqrt(dx**2 + dy**2)
        vel_mag = np.sqrt(vx**2 + vy**2)

        # Normalize vectors for quiver plots (avoid division by zero)
        unit_dx = np.divide(dx, disp_mag, where=disp_mag != 0, out=np.zeros_like(dx))
        unit_dy = np.divide(dy, disp_mag, where=disp_mag != 0, out=np.zeros_like(dy))
        unit_vx = np.divide(vx, vel_mag, where=vel_mag != 0, out=np.zeros_like(vx))
        unit_vy = np.divide(vy, vel_mag, where=vel_mag != 0, out=np.zeros_like(vy))

        valid_mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(dx) & np.isfinite(dy) & np.isfinite(vx) & np.isfinite(vy) & np.isfinite(vort)
        valid_count = np.sum(valid_mask)
        
        if valid_count == 0:
            raise ValueError("All data points are invalid (NaN or Inf). Check the input data.")

        triang = tri.Triangulation(x[valid_mask], y[valid_mask])
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: Displacement Magnitude
        disp_plot = axs[0].tricontourf(triang, disp_mag[valid_mask], cmap='viridis', levels=100)
        axs[0].quiver(x[valid_mask], y[valid_mask], unit_dx[valid_mask], unit_dy[valid_mask],
                    angles='xy', scale_units='xy', scale=0.1, color='black')
        fig.colorbar(disp_plot, ax=axs[0])
        axs[0].set_title("Displacement")
        axs[0].set_xlabel("X")
        axs[0].set_ylabel("Y")
        axs[0].axis('equal')

        # Plot 2: Velocity Field
        vel_plot = axs[1].tricontourf(triang, vel_mag[valid_mask], cmap='viridis', levels=100)
        axs[1].quiver(x[valid_mask], y[valid_mask], unit_vx[valid_mask], unit_vy[valid_mask],
                    angles='xy', scale_units='xy', scale=0.1, color='blue')
        fig.colorbar(vel_plot, ax=axs[1])
        axs[1].set_title("Velocity")
        axs[1].set_xlabel("X")
        axs[1].axis('equal')

        # Plot 3: Vorticity Field
        vort_plot = axs[2].tricontourf(triang, vort[valid_mask], cmap='coolwarm', levels=100)
        fig.colorbar(vort_plot, ax=axs[2])
        axs[2].set_title("Vorticity")
        axs[2].set_xlabel("X")
        axs[2].axis('equal')

        plt.tight_layout()
        plt.show()