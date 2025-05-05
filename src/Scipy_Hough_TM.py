from utility.py_import import np, plt, tri, dataclass, field
import matplotlib.tri as tri
from src.T0_grid_struct import T0GridStruct
from src.dT_grid_struct import DTGridStruct
from src.parametric_X import ParametricX
from src.parametric_opt import ParameterOptimizer


@dataclass
class HoughTM:
    path_ref    : str
    path_mov    : str
    num_lines   : int

    verbose     : bool = True
    density     : int = 10
    threshold   : float = 0.2
    temp_scale  : float = 0.67
    window_scale: float = 1.2
    search_scale: float = 2.0
    down_scale  : int = 4

    displacement: np.ndarray = field(init=False)
    solve_bool  : bool = field(init=False)
    grid_T0     : T0GridStruct = field(init=False)
    grid_dT     : DTGridStruct = field(init=False)

    def __post_init__(self):
        shape = (self.num_lines, self.num_lines)
        self.grid_T0 = T0GridStruct(
            shape, 
            self.path_ref, 
            num_lines=self.num_lines, 
            threshold=self.threshold, 
            density=self.density,
            temp_scale=self.temp_scale
            )
        
        self.grid_dT = DTGridStruct(
            self.grid_T0, 
            self.path_mov, 
            down_scale=self.down_scale, 
            window_scale=self.window_scale, 
            search_scale=self.search_scale, 
            rotate_range=45
            )

        # Initialize displacement field
        self.disp_field = np.empty(shape, dtype=object)
        self._optimize(self.grid_T0)
        self._optimize(self.grid_dT, visualize=False)
        self.solve_bool = False
        

    def _optimize(self, grid_obj, visualize=False) -> None:
        """
        Optimize the parameters of the template matching algorithm
        """
        for i in range(grid_obj.shape[0]):
            for j in range(grid_obj.shape[1]):
                if grid_obj.grid[i, j] is not None and grid_obj.params[i, j] is not None:
                    x, y  = grid_obj.grid[i, j]
                    ang1, ang2, leg_len = grid_obj.params[i, j]
                    parametricX_obj = ParametricX(
                        center=(int(round(x)), int(round(y))), 
                        shape=(ang1, ang2, 0.5, 4, leg_len),
                        image=grid_obj.image
                        )
                    
                    optimizer = ParameterOptimizer(
                        parametricX_obj, lock_angle=False, verbose=self.verbose
                        )
                    optimizer.quad_optimize()
                    if visualize:
                        optimizer.visualize()
                    grid_obj.grid[i, j] = parametricX_obj.params[0:2]

        print("Optimization complete.")


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
        
        rows, cols      = self.grid_T0.shape
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