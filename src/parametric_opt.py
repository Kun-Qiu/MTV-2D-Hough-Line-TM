from utility.py_import import np, plt, cv2, warnings, dataclass, field, Tuple
from cython_build.ParametricX import ParametricX
from scipy.optimize import minimize


@dataclass
class ParameterOptimizer:
    parametric_X: ParametricX
    uncertainty : float = 3.0
    num_interval: int = 10
    generation  : int = 3
    shrnk_factor: int = 2

    verbose   : bool = False

    rad   : Tuple[float, float, float, float, float, float] = field(init=False)
    n_rad : Tuple[int, int, int, int, int, int] = field(init=False)

    def __post_init__(self):
        shape = self.parametric_X.shape

        self.rad = np.array([
            self.uncertainty*2, self.uncertainty*2,
            np.arctan(self.uncertainty/shape[4]),
            np.arctan(self.uncertainty/shape[4]), 
            np.min([shape[2], 1-shape[2]]) / 2, 
            0.75*shape[4]
            ])

        NRPos = np.ceil(self.num_interval/2)
        self.n_rad = np.array([
            NRPos, NRPos, self.num_interval, self.num_interval, 
            self.num_interval, self.num_interval
            ]).astype(int)
        
        if self.verbose:
            print("#############################################################################")
            print(f"#### Initial parameters: {self.__format_verbose(self.parametric_X.get_params())}")
            print(f"#### Initial radius: {self.__format_verbose(self.rad)}")
            print(f"#### Initial number of intervals: {self.__format_verbose(self.n_rad)}")
            print("#############################################################################")


    @staticmethod
    def __format_verbose(arr: np.ndarray) -> str:
        return ', '.join([f'{x:.4f}' for x in arr])


    def __correlate_batch(self, params_array: np.ndarray) -> np.ndarray:
        return np.array([
            self.parametric_X.correlate(p)['correlation'] for p in params_array
            ])


    def quad_optimize_gradient(self) -> np.ndarray:
        try:
            x = self.parametric_X.get_params()
            x0 = x[:6]
            leg_len = x[6]

            # bounds = [
            #     (x0[0] - self.rad[0], x0[0] + self.rad[0]),  # x
            #     (x0[1] - self.rad[1], x0[1] + self.rad[1]),  # y
            #     (x0[2] - self.rad[2], x0[2] + self.rad[2]),  # θ1
            #     (x0[3] - self.rad[3], x0[3] + self.rad[3]),  # θ2
            #     (x0[4] - self.rad[4], x0[4] + self.rad[4]),  # I
            #     (x0[5] - self.rad[5], x0[5] + self.rad[5])   # FWHM
            #     ]
            bounds = [
                (x0[0] - 1.0, x0[0] + 1.0),  # x (±1 pixel)
                (x0[1] - 1.0, x0[1] + 1.0),  # y (±1 pixel)
                (x0[2] - np.radians(5), x0[2] + np.radians(5)),  # θ1
                (x0[3] - np.radians(5), x0[3] + np.radians(5)),  # θ2
                (max(0, x0[4] * 0.9), x0[4] * 1.1),  # I (10% variation)
                (max(0.1, x0[5] * 0.9), x0[5] * 1.1)   # FWHM (10% variation)
                ]
            
            def objective(params: np.ndarray) -> float:
                try:
                    full_params = np.concatenate([params, [leg_len]])
                    corr = self.parametric_X.correlate(full_params)['correlation']
                    if not np.isfinite(corr):
                        return 1e6 
                    return -corr 
                except:
                    return 1e6
            
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 100, 
                    'ftol': 1e-6,    # function tolerance
                    'gtol': 1e-6,    # gradient tolerance
                    'eps': 1e-5 
                    }
                )

            if result.success:
                self.parametric_X.update_params(np.arange(len(x0)), result.x)
            else:
                print(f"Optimization failed: {result.message}")
        
        except Exception as e:
            print(f"Error in gradient descent: {e}")
            raise

        return self.parametric_X.get_params()
    

    def quad_optimize(self) -> np.ndarray:
        try:
            warnings.filterwarnings("error")

            for G in range(self.generation):
                cur_rad = self.rad / (self.shrnk_factor ** G)

                x_vals = np.linspace(
                    self.parametric_X.get_params()[0] - cur_rad[0],
                    self.parametric_X.get_params()[0] + cur_rad[0] + 1e-8,
                    num=(2*self.n_rad[0])+1
                    )
                
                y_vals = np.linspace(
                    self.parametric_X.get_params()[1] - cur_rad[1],
                    self.parametric_X.get_params()[1] + cur_rad[1] + 1e-8,
                    num=(2*self.n_rad[1])+1
                    )
                
                xx, yy = np.meshgrid(x_vals, y_vals)
                params_batch = np.tile(self.parametric_X.get_params(), (xx.size, 1))
                params_batch[:, 0] = xx.ravel()
                params_batch[:, 1] = yy.ravel()

                grid_corrs = self.__correlate_batch(params_batch).reshape(xx.shape)
                opt_x, opt_y = self._quad_fit_2D(
                    x_vals, y_vals, grid_corrs
                    )
                self.parametric_X.update_params([0, 1], [opt_x, opt_y])

                for idx in [2, 3, 4, 5]:
                    vals = np.linspace(
                        -cur_rad[idx], cur_rad[idx], 
                        num=(2*self.n_rad[idx])+1
                        )
                    params_batch = np.tile(self.parametric_X.get_params(), (len(vals), 1))
                    params_batch[:, idx] += vals
                    corrs = self.__correlate_batch(params_batch)
                    
                    opt_dval = self._quad_fit_1D(vals, corrs)
                    opt_val = self.parametric_X.get_params()[idx] + opt_dval
                    self.parametric_X.update_params([idx], [opt_val])

        except Warning as w:
            print(f"Warning encountered during optimization: {w}")
        except Exception as e:
            print(f"Error encountered during optimization: {e}")
            print(self.parametric_X.shape)
            raise
        finally:
            warnings.filterwarnings("default")
        
        return self.parametric_X.get_params()


    def _quad_fit_1D(self, values: np.ndarray, corrs: np.ndarray) -> float:
        try:
            coeffs = np.polyfit(values, corrs, 2)
            a = coeffs[0]
            optimal = -coeffs[1]/(2 * a)

            if a >= 0 or optimal < values[0] or optimal > values[-1]:
                return values[np.argmax(corrs)]
            
            return optimal
        except np.linalg.LinAlgError:
            return values[np.argmax(corrs)]

    
    def _quad_fit_2D(
            self, x_vals: np.ndarray, y_vals: np.ndarray, corr_matrix: np.ndarray
            ) -> Tuple[float, float]:
        
        max_idx = np.argmax(corr_matrix)
        j_max, i_max = np.unravel_index(max_idx, corr_matrix.shape)
        opt_x, opt_y = x_vals[i_max], y_vals[j_max]
        X, Y = np.meshgrid(x_vals, y_vals)
        
        A = np.column_stack((
            X.ravel(), Y.ravel(),
            X.ravel()**2, Y.ravel()**2,
            X.ravel() * Y.ravel(),
            np.ones_like(X.ravel())
        ))
        b = corr_matrix.ravel()

        try:
            coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return opt_x, opt_y

        # Solve for critical point: ∇f = 0
        M = np.array([[2 * coef[2], coef[4]], [coef[4], 2 * coef[3]]])
        rhs = np.array([-coef[0], -coef[1]])
        
        try:
            x_s, y_s = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return opt_x, opt_y
        
        if coef[2] < 0 and (4 * coef[2] * coef[3] - coef[4]**2) > 0:
            # Validate bounds
            if (x_vals[0] <= x_s <= x_vals[-1] and 
                y_vals[0] <= y_s <= y_vals[-1]):
                return x_s, y_s

        return opt_x, opt_y
    

    def visualize(self) -> None:
        """
        Visualize the current state of the ParametricX instance with the 
        template overlay on the image.
        """

        img = self.parametric_X.image
        if img is None:
            raise ValueError("No image available for visualization")
        
        template, (min_col, min_row) = self.parametric_X.get_parametric_X()
        
        fig = plt.figure(figsize=(15, 7))
        ax1 = plt.subplot(121)
        plt.imshow(img, cmap='gray')
        
        extent = [
            min_col - 0.5, 
            min_col + template.shape[1] - 0.5,  
            min_row + template.shape[0] - 0.5,  
            min_row - 0.5 
        ]
        
        plt.imshow(template, cmap='viridis', alpha=0.7, extent=extent)
        plt.scatter(min_col + template.shape[1]/2, 
                min_row + template.shape[0]/2,
                c='cyan', marker='o', s = 100,
                edgecolors='red', linewidth=1)
        plt.title("Template Overlay on Image")
        
        ax2 = plt.subplot(122)
        plt.imshow(template, cmap='viridis', 
                extent=[min_col, min_col + template.shape[1],
                        min_row + template.shape[0], min_row])
        plt.colorbar(label='Template Intensity')
        plt.title("Template Only")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage of the ParameterOptimizer 
    # with a ParametricX instance
    
    import os
    image_dir = os.path.abspath("data/Synthetic_Data/Image/SNR_1/0/displaced_lamb_oseen.png")
    fwhm = 4

    img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    parameter_X = ParametricX(
        center=(110, 123),
        shape=(np.pi / 6, -np.pi / 6, 0.5, fwhm, 38 * 0.7),
        image=img
    )

    # Initialize and optimize
    optimizer = ParameterOptimizer(
        parametric_X=parameter_X,
        lock_angle=False
    )
    result = optimizer.parametric_X.correlate(optimizer.parametric_X.params)
    print("Correlation Result:", result)
    corr = optimizer.quad_optimize()
    print("Optimized Parameters:", optimizer.parametric_X.params)
    optimizer.visualize()