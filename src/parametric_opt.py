from utility.py_import import np, plt, cv2, warnings, dataclass, field, Tuple
from src.parametric_X import ParametricX

@dataclass
class ParameterOptimizer:
    parametric_X: ParametricX
    uncertainty : float = 1.0
    num_interval: int = 5
    generation  : int = 3
    shrnk_factor: int = 2

    lock_angle: bool = False
    verbose   : bool = False

    rad   : Tuple[float, float, float, float, float, float] = field(init=False)
    n_rad : Tuple[int, int, int, int, int, int] = field(init=False)

    def __post_init__(self):
        shape = self.parametric_X.shape

        self.rad = np.array([
            self.uncertainty*2, self.uncertainty*2,
            np.arctan(self.uncertainty/shape[4]),
            np.arctan(self.uncertainty/shape[4]), 
            np.min([shape[2], 1-shape[2]])/2, 
            0.75*shape[2]
            ])
        
        NRPos = np.ceil(self.num_interval/2)
        self.n_rad = np.array([
            NRPos, NRPos, self.num_interval, self.num_interval, 
            self.num_interval, self.num_interval
            ])
        
        if self.verbose:
            print("#############################################################################")
            print(f"#### Initial parameters: {self.__format_verbose(self.parametric_X.params)}")
            print(f"#### Initial radius: {self.__format_verbose(self.rad)}")
            print(f"#### Initial number of intervals: {self.__format_verbose(self.n_rad)}")
            print("#############################################################################")


    @staticmethod
    def __format_verbose(arr: np.ndarray) -> str:
        return ', '.join([f'{x:.2f}' for x in arr])


    def __correlate_batch(self, params_array: np.ndarray, temp_opt: ParametricX) -> np.ndarray:
        return np.array([
            temp_opt.correlate(p)['correlation'] for p in params_array
            ])


    def quad_optimize(self) -> np.ndarray:
        num_params = len([
            r for r, nr in zip(self.rad[2:], self.n_rad[2:]) 
            if r > 0 and nr > 0
            ])
        
        max_steps = (self.generation * (num_params + 1)) + 1
        corr = np.full(max_steps, np.nan) # 1D array for each template
        
        try:
            warnings.filterwarnings("error")
            
            temp_opt = ParametricX(
                center=self.parametric_X.center,
                shape=self.parametric_X.shape,
                image=self.parametric_X.image
                )
            
            cur_corr = self.parametric_X.correlate(self.parametric_X.params)
            corr[0] = cur_corr['correlation']

            for G in range(self.generation):
                cur_rad = self.rad / (self.shrnk_factor ** G)
                corr_idx = (G * (num_params + 1)) + 1

                if cur_rad[0] > 1e-9 and cur_rad[1] > 1e-9:
                    x_vals = np.linspace(
                        self.parametric_X.params[0] - cur_rad[0],
                        self.parametric_X.params[0] + cur_rad[0] + 1e-8,
                        num=int(2*(self.n_rad[0])+1)
                        )
                    
                    y_vals = np.linspace(
                        self.parametric_X.params[1] - cur_rad[1],
                        self.parametric_X.params[1] + cur_rad[1] + 1e-8,
                        num=int(2*(self.n_rad[1])+1)
                        )
                    
                    xx, yy = np.meshgrid(x_vals, y_vals)
                    params_batch = np.tile(self.parametric_X.params, (xx.size, 1))
                    params_batch[:, 0] = xx.ravel()
                    params_batch[:, 1] = yy.ravel()

                    grid_corrs = self.__correlate_batch(params_batch, temp_opt).reshape(xx.shape)
                    opt_x, opt_y = self._quad_fit_2D(
                        x_vals, y_vals, grid_corrs, 
                        (x_vals[0], x_vals[-1]), 
                        (y_vals[0], y_vals[-1])
                        )
                    self.parametric_X.params[0], self.parametric_X.params[1] = opt_x, opt_y
                    corr[corr_idx] = self.parametric_X.correlate(self.parametric_X.params)['correlation']
                    corr_idx += 1

                # Vectorized angle optimization
                if self.lock_angle:
                    ang_vals = np.linspace(
                        -cur_rad[2], cur_rad[2], 
                        num=int(2*(self.n_rad[2])+1)
                        )
                    
                    params_batch = np.tile(self.parametric_X.params, (len(ang_vals), 1))
                    params_batch[:, 2] += ang_vals
                    params_batch[:, 3] += ang_vals
                    ang_corrs = self.__correlate_batch(params_batch, temp_opt)
                    
                    best_da, a_coeff = self._quad_fit_1D(ang_vals, ang_corrs)
                    if (a_coeff >= 0) or (best_da < ang_vals[0]) or (best_da > ang_vals[-1]):
                        best_da = ang_vals[np.argmax(ang_corrs)]
                    
                    self.parametric_X.params[2] += best_da
                    self.parametric_X.params[3] += best_da
                    corr[corr_idx] = self.parametric_X.correlate(self.parametric_X.params)['correlation']
                    corr_idx += 1
                else:
                    for ang_idx in [2, 3]:
                        ang_vals = np.linspace(
                            -cur_rad[ang_idx], cur_rad[ang_idx], 
                            num=int(2*(self.n_rad[ang_idx])+1)
                            )
                        params_batch = np.tile(self.parametric_X.params, (len(ang_vals), 1))
                        params_batch[:, ang_idx] += ang_vals
                        ang_corrs = self.__correlate_batch(params_batch, temp_opt)
                        
                        best_da, a_coeff = self._quad_fit_1D(ang_vals, ang_corrs)
                        if (a_coeff >= 0) or (best_da < ang_vals[0]) or (best_da > ang_vals[-1]):
                            best_da = ang_vals[np.argmax(ang_corrs)]
                        
                        self.parametric_X.params[ang_idx] += best_da
                        corr[corr_idx] = self.parametric_X.correlate(self.parametric_X.params)['correlation']
                        corr_idx += 1

                # Vectorized parameter optimization (4,5)
                for p_idx in [4, 5]:
                    p_vals = np.linspace(
                        -cur_rad[p_idx], cur_rad[p_idx], 
                        num=int(2*(self.n_rad[p_idx])+1)
                        )
                    params_batch = np.tile(self.parametric_X.params, (len(p_vals), 1))
                    params_batch[:, p_idx] += p_vals
                    p_corrs = self.__correlate_batch(params_batch, temp_opt)
                    
                    best_dp, a_coeff = self._quad_fit_1D(p_vals, p_corrs)
                    if (a_coeff >= 0) or (best_dp < p_vals[0]) or (best_dp > p_vals[-1]):
                        best_dp = p_vals[np.argmax(p_corrs)]
                    
                    self.parametric_X.params[p_idx] += best_dp
                    corr[corr_idx] = self.parametric_X.correlate(self.parametric_X.params)['correlation']
                    corr_idx += 1

        except Warning as w:
            print(f"Warning encountered during optimization: {w}")
        except Exception as e:
            print(f"Error encountered during optimization: {e}")
            raise
        finally:
            warnings.filterwarnings("default")
        
        if self.verbose:
            print(f"Optimized parameters: {self.__format_verbose(self.parametric_X.params)}")
        return corr


    def _quad_fit_1D(self, values: np.ndarray, corrs: np.ndarray) -> Tuple[float, float]:
        """Optimized quadratic fit for 1D parameter optimization"""
        if np.all(values == values[0]) or len(values) < 3:
            return values[np.argmax(corrs)], None
        
        try:
            coeffs = np.polyfit(values, corrs, 2)
            a = coeffs[0]
            if a >= 0:
                return values[np.argmax(corrs)], a
            
            optimal = -coeffs[1]/(2*coeffs[0])
            return optimal, a
        except np.linalg.LinAlgError:
            return values[np.argmax(corrs)], None

    
    def _quad_fit_2D(
            self, x_vals: np.ndarray, y_vals: np.ndarray, 
            corr_matrix: np.ndarray, x_lim: Tuple[float, float], 
            y_lim: Tuple[float, float]
            ) -> Tuple[float, float]:
        
        max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        x_coeffs = np.polyfit(x_vals, corr_matrix[max_idx[0], :], 2)
        y_coeffs = np.polyfit(y_vals, corr_matrix[:, max_idx[1]], 2)

        if x_coeffs[0] < 0:
            opt_x_candidate = -x_coeffs[1] / (2 * x_coeffs[0])
            if x_lim[0] <= opt_x_candidate <= x_lim[1]:
                opt_x = opt_x_candidate
            else:
                opt_x = x_vals[max_idx[1]]
        else:
            opt_x = x_vals[max_idx[1]]
        
        # Calculate optimal y with boundary check
        if y_coeffs[0] < 0:
            opt_y_candidate = -y_coeffs[1] / (2 * y_coeffs[0])
            if y_lim[0] <= opt_y_candidate <= y_lim[1]:
                opt_y = opt_y_candidate
            else:
                opt_y = y_vals[max_idx[0]]
        else:
            opt_y = y_vals[max_idx[0]]
        
        return opt_x, opt_y


def visualize(self) -> None:
        img = self.parametric_X.image
        if img is None:
            raise ValueError("No image available for visualization")
        
        template, (min_col, min_row) = self.parametric_X.get_parametric_X()
        
        # Create figure
        fig = plt.figure(figsize=(15, 7))
        ax1 = plt.subplot(121)
        plt.imshow(img, cmap='gray')
        
        extent = [
            min_col - 0.5,  # left
            min_col + template.shape[1] - 0.5,  # right
            min_row + template.shape[0] - 0.5,  # bottom
            min_row - 0.5  # top
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
        return None


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