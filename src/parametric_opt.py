from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.signal import correlate2d, convolve2d
import numpy as np
import matplotlib.pyplot as plt
import cv2
import warnings


@dataclass
class ParameterOptimizer:
    center      : Tuple[float, float]
    shape       : Tuple[float, float, float, float, float]
    image       : Optional[np.ndarray]

    uncertainty : float = 1
    num_interval: int = 5
    generation  : int = 3
    shrnk_factor: int = 2
    lock_angle  : bool = False
    num_par     : int = 11

    rad         : Tuple[float, float, float, float, float, float] = field(init=False)
    n_rad       : Tuple[float, float, float, float, float, float] = field(init=False)
    params      : List[float] = field(init=False)


    def __post_init__(self):
        self.params = [
            self.center[0], self.center[1],  # x, y
            self.shape[0] , self.shape[1],   # ang1, ang2
            self.shape[2] , self.shape[3],   # rel_intens, lin_wid
            self.shape[4]   # leg_len
        ]

        NRPos = np.ceil(self.num_interval / 2)
        self.rad = np.array([
            self.uncertainty * 2, self.uncertainty * 2,
            np.arctan(self.uncertainty / self.shape[4]),
            np.arctan(self.uncertainty / self.shape[4]), 
            np.min([self.shape[2], 1-self.shape[2]]) / 2, 
            0.75 * self.shape[2]
        ])
        
        self.n_rad = np.array([
            NRPos, NRPos, self.num_interval, self.num_interval, 
            self.num_interval, self.num_interval
        ])
         
        print(f"Initialized with parameters: {self.params}")
        print(f"Uncertainty values: {self.rad}")


    @staticmethod
    def _rotation_matrix(angle: float, counter_clock=True) -> np.ndarray:
        if counter_clock:
            R = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
        else:
            R = np.array([
                [np.cos(angle), np.sin(angle)],
                [-np.sin(angle), np.cos(angle)]
            ])
        return R


    def _parametric_template(self, params: List[float]) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Generate a parametric template based on the provided parameters"""
        x, y, ang1, ang2, rel_intens, lin_wid, leg_len = params
        half_leg_len = int(round(leg_len / 2))
        
        x_vals = np.arange(-half_leg_len, half_leg_len)
        y_vals = np.arange(-half_leg_len, half_leg_len)
        xx, yy = np.meshgrid(x_vals, y_vals)
        
        rot1, rot2 = self._rotation_matrix(ang1), self._rotation_matrix(ang2, False)
        
        coords = np.stack([xx.ravel(), yy.ravel()]).T
        rot_coords1 = coords @ rot1
        rot_coords2 = coords @ rot2
        
        sigma = lin_wid/ (2 * np.sqrt(2 * np.log(2)))
        leg1 = np.exp(-(rot_coords1[:, 0]**2) / (2 * sigma**2))
        leg2 = np.exp(-(rot_coords2[:, 0]**2) / (2 * sigma**2))
        template = rel_intens * leg1 + (1 - rel_intens) * leg2
        template = template.reshape(xx.shape)
        
        min_col = int(np.clip(x - half_leg_len, 0, self.image.shape[1]))
        min_row = int(np.clip(y - half_leg_len, 0, self.image.shape[0]))
        
        return template, (min_col, min_row)


    def visualize(self) -> None:
        if self.image is None:
            raise ValueError("No image available for visualization")
        
        template, (min_col, min_row) = self._parametric_template(self.params)
        
        # Create figure
        fig = plt.figure(figsize=(15, 7))
        
        # Original image with template overlay
        ax1 = plt.subplot(121)
        plt.imshow(self.image, cmap='gray')
        
        # Calculate template extent in image coordinates
        extent = [
            min_col - 0.5,  # left
            min_col + template.shape[1] - 0.5,  # right
            min_row + template.shape[0] - 0.5,  # bottom
            min_row - 0.5  # top
        ]
        
        # Overlay template with transparency
        plt.imshow(template, cmap='viridis', alpha=0.7, extent=extent)
        plt.scatter(min_col + template.shape[1]/2, 
                min_row + template.shape[0]/2,
                c='cyan', marker='o', s=100,
                edgecolors='red', linewidth=2)
        plt.title("Template Overlay on Image")
        
        # Template visualization
        ax2 = plt.subplot(122)
        plt.imshow(template, cmap='viridis', 
                extent=[min_col, min_col+template.shape[1],
                        min_row+template.shape[0], min_row])
        plt.colorbar(label='Template Intensity')
        plt.title("Template Only")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        
        plt.tight_layout()
        plt.show()


    def correlate(self, params: List[float]) -> dict:
        result = {
            'correlation': -np.inf,
            'peak': 0.0,
            'background': 0.0,
            'noise': np.nan,
            'difference': None
        }

        template, (min_col, min_row) = self._parametric_template(params)
        t_height, t_width = template.shape
        
        img_patch = self.image[min_row:min_row + t_height, 
                               min_col:min_col + t_width]
        if img_patch.shape != template.shape:
            return -np.inf
        
        template_mean, template_std = np.mean(template), np.std(template)
        img_mean, img_std = np.mean(img_patch), np.std(img_patch)

        template_norm = (template - template_mean) / (template_std + 1e-9)
        img_norm = (img_patch - img_mean) / (img_std + 1e-9)

        corr_map = correlate2d(img_norm, template_norm, mode='same', boundary='fill')
        scaled_diff = (img_norm - template_norm) * img_std

        result['peak']            = np.max(corr_map)
        result['correlation']     = np.max(corr_map) / (template.size - 1)
        result['background']      = ((1 - template_mean) / template_std * img_std) + img_mean

        if scaled_diff.shape[0] > 2 and scaled_diff.shape[1] > 2:
            kernel = np.ones((3, 3)) / 9
            local_mean = convolve2d(scaled_diff, kernel, mode='valid')
            noise = scaled_diff[1:-1, 1:-1] - local_mean
            result['noise'] = 5 * np.std(noise)
        
        return result
    

    def quad_optimize(self) -> np.ndarray:
        num_params = len([r for r, nr in zip(self.rad[2:5], self.n_rad[2:5]) if r > 0 and nr > 0])
        corr = np.full((self.generation * (num_params + 1) + 1, len(self.params)), np.nan)
        
        try:
            warnings.filterwarnings("error")
            
            temp_opt = ParameterOptimizer(
                center=self.center,
                shape=self.shape,
                image=self.image
            )
            initial_corr = self.correlate(self.params)
            corr[0] = initial_corr['correlation']

            for G in range(self.generation):
                cur_rad = self.rad / (self.shrnk_factor ** G)
                increment = cur_rad / self.n_rad

                x_vals = np.arange(
                    self.params[0] - cur_rad[0],
                    self.params[0] + cur_rad[0],
                    increment[0], dtype=np.float64
                )
                y_vals = np.arange(
                    self.params[1] - cur_rad[1],
                    self.params[1] + cur_rad[1],
                    increment[1], dtype=np.float64
                )  
                pos_corrs = np.zeros((len(y_vals), len(x_vals)))
                for i, y in enumerate(y_vals):
                    for j, x in enumerate(x_vals):
                        temp_params = self.params.copy()
                        temp_params[0] = x
                        temp_params[1] = y
                        res = temp_opt.correlate(temp_params)
                        pos_corrs[i, j] = res['correlation']

                self.params[0], self.params[1] = self._quad_fit_2D(x_vals, y_vals, pos_corrs)
                corr_idx = G * num_params + 1

                current_result = self.correlate(self.params)
                corr[corr_idx] = current_result['correlation']

                if self.lock_angle:
                    ang_vals = np.arange(-cur_rad[2], cur_rad[2], self.n_rad[2])
                    ang_corrs = []
                    for da in ang_vals:
                        temp_params = self.params.copy()
                        temp_params[2] += da
                        temp_params[3] += da
                        res = temp_opt.correlate(temp_params)
                        ang_corrs.append(res['correlation'])

                    best_da = self._quad_fit_1D(ang_vals, ang_corrs)
                    self.params[2] += best_da
                    self.params[3] += best_da
                else:
                    for ang_idx in [2, 3]:
                        ang_vals = np.arange(-cur_rad[ang_idx],
                                             cur_rad[ang_idx],
                                             self.n_rad[ang_idx])
                        ang_corrs = []
                        for av in ang_vals:
                            temp_params = self.params.copy()
                            temp_params[ang_idx] += av
                            res = temp_opt.correlate(temp_params)
                            ang_corrs.append(res['correlation'])
                        best_da = self._quad_fit_1D(ang_vals, ang_corrs)
                        self.params[ang_idx] += best_da

                corr_idx += 1
                current_result = self.correlate(self.params)
                corr[corr_idx] = current_result['correlation']

                param_indices = [4, 5]
                for p_idx in param_indices:
                    p_vals = np.arange(self.params[p_idx] - cur_rad[p_idx],
                                        self.params[p_idx] + cur_rad[p_idx],
                                        increment[p_idx])
                    p_corrs = []
                    for pv in p_vals:
                        temp_params = self.params.copy()
                        temp_params[p_idx] = pv
                        res = temp_opt.correlate(temp_params)
                        p_corrs.append(res['correlation'])
                    best_dp = self._quad_fit_1D(p_vals, p_corrs)
                    self.params[p_idx] += best_dp

                    corr_idx += 1
                    current_result = self.correlate(self.params)
                    corr[corr_idx] = current_result['correlation']

        except Warning as w:
            print(f"Warning encountered during optimization: {w}")
        except Exception as e:
            print(f"Error encountered during optimization: {e}")
            raise
        finally:
            warnings.filterwarnings("default")

        return corr

    
    def _quad_fit_1D(self, values, corrs):
        """Quadratic fit for 1D parameter optimization"""
        coeffs = np.polyfit(values, corrs, 2)
        
        if coeffs[0] >= 0:  # Check for minimum
            return values[np.argmax(corrs)]
        
        optimal = -coeffs[1] / (2 * coeffs[0])
        return optimal if (values[0] <= optimal <= values[-1]) else values[np.argmax(corrs)]

    
    def _quad_fit_2D(self, x_vals, y_vals, corr_matrix):
        """Quadratic fit for 2D position optimization"""
        max_idx = np.unravel_index(np.argmax(corr_matrix), corr_matrix.shape)
        x_coeffs = np.polyfit(x_vals, corr_matrix[max_idx[0], :], 2)
        y_coeffs = np.polyfit(y_vals, corr_matrix[:, max_idx[1]], 2)
        
        opt_x = -x_coeffs[1]/(2*x_coeffs[0]) if x_coeffs[0] < 0 else x_vals[max_idx[1]]
        opt_y = -y_coeffs[1]/(2*y_coeffs[0]) if y_coeffs[0] < 0 else y_vals[max_idx[0]]
        
        return opt_x, opt_y

    
    def _get_temp_params(self, param_idx, value):
        """Helper to generate temporary parameter sets"""
        temp_params = self.params.copy()
        temp_params[param_idx] = value
        return temp_params



if __name__ == "__main__":
    import os
    # image_dir = os.path.abspath("data/Experimental_Data/Target/frame_2_2us.png")
    image_dir = os.path.abspath("data/Synthetic_Data/Image/displaced_poiseuille.png")
    fwhm      = 6             # Full width at half maximum for the Gaussian lines
    angle     = 120           # Angle for intersecting lines

    img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    
    # Initialize and optimize
    optimizer = ParameterOptimizer(
        center=(67, 109),
        shape=(np.pi / 6, np.pi / 6, 0.5, fwhm, 38 * 0.8),
        image=img
    )
    result = optimizer.correlate(optimizer.params)
    print("Correlation Result:", result)
    corr = optimizer.quad_optimize()
    print("Optimized Parameters:", optimizer.params)
    optimizer.visualize()