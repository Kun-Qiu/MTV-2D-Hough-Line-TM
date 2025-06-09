from utility.py_import import plt, np, convolve2d, dataclass, field, List, Tuple, Optional

@dataclass
class ParametricX:
    center: Tuple[float, float]
    shape : Tuple[float, float, float, float, float]
    image : Optional[np.ndarray]

    params: List[float] = field(init=False)

    def __post_init__(self):
        self._validate_inputs()
        self.params = np.array([
            self.center[0], self.center[1],  # x, y
            self.shape[0] , self.shape[1],   # ang1, ang2
            self.shape[2] , self.shape[3],   # rel_intens, lin_wid
            self.shape[4]                    # leg_len
        ])

    
    @staticmethod
    def _rotation_matrix(angle: float, counter_clockwise:bool = False) -> np.ndarray:
        if counter_clockwise:
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
    

    def _validate_inputs(self) -> None:
        if not isinstance(self.center, tuple) or len(self.center) != 2:
            raise ValueError("center must be a tuple of (x, y) coordinates")
        if not isinstance(self.shape, tuple) or len(self.shape) != 5:
            raise ValueError("shape must be a tuple of (ang1, ang2, rel_intens, lin_wid, leg_len)")
        if not (-2*np.pi <= self.shape[0] <= 2*np.pi):
            raise ValueError("ang1 must be between -2π and 2π")
        if not (-2*np.pi <= self.shape[1] <= 2*np.pi):
            raise ValueError("ang2 must be between -2π and 2π")
        if not (0 <= self.shape[2] <= 1):
            raise ValueError("rel_intens must be between 0 and 1")
        if self.shape[3] <= 0:
            raise ValueError("lin_wid must be positive")
        if self.shape[4] <= 0:
            raise ValueError("leg_len must be positive")
        

    def _parametric_template(self, params: List[float] = None) -> Tuple[np.ndarray, Tuple[int, int]]:
        if params is None:
            params = self.params

        x, y, ang1, ang2, rel_intens, lin_wid, leg_len = params
        half_leg_len = int(round(leg_len / 2))
        
        x_vals = np.arange(-half_leg_len, half_leg_len)
        y_vals = np.arange(-half_leg_len, half_leg_len)
        xx, yy = np.meshgrid(x_vals, y_vals)
        
        rot1, rot2 = self._rotation_matrix(ang1), self._rotation_matrix(ang2, counter_clockwise=True)
        
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
    

    def correlate(self, params: List[float]) -> dict:
        result = {
            'correlation': -np.inf,
            'background': 0.0,
            'noise': np.nan,
            'difference': None
        }

        template, (min_col, min_row) = self._parametric_template(params)
        t_height, t_width = template.shape
        
        img_patch = self.image[
            min_row:min_row + t_height, 
            min_col:min_col + t_width
            ]
        
        if img_patch.shape != template.shape:
            return result
        
        # Normalization of template and image patch
        template_mean, template_std = np.mean(template), np.std(template)
        img_mean, img_std = np.mean(img_patch), np.std(img_patch)
        template_norm = (template - template_mean) / (template_std + 1e-9)
        img_norm = (img_patch - img_mean) / (img_std + 1e-9)

        # scaled_diff = (img_norm - template_norm) * img_std
        corr_coef = np.corrcoef(template_norm.flatten(), img_norm.flatten())[0, 1]

        if np.isnan(corr_coef):
            corr_coef = -np.inf
        result['correlation'] = corr_coef
        # result['background']  = ((1 - template_mean) / template_std * img_std) + img_mean
        # result['difference']  = scaled_diff

        # if scaled_diff.shape[0] > 2 and scaled_diff.shape[1] > 2:
        #     kernel = np.ones((3, 3)) / 9
        #     local_mean = convolve2d(scaled_diff, kernel, mode='valid')
        #     noise = scaled_diff[1:-1, 1:-1] - local_mean
        #     result['noise'] = 5 * np.std(noise)
        return result
    

    def get_parametric_X(self, params: List[float] = None) -> np.ndarray:
        return self._parametric_template(params)
    

    def visualize(self) -> None:    
        template, _ = self._parametric_template()
        
        plt.figure(figsize=(10, 5))
        plt.imshow(template, cmap='hot', alpha=0.5)
        plt.title('Parametric X Template')
        plt.axis('off') 
        
        plt.tight_layout()
        plt.show()
