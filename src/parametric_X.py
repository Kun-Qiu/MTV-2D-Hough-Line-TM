from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class ParametricX:
    center      : Tuple[float, float]
    shape       : Tuple[float, float, float, float, float]
    img_shape   : Tuple[float, float]

    params : List[float] = field(init=False)

    def __post_init__(self):
        self._validate_inputs()
        self.params = np.array([
            self.center[0], self.center[1],  # x, y
            self.shape[0] , self.shape[1],   # ang1, ang2
            self.shape[2] , self.shape[3],   # rel_intens, lin_wid
            self.shape[4]                    # leg_len
        ])

    
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
        
        min_col = int(np.clip(x - half_leg_len, 0, self.img_shape[1]))
        min_row = int(np.clip(y - half_leg_len, 0, self.img_shape[0]))
        
        return template, (min_col, min_row)
