from dataclasses import dataclass
import numpy as np
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial import Delaunay


@dataclass
class dim2Interpolator:
    xy : np.ndarray
    dxy: np.ndarray
    
    method: int = 0
    radius: int = None
    extrapolate: bool = False 

    def __post_init__(self):
        assert self.xy.shape == self.dxy.shape, "xy and dxy must have the same shape"
        assert self.xy.ndim == 2, "xy must be a 2D array"
        assert self.dxy.ndim == 2, "dxy must be a 2D array"

        self.__tri = Delaunay(self.xy)

        if self.method == 0:
            self.__interpolator = [
                CloughTocher2DInterpolator(
                        self.xy,
                        self.dxy[:, i],  # i-th component of dxy
                        fill_value=np.nan
                        )
                    for i in range(self.dxy.shape[1])
                ]
        elif self.method == 1:
            if self.radius is None or self.radius < 0:
                raise "Radius must be positive integer"
            
            self.__interpolator = [ 
                # i-th component of dxy
                RBFInterpolator(
                    self.xy, 
                    self.dxy[:, i], 
                    kernel='thin_plate_spline',
                    neighbors=min(self.radius, len(self.xy)),
                    smoothing=0.01
                    )
                for i in range(self.dxy.shape[1])
                ]
            

    def interpolate(self, xy: np.ndarray) -> np.ndarray:
        # Stack results from both interpolators dx, dy
        return np.column_stack([
            interpolator(xy)
            for interpolator in self.__interpolator
        ])
    

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        # Alias for interpolate()
        return self.interpolate(points)
    

    def is_inside_bounds(self, points: np.ndarray) -> np.ndarray:
        # Check which points lie within the convex hull of the input data
        return self.__tri.find_simplex(points) >= 0