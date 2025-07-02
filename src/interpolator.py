from utility.py_import import np, dataclass
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import RBFInterpolator
from scipy.spatial import Delaunay


@dataclass
class dim2Interpolator:
    xy  : np.ndarray
    dxy : np.ndarray

    def __post_init__(self):
        assert self.xy.shape == self.dxy.shape, "xy and dxy must have the same shape"
        assert self.xy.ndim == 2, "xy must be a 2D array"
        assert self.dxy.ndim == 2, "dxy must be a 2D array"

        self.tri = Delaunay(self.xy)
        self.interpolators = [
            RBFInterpolator(self.xy, self.dxy[:, i], kernel='thin_plate_spline')
            for i in range(self.dxy.shape[1])
            ]

        # Only for Interpolation and no extrapolation
        # Uncomment the following lines to use CloughTocher2DInterpolator instead
        
        # self.interpolators = [
        #     CloughTocher2DInterpolator(
        #         self.xy,
        #         self.dxy[:, i],  # i-th component of dxy
        #         fill_value=0
        #         )
        #     for i in range(self.dxy.shape[1])
        #     ]
        
    def interpolate(self, xy: np.ndarray) -> np.ndarray:
        # Stack results from both interpolators dx, dy
        return np.column_stack([
            interpolator(xy)
            for interpolator in self.interpolators
        ])
    

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        # Alias for interpolate()
        return self.interpolate(points)
    

    def is_inside_bounds(self, points: np.ndarray) -> np.ndarray:
        # Check which points lie within the convex 
        # hull of the input data.
        return self.tri.find_simplex(points) >= 0