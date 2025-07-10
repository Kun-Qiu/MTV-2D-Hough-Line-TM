from utility.py_import import np, dataclass
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial import Delaunay


@dataclass
class dim2Interpolator:
    xy : np.ndarray
    dxy: np.ndarray
    ########################################################### 
    ## 0 for RBFInterpolator (Extrapolation + Interpolation) ##
    ## 1 for CloughTocher2DInterpolator (Interpolation only) ##
    ###########################################################
    extrapolate: bool = False 

    def __post_init__(self):
        assert self.xy.shape == self.dxy.shape, "xy and dxy must have the same shape"
        assert self.xy.ndim == 2, "xy must be a 2D array"
        assert self.dxy.ndim == 2, "dxy must be a 2D array"

        self.tri = Delaunay(self.xy)

        if self.extrapolate:
            self.interpolator = [ # i-th component of dxy
                RBFInterpolator(self.xy, self.dxy[:, i], kernel='thin_plate_spline')
                for i in range(self.dxy.shape[1])
                ]
        else:
            self.interpolator = [
                CloughTocher2DInterpolator(
                    self.xy,
                    self.dxy[:, i],  # i-th component of dxy
                    fill_value=np.nan
                    )
                for i in range(self.dxy.shape[1])
                ]
            

    def interpolate(self, xy: np.ndarray) -> np.ndarray:
        # Stack results from both interpolators dx, dy
        return np.column_stack([
            interpolator(xy)
            for interpolator in self.interpolator
        ])
    

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        # Alias for interpolate()
        return self.interpolate(points)
    

    def is_inside_bounds(self, points: np.ndarray) -> np.ndarray:
        # Check which points lie within the convex hull of the input data
        return self.tri.find_simplex(points) >= 0