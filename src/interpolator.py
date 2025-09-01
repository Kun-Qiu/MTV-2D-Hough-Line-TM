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
    jitter: float = 1e-6
    extrapolate: bool = False 

    def __post_init__(self):
        assert self.xy.shape == self.dxy.shape, "xy and dxy must have the same shape"
        assert self.xy.ndim == 2, "xy must be a 2D array"
        assert self.dxy.ndim == 2, "dxy must be a 2D array"

        # rng = np.random.default_rng(42)  # For reproducibility
        # self.xy_jittered = self.xy + rng.normal(0, self.jitter, self.xy.shape)

        self.__tri = Delaunay(self.xy)
        self.__interpolator = [
             CloughTocher2DInterpolator(
                    self.xy,
                    self.dxy[:, i],  # i-th component of dxy
                    fill_value=np.nan
                    )
                for i in range(self.dxy.shape[1])
            ]

        # self.__interpolator = [ # i-th component of dxy
        #     RBFInterpolator(
        #         self.xy, 
        #         self.dxy[:, i], 
        #         kernel='thin_plate_spline',
        #         neighbors=min(30, len(self.xy)),
        #         smoothing=0.01
        #         )
        #     for i in range(self.dxy.shape[1])
        #     ]
            

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