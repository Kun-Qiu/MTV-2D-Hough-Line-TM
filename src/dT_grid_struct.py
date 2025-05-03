from utility.py_import import np, cv2, dataclass, field, Tuple
from utility.image_utility import skeletonize_img, transform_image
from src.T0_grid_struct import T0GridStruct
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale

@dataclass
class DTGridStruct:
    T0_grid     : T0GridStruct
    image_path  : str

    # Default Parameters
    down_scale    : int = 4
    window_scale  : float = 1.2
    search_scale  : float = 2
    down_scale    : int = 4
    rotate_range  : int = 45

    shape       : Tuple[int, int] = field(init=False)
    grid        : np.ndarray = field(init=False)
    params      : np.ndarray = field(init=False)
    image       : np.ndarray = field(init=False)
    image_skel  : np.ndarray = field(init=False)
    shifts      : Tuple[float, float] = field(init=False)

    def __post_init__(self):
        self.shape = self.T0_grid.shape
        self.grid = np.empty(
            self.shape, dtype=object
            )
        
        self.image = cv2.imread(
            self.image_path, cv2.IMREAD_GRAYSCALE
            )
        _, self.image_skel = skeletonize_img(self.image)
        self.params = self.T0_grid.params.copy()
        
        # Shift from moving image to reference image
        self.shifts, _, _ = phase_cross_correlation(
            rescale(self.T0_grid.image_skel, 1 / self.down_scale, anti_aliasing=True), 
            rescale(self.image_skel, 1 / self.down_scale, anti_aliasing=True)
            )
        self.shifts *= self.down_scale
        
        ############################
        ### Initialize the grids ###
        ############################
        self._populate_grid(self.window_scale, self.search_scale)


    def _populate_grid(self, window_scale: float=1.2, search_scale: int=2) -> None:
        """
        Create search patches for the template matching algorithm by maximizing
        similarity between self.reference_img and self.image_skel
        """
        
        assert window_scale >= 1, "window_scale must be greater than or equal to 1"
        assert search_scale >= 2, "search_scale must be greater than or equal to 2"

        warped_search_im = transform_image(
            self.image_skel, 
            self.shifts[1], 
            self.shifts[0]
            )
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                
                center = self.T0_grid.grid[i, j]
                if np.isnan(center).any() or np.any((self.T0_grid.template[i, j]) is None):
                    continue

                x_min, y_min, x_max, y_max = self.T0_grid.template[i, j]
                rect_width                 = abs((x_max - x_min))
                rect_height                = abs((y_max - y_min))
                x_center, y_center         = center
                bound_y, bound_x           = np.shape(self.T0_grid.image_skel)

                def get_bound(x_c, y_c, width, height, scale, bound_width, bound_height):
                    bound_x_min = max(0, int(x_c - (scale * width) / 2))
                    bound_x_max = min(bound_width, int(x_c + (scale * width) / 2))
                    bound_y_min = max(0, int(y_c - (scale * height) / 2))
                    bound_y_max = min(bound_height, int(y_c + (scale * height) / 2))

                    return np.array([bound_x_min, bound_y_min, bound_x_max, bound_y_max])
                
                temp_x_min, temp_y_min, temp_x_max, temp_y_max = get_bound(
                    x_center, y_center, 
                    rect_width, rect_height, 
                    window_scale, 
                    bound_x, bound_y
                    )
                
                search_x_min, search_y_min, search_x_max, search_y_max = get_bound(
                    x_center, y_center, 
                    rect_width, rect_height, 
                    search_scale, 
                    bound_x, bound_y
                    )
                
                template = self.T0_grid.image_skel[
                    temp_y_min:temp_y_max, temp_x_min:temp_x_max
                    ]
                
                search_region_warped = warped_search_im[
                    search_y_min:search_y_max, search_x_min:search_x_max
                    ]

                opt_score = -np.inf
                opt_loc   = None

                for angle in range(-self.rotate_range, self.rotate_range, 5):
                    rotate_center = (template.shape[1] // 2, template.shape[0] // 2)
                    rot_mat = cv2.getRotationMatrix2D(
                        rotate_center, angle, scale=1
                        )
                    
                    rotate_dst = cv2.warpAffine(
                        template, rot_mat, (template.shape[1], template.shape[0])
                        )    
                    
                    match_result = cv2.matchTemplate(
                        search_region_warped, rotate_dst, cv2.TM_CCOEFF_NORMED
                        )
                    
                    _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
                    
                    if max_val > opt_score:
                        opt_score = max_val
                        opt_loc   = max_loc

                x_opt, y_opt = opt_loc
                x_opt = search_x_min + x_opt + template.shape[1] / 2 - self.shifts[1]
                y_opt = search_y_min + y_opt + template.shape[0] / 2 - self.shifts[0]
                self.grid[i, j] = np.array([x_opt, y_opt])

        return None
    