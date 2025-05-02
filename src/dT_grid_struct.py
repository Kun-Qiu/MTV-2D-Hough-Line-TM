from src.py_import import np, cv2, dataclass, field, Tuple
from src.image_utility import skeletonize_img
from src.T0_grid_struct import T0GridStruct

@dataclass
class DTGridStruct:
    T0_grid     : T0GridStruct
    image_path  : str

    # Default Parameters
    temp_scale    : float = 0.67
    window_scale  : float = 1.2
    search_scale  : float = 2
    down_scale    : int = 4
    rotate_range  : int = 45

    grid        : np.ndarray = field(init=False)
    image       : np.ndarray = field(init=False)
    image_skel  : np.ndarray = field(init=False)

    def __post_init__(self):
        self.grid = np.empty(T0GridStruct.shape, dtype=object)
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        _, self.image_skel = skeletonize_img(self.image)
        self.grid  = np.empty(T0GridStruct.shape, dtype=object)
        self.template = np.empty(T0GridStruct.shape, dtype=object)
    