from py_import import plt, np, cv2, cm, dataclass, field, List, Tuple, Optional
from grid_struct import GridStruct


def sort_lines(lines: np.ndarray) -> np.ndarray:
    return lines[np.argsort(lines[:, 1])]


@dataclass
class TemplateMatch:
    grid_obj : GridStruct

    # Default Parameters
    temp_scale    : float = 0.67
    window_scale  : float = 1.2
    search_scale  : float = 2
    down_scale    : int = 4
    rotate_range  : int = 45

    # Calculated fields
    shifts      : Tuple[float, float]   = field(init=False)
    dt_grid     : np.ndarray            = field(init=False)
    temp_state  : np.ndarray            = field(init=False)


    def __post_init__(self):
        self.t0_grid  = np.empty(self.shape, dtype=object)
        self.dt_grid  = np.empty(self.shape, dtype=object)
        self.template = np.empty(self.shape, dtype=object)
        self.line_ang = np.empty(self.shape, dtype=object)