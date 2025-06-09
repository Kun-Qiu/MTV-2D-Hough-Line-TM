from utility.py_import import np, cv2, dataclass, field, Tuple, plt, List
from src.T0_grid_struct import T0GridStruct

@dataclass
class DTGridStruct:
    T0_grid    : T0GridStruct
    image_path : str

    # Default Parameters
    win_size  : Tuple[int, int] = (31, 31)
    max_level : int = 5
    iteration : int = 10
    epsilon   : float = 0.03

    shape  : Tuple[int, int] = field(init=False)
    grid   : np.ndarray = field(init=False)
    params : np.ndarray = field(init=False)
    image  : np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = self.T0_grid.shape
        self.params = self.T0_grid.params.copy()

        self.grid = np.empty(
            self.shape, dtype=object
            )
        
        self.image = cv2.imread(
            self.image_path, cv2.IMREAD_GRAYSCALE
            )
        
        ############################
        ### Initialize the grids ###
        ############################
        self._populate_grid_LK(
            self.win_size, self.max_level, 
            self.iteration, self.epsilon
            )
        
    
    def _populate_grid_LK(
        self, winSize: Tuple[int, int] = (31, 31), maxLevel: int = 7,
        iterations: int = 10, epsilon: float = 0.0001
        ) -> None:

        valid_mask = np.array([[pt is not None for pt in row] for row in self.T0_grid.grid])
        valid_indices = np.where(valid_mask)
        prev_pts = np.stack(self.T0_grid.grid[valid_mask]).astype(np.float32).reshape(-1, 1, 2)

        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=self.T0_grid.image,
            nextImg=self.image,
            prevPts=prev_pts,   
            nextPts=None,
            winSize=winSize,
            maxLevel=maxLevel,
            criteria=criteria
            )

        tracked_idx = 0
        for i, j in zip(*valid_indices):
            if (tracked_idx < len(status) and 
                status[tracked_idx] and 
                self.__within_bounds(next_pts[tracked_idx])
                ):

                self.grid[i][j] = next_pts[tracked_idx].ravel()
            else:
                self.grid[i][j] = None
            tracked_idx += 1

        return 


    def _farneback_refinement(self, pts: np.ndarray, params: dict = None) -> np.ndarray:
        pts = pts.reshape(-1, 2)
        if params is None:
            params = {
                'pyr_scale': 0.5,
                'levels': 5,
                'winsize': 15,
                'iterations': 5,
                'poly_n': 7,
                'poly_sigma': 1.2,
                'flags': cv2.OPTFLOW_FARNEBACK_GAUSSIAN
                }
        
        flow = cv2.calcOpticalFlowFarneback(
            prev=self.T0_grid.image,
            next=self.image,
            flow=None,
            **params
            )

        refined_pts = []
        for x, y in pts:
            if np.any(np.isnan([x, y])):
                refined_pts.append([np.nan, np.nan])
                continue
                
            if 0 <= y < flow.shape[0] and 0 <= x < flow.shape[1]:
                dx = cv2.getRectSubPix(flow[..., 0], (1, 1), (x, y))[0, 0]
                dy = cv2.getRectSubPix(flow[..., 1], (1, 1), (x, y))[0, 0]
                refined_pts.append([x + dx, y + dy])
            else:
                refined_pts.append([np.nan, np.nan])
        
        return np.array(refined_pts)


    def __within_bounds(self, pt: np.ndarray) -> bool:
        x, y = pt.ravel()
        height, width = self.image.shape[:2]
        return 0 <= x < width and 0 <= y < height
    

    def visualize(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(self.image, cmap='gray')
        
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                if self.grid[i, j] is not None:
                    x, y = self.grid[i, j]
                    ax.scatter(x, y, c='red', s=10, marker='o')
        
        plt.title("Displacement Grid Visualization")
        plt.show()