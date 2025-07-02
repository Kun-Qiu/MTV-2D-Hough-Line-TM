from utility.py_import import np, cv2, dataclass, field, Tuple, plt, List
from src.T0_grid_struct import T0GridStruct
from src.interpolator import dim2Interpolator
from scipy.optimize import minimize


@dataclass
class DTGridStruct:
    T0_grid    : T0GridStruct
    image_path : str

    # Default Parameters
    win_size  : Tuple[int, int] = (31, 31)
    max_level : int = 7
    iteration : int = 10
    epsilon   : float = 0.0001

    shape : Tuple[int, int] = field(init=False)
    grid  : np.ndarray = field(init=False)
    image : np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = self.T0_grid.shape
        self.grid = np.empty(self.shape, dtype=object)
        self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)

        valid_mask = np.array([[pt is not None for pt in row] for row in self.T0_grid.grid])
        valid_indices = np.where(valid_mask)
        prev_pts = np.stack(self.T0_grid.grid[valid_mask]).astype(np.float32).reshape(-1, 1, 2)
        
        self._grid_LK(prev_pts, valid_indices)
        self._dewarp_optimization(prev_pts, valid_indices, levels=0)
        # self._optimization()


    def _grid_LK(self, prev_pts: np.ndarray, valid_indices: np.ndarray) -> None:
        
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iteration, self.epsilon)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=self.T0_grid.image,
            nextImg=self.image,
            prevPts=prev_pts,   
            nextPts=None,
            winSize=self.win_size,
            maxLevel=self.max_level,
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


    def _dewarp_optimization(
            self, prev_pts: np.ndarray, valid_indices: np.ndarray, levels: int=3
            ) -> None:
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iteration, self.epsilon)
        dewarped_image = None

        for _ in range(levels):
            flow = self._interpolate_flow()
            dewarped_image = self.__dewarp(self.image, flow)

            next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                prevImg=self.T0_grid.image,
                nextImg=dewarped_image,
                prevPts=prev_pts,   
                nextPts=None,
                winSize=self.win_size,
                maxLevel=self.max_level,
                criteria=criteria
                )

            tracked_idx = 0
            for i, j in zip(*valid_indices):
                if (tracked_idx < len(status) and 
                    status[tracked_idx] and 
                    self.__within_bounds(next_pts[tracked_idx])
                    ):
                    displacement = next_pts[tracked_idx].ravel() - prev_pts[tracked_idx].ravel()
                    self.grid[i][j] += displacement
                tracked_idx += 1

        return


    def _interpolate_flow(self) -> np.ndarray:
        h, w = self.image.shape[:2]

        valid_grid ,valid_T0_grid = self.__get_valid_cells()
        valid_mask = valid_grid & valid_T0_grid
        valid_ij = np.argwhere(valid_mask)
        
        if valid_ij.size == 0:
            return np.zeros((h, w, 2), dtype=np.float32)

        src_points = self.grid[valid_ij[:, 0], valid_ij[:, 1]]
        src_points = np.vstack(src_points).astype(np.float32)
        # print(f"Shape of valid points: {np.shape(src_points)}")
        in_bounds = (
            (src_points[:, 0] >= 0) & (src_points[:, 0] < w) & \
            (src_points[:, 1] >= 0) & (src_points[:, 1] < h)
            )
        valid_ij = valid_ij[in_bounds]
        src_points = src_points[in_bounds]
            
        if src_points.size == 0:
            return np.zeros((h, w, 2), dtype=np.float32)
        if len(src_points) < 4:
            raise ValueError(f"Interpolation requires >4 points, got {len(src_points)}")

        next_points = self.T0_grid.grid[valid_ij[:, 0], valid_ij[:, 1]]
        next_points = np.vstack(next_points).astype(np.float32)
        flow_vectors = next_points - src_points
        flow = np.zeros((h, w, 2), dtype=np.float32)

        interpolator = dim2Interpolator(src_points, flow_vectors)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        query_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        interpolated_flow = interpolator.interpolate(query_points)
        flow = interpolated_flow.reshape(h, w, 2)

        return flow


    def _optimization(self) -> None:
        T0_grid_valid, dT_grid_valid = self.__get_valid_cells()
        
        valid_mask = T0_grid_valid & dT_grid_valid
        valid_indices = np.argwhere(valid_mask)

        for i, j in valid_indices:
            x1 = self.T0_grid.grid[i, j]
            x0 = self.grid[i, j]
            x_s = self.__optimizer(x0, x1)
            self.grid[i, j] = x_s
        
        return


    def __optimizer(self, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        bounds = [
            (x0[0] - 0.5, x0[0] + 0.5),  # x (±0.5 pixel)
            (x0[1] - 0.5, x0[1] + 0.5),  # y (±0.5 pixel)
            ]
        
        winSize = self.win_size
        maxLevel = self.max_level
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iteration, self.epsilon)

        def objective(pt: np.ndarray) -> float:
            try:
                pt = np.array(pt, dtype=np.float32).reshape(-1, 1, 2)
                next_pt, status, _ = cv2.calcOpticalFlowPyrLK(
                    prevImg=self.image,
                    nextImg=self.T0_grid.image, 
                    prevPts=pt,
                    nextPts=None,
                    winSize=winSize,
                    maxLevel=maxLevel,
                    criteria=criteria
                    )
                
                if status is None or status[0] != 1:
                    return np.inf
                    
                err = np.linalg.norm(next_pt - x1)
                return float(err)
            
            except Exception as e:
                print(f"Error in objective function: {e}")
                return np.inf
                
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 50, 
                    'ftol': 1e-4,
                    'gtol': 1e-4,
                    'eps': 1e-3 
                    }
                )
            if result.success:
                return result.x
            else:
                # Return initial guess if optimization fails
                return x0  
        except Exception as e:
            print(f"Error in gradient descent: {e}")
            raise


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


    def plot_convex_hull(self):
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        magnitude = np.sqrt(self.flow[...,0]**2 + self.flow[...,1]**2)
        masked_magnitude = np.ma.masked_invalid(magnitude)
        plt.imshow(masked_magnitude, cmap='hot', vmin=np.nanmin(magnitude), vmax=np.nanmax(magnitude))
        plt.colorbar(label='Flow Magnitude (pixels)')
        plt.title("Optical Flow Magnitude")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        convex = self.__img_convex_hull(self.T0_grid.image, 0)
        plt.imshow(convex, cmap='gray')
        plt.title("Convex Hull of T0 Image")

        plt.subplot(1, 3, 3)
        convex = self.__img_convex_hull(self.image, 1)
        plt.imshow(convex, cmap='gray')
        plt.title("Convex Hull of dT image")
        plt.tight_layout()
        plt.show()

    
    ########## Private Helper Functions ##########

    def __get_valid_cells(self) -> Tuple[np.ndarray, np.ndarray]:
        T0_grid_valid = np.array(
            [[cell is not None for cell in row] 
            for row in self.T0_grid.grid]
            )
        
        dT_grid_valid = np.array(
            [[cell is not None for cell in row] 
            for row in self.grid]
            )

        return T0_grid_valid, dT_grid_valid
    
    
    def __within_bounds(self, pt: np.ndarray) -> bool:
        x, y = pt.ravel()
        height, width = self.image.shape[:2]
        return 0 <= x < width and 0 <= y < height
    

    @staticmethod
    def __dewarp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        remap_x = (x - flow[..., 0]).astype(np.float32)
        remap_y = (y - flow[..., 1]).astype(np.float32)
        dewarped_image = cv2.remap(
            img, remap_x, remap_y,
            interpolation=cv2.INTER_CUBIC 
            )
        
        return dewarped_image
    

    def __img_convex_hull(
            self, img: np.ndarray, 
            mode: int, dilate_win_size: Tuple[int, int]=None
            ) -> np.ndarray:
        
        h, w = img.shape[:2]
        img_hull = np.zeros((h, w), dtype=np.float32)

        valid_mask = None
        if mode == 0:
            valid_mask, _ = self.__get_valid_cells()
        elif mode == 1:
            _, valid_mask = self.__get_valid_cells()

        valid_ij = np.argwhere(valid_mask)
        if valid_ij.size == 0:
            return img_hull

        grid = self.grid if mode == 1 else self.T0_grid.grid
        src_points = grid[valid_ij[:, 0], valid_ij[:, 1]]
        src_points = np.vstack(src_points).astype(np.float32)

        pixel_coords = np.round(src_points).astype(np.int32)
        pixel_coords = pixel_coords[
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h)
            ]

        if pixel_coords.shape[0] < 3:
            return img_hull 

        # Compute convex hull
        hull = cv2.convexHull(pixel_coords)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull, 1)

        if dilate_win_size is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, dilate_win_size)
            mask = cv2.dilate(mask, kernel, iterations=1)
        img_hull[mask == 1] = img[mask == 1]

        return img_hull