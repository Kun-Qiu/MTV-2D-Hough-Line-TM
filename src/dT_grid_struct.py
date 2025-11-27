from dataclasses import dataclass, field
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.T0_grid_struct import T0GridStruct
from src.interpolator import dim2Interpolator
from src.img_enhance import SingleShotEnhancer


@dataclass
class DTGridStruct:
    T0_grid  : T0GridStruct
    image    : np.ndarray
    
    avg_image: np.ndarray = None
    win_size : Tuple[int, int] = (31, 31)
    max_level: int = 5
    iteration: int = 10
    epsilon  : float = 0.0001

    shape: Tuple[int, int] = field(init=False)
    grid : np.ndarray = field(init=False)

    def __post_init__(self):
        self.shape = self.T0_grid.num_lines
        self.grid = np.empty(self.shape, dtype=object)

        if self.avg_image is not None:
            self.image = self.__filter_img(self.image, self.avg_image)

        valid_mask = np.array([[pt is not None for pt in row] for row in self.T0_grid.grid])
        valid_indices = np.where(valid_mask)
        prev_pts = np.stack(self.T0_grid.grid[valid_mask]).astype(np.float32).reshape(-1, 1, 2)

        # Initial Frame
        self._grid_LK(
            prev_img=self.T0_grid.image, next_img=self.image, 
            prev_pts=prev_pts, valid_indices=valid_indices
            )


    def _grid_LK(self, prev_img: np.ndarray, next_img: np.ndarray, 
                 prev_pts: np.ndarray, valid_indices: np.ndarray) -> None:
        # Perform LK optical flow to track points from T0_grid to dT_grid
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.iteration, self.epsilon)

        # Forward Lucas Kanade
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prevImg=prev_img,
            nextImg=next_img,
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


    def sequence_solver(self, single_sequence: list, avg_sequence: list = None) -> None:
        prev_img = self.image
        next_img = None

        if avg_sequence is None:
            avg_sequence = [None] * len(single_sequence)

        for single_frame, avg_frame in zip(single_sequence, avg_sequence):
            if avg_frame is not None:
                next_img = self.__filter_img(single_frame, avg_frame)

            valid_mask = np.array([[pt is not None for pt in row] for row in self.grid])
            valid_indices = np.where(valid_mask)
            prev_pts = np.stack(self.grid[valid_mask]).astype(np.float32).reshape(-1, 1, 2) 

            self._grid_LK(
                prev_img=prev_img, next_img=next_img, 
                prev_pts=prev_pts, valid_indices=valid_indices
                )
            prev_img = next_img

        self.image = next_img
        return


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
    

    def _interpolate_flow(self) -> np.ndarray:
        h, w = self.image.shape[:2]

        valid_grid, valid_T0_grid = self.__get_valid_cells()
        valid_mask = valid_grid & valid_T0_grid
        valid_ij = np.argwhere(valid_mask)
        
        if valid_ij.size == 0:
            return np.zeros((h, w, 2), dtype=np.float32)

        src_points = self.T0_grid.grid[valid_ij[:, 0], valid_ij[:, 1]]
        src_points = np.vstack(src_points).astype(np.float32)
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

        next_points = self.grid[valid_ij[:, 0], valid_ij[:, 1]]
        next_points = np.vstack(next_points).astype(np.float32)
        flow_vectors = next_points - src_points
        flow = np.zeros((h, w, 2), dtype=np.float32)

        interpolator = dim2Interpolator(src_points, flow_vectors)
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        query_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        interpolated_flow = interpolator.interpolate(query_points)
        flow = interpolated_flow.reshape(h, w, 2)

        return flow
    

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
    

    def __filter_img(self, image: np.ndarray, avg_image: np.ndarray) -> np.ndarray:
        enhancer_source = SingleShotEnhancer(avg_shot=avg_image, single_shot=image)
        filter_img = enhancer_source.filter()

        return filter_img