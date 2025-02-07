import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from grid_struct import GridStruct
from image_utility import skeletonize_img, stereo_transform, show_hough


class HoughTM:
    def __init__(self, t0_im_loc, dt_im_loc, num_lines, ang_density=360, threshold=0.2):

        t0_im                       = cv2.imread(t0_im_loc, cv2.IMREAD_GRAYSCALE)
        dt_im                       = cv2.imread(dt_im_loc, cv2.IMREAD_GRAYSCALE)
        t0_im_thresh, t0_im_skel    = skeletonize_img(image=self.t0_im)
        dt_im_thresh, dt_im_skel    = skeletonize_img(image=self.dt_im)
        num_lines                   = num_lines
        threshold                   = threshold

        assert np.shape(t0_im) == np.shape(dt_im), "Shape of images does not match."
        im_shape                    = t0_im.shape[:2]
        test_angles                 = np.linspace(-np.pi / 2, np.pi / 2, ang_density, endpoint=True)
        pos_lines                   = np.empty((0, 2), dtype=float)  
        neg_lines                   = np.empty((0, 2), dtype=float)

    def hough_line_source(self):
        h, theta, d = hough_line(self.t0_im_skel, theta=self.test_angles)

        for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=self.threshold*h.max(),
                                                    num_peaks=self.num_lines)):
            slope = np.tan(angle + np.pi / 2) 

            if abs(slope) > 0.1:
                if slope >= 0:
                    self.pos_lines = np.vstack((self.pos_lines, [angle, dist]))
                else:
                    self.neg_lines = np.vstack((self.neg_lines, [angle, dist]))

        return None


    def solve(self):
        grid_object_skel    = GridStruct(pos_lines=pos_lines, neg_lines=neg_lines, 
                                         img=skeleton, img2=dt_skeleton,
                                         temp_scale=0.7, window_scale=1.2, search_scale=1.5)
        grid_object_img     = GridStruct(pos_lines=pos_lines, neg_lines=neg_lines, img=image, img2=dt_img,
                                    temp_scale=0.7, window_scale=1.2, search_scale=1.5)

grid_displace = np.empty(grid_object_skel.shape, dtype=object)
# show_hough(image, dt_img, adaptive_thresh, skeleton, points_arr, boolean=True)

for i in range(grid_object_skel.shape[0]):
    for j in range(grid_object_skel.shape[1]):
        if grid_object_skel.template[i, j] is not None and grid_object_skel.search_patch[i, j] is not None:
            _ , _ , template_skel               = grid_object_skel.get_template(i, j)
            _ , _ , template_img                = grid_object_img.get_template(i, j)
            x_min, y_min, search_region_skel    = grid_object_skel.get_search(i, j)
            _ , _ , search_region_img           = grid_object_img.get_search(i, j)

            w, h = template_skel.shape[::-1]

            method = cv2.TM_CCORR_NORMED
            if search_region_skel.shape[0] < template_skel.shape[0] or search_region_skel.shape[1] < template_skel.shape[1]:
                continue
            res = cv2.matchTemplate(search_region_skel, template_skel, method)

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Determine the top-left corner
            top_left = max_loc  # From cv.minMaxLoc
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2

            absolute_x = x_min + center_x
            absolute_y = y_min + center_y

            grid_displace[i, j] = np.array([absolute_x, absolute_y])