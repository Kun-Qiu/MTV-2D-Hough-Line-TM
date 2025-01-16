import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from grid_struct import GridStruct
from image_utility import skeletonize_img, stereo_transform


def show_hough(image, dt_img, adaptive_thresh, skeleton, points, 
                grid_struct=None, grid_displace=None, boolean=False):
    # Show the images
    fig, axes = plt.subplots(2, 3, figsize=(20, 6))
    ax = axes.ravel()

    ax[0].imshow(image, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    ax[1].imshow(adaptive_thresh, cmap=cm.gray)
    ax[1].set_title('Binary')
    ax[1].set_axis_off()

    ax[2].imshow(skeleton, cmap=cm.gray)
    ax[2].set_title('Skeleton Image')
    ax[2].set_axis_off()

    ax[3].imshow(image, cmap=cm.gray)
    ax[3].set_ylim((image.shape[0], 0))
    ax[3].set_axis_off()
    ax[3].set_title('Detected lines')

    ax[4].imshow(image, cmap=cm.gray)
    ax[4].set_ylim((image.shape[0], 0))
    ax[4].set_axis_off()
    ax[4].set_title('Source Intersections')

    ax[5].imshow(dt_img, cmap=cm.gray)
    ax[5].set_ylim((dt_img.shape[0], 0))
    ax[5].set_axis_off()
    ax[5].set_title('Target Intersections')

    for point in points:
        if len(point) >= 2:
            point.sort()
            ax[3].plot([point[0][0], point[1][0]],
                    [point[0][1], point[1][1]], color='red')
    
    if grid_struct is not None and grid_displace is not None:
        for i in range(grid_struct.shape[0]):
            for j in range(grid_struct.shape[1]):
                x, y = grid_struct[i, j]
                ax[4].scatter(x, y, s=1, color='red')

                # print(grid_displace)
                if grid_displace[i,j] is None:
                    continue
                
                x_new, y_new = grid_displace[i, j]
                ax[5].scatter(x_new, y_new, s=1, color='red')

    if boolean:
        plt.tight_layout()
        plt.show()

image_location = "Experimental_Data/Source/frame_1.png"
# image_location = "Synthetic_Data/SNR_2/Set_0/Gaussian_Grid_Image_Set_0.png"
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
image = stereo_transform(image)

img_shape = np.shape(image)
adaptive_thresh, skeleton = skeletonize_img(image=image)

# Hough transform parameters
# tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=True)
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360 * 10, endpoint=True)
h, theta, d = hough_line(skeleton, theta=tested_angles)
height, width = image.shape[:2]

# Store line parameters (angle and distance)
pos_lines = np.empty((0, 2), dtype=float)  # Array to store positive slope lines
neg_lines = np.empty((0, 2), dtype=float)  # Array to store negative slope lines

threshold = 0.2 * h.max()
num_peaks = 11 * 2   #--> User Defined
points_arr = []

# Extract lines using hough_line_peaks
for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=threshold,
                                            num_peaks=num_peaks)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope = np.tan(angle + np.pi / 2)
    if abs(slope) > 0.1:
        if slope >= 0:
            pos_lines = np.vstack((pos_lines, [angle, dist]))
        else:
            neg_lines = np.vstack((neg_lines, [angle, dist]))

        # borders = [(0, 0), (width, 0), (0, height), (width, height)]
        x_left, x_right = 0, width
        y_left = slope * (x_left - x0) + y0
        y_right = slope * (x_right - x0) + y0
        y_top, y_bottom = 0, height
        x_top = (y_top - y0) / slope + x0 if slope != 0 else np.inf
        x_bottom = (y_bottom - y0) / slope + x0 if slope != 0 else np.inf

        points = [(x_left, y_left), (x_right, y_right), (x_top, y_top), (x_bottom, y_bottom)]
        points_in_image = [(x, y) for x, y in points if 0 <= x <= width and 0 <= y <= height]
        points_arr.append(points_in_image)


dt_img_loc = "Experimental_Data/Target/frame_1_2us.png"
# dt_img_loc = "Synthetic_Data/SNR_2/Set_0/Translational_Flow_Image_Set_0.png"
dt_img = cv2.imread(dt_img_loc, cv2.IMREAD_GRAYSCALE)
dt_img = stereo_transform(dt_img)
dt_adaptive_thresh, dt_skeleton = skeletonize_img(image=dt_img)

grid_object_skel    = GridStruct(pos_lines=pos_lines, neg_lines=neg_lines, img=skeleton, img2=dt_skeleton,
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

            # # Show the images
            # fig, axes = plt.subplots(2, 3, figsize=(20, 6))
            # ax = axes.ravel()

            # ax[0].imshow(template_img, cmap=cm.gray)
            # ax[0].set_title('Template')
            # ax[0].set_axis_off()

            # ax[1].imshow(res, cmap=cm.jet)
            # ax[1].set_title('Cross Corrolation')
            # ax[1].set_axis_off()

            # ax[2].imshow(search_region_skel, cmap=cm.gray)
            # ax[2].scatter([center_x], [center_y], c='red', s=5, label='Detected Center')
            # ax[2].set_title('Search Region - Skeleton')
            # ax[2].set_axis_off()

            # ax[3].imshow(search_region_img, cmap=cm.gray)
            # ax[3].scatter([center_x], [center_y], c='red', s=5, label='Detected Center')
            # ax[3].set_title('Search Region - Image')
            # ax[3].set_axis_off()

            # ax[4].imshow(image, cmap=cm.gray)
            # ax[4].set_ylim((image.shape[0], 0))
            # ax[4].set_axis_off()
            # ax[4].set_title('Corr - Source')

            # ax[5].imshow(dt_img, cmap=cm.gray)
            # ax[5].set_ylim((image.shape[0], 0))
            # ax[5].set_axis_off()
            # ax[5].set_title('Corr - dt(Source)')

            # # Plot source point
            # point = grid_object_skel.grid[i, j]
            # if point is not None or point is not (np.nan, np.nan):
            #     ax[4].scatter(point[0], point[1], color='green', s=5)

            # # Plot the intersection on the original image
            # ax[5].plot(absolute_x, absolute_y, 'ro', markersize=5)

            # plt.tight_layout()
            # plt.show()

show_hough(image, dt_img, adaptive_thresh, skeleton, points_arr, 
           grid_object_skel.grid, grid_displace, boolean=True)