import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from skimage.transform import hough_line, hough_line_peaks
from skimage.morphology import skeletonize


class GridStruct:
    def __init__(self, pos_lines, neg_lines, img, img2):
        """
        Default constructor

        :param pos_lines :  [rho, theta] of positively sloped lines  
        :param neg_lines :  [rho, theta] of negatively sloped lines
        :param img       :  Input image (Before Transformation)
        :param img2      :  Input image (After Transformation)
        """

        def sort_lines(lines):
            return lines[np.argsort(lines[:, 1])]

        self.pos_lines = sort_lines(pos_lines)      # Shape = (11, 2)
        self.neg_lines = sort_lines(neg_lines)      # Shape = (11, 2)
        
        self.img = img
        self.img2 = img2
        self.img_shape = np.shape(self.img)         # Shape of image
        self.img2_shape = np.shape(self.img2)

        self.grid = np.empty((len(pos_lines), len(neg_lines)), dtype=object)
        self.template = np.empty((len(pos_lines), len(neg_lines)), dtype=object)
        self.search_patch = np.empty((len(pos_lines), len(neg_lines)), dtype=object)
        self.num_intersections = 0

        ### Immediately initialize and populate the data structure ###
        self.populate_grid()
        self.generate_template()
        self.generate_search_patch(window_scale=1.5,search_scale=5,alpha=1,beta=0.005)


    def _is_within_bounds(self, x, y):
        """
        Check if a point (x, y) lies within the image boundaries.

        :param x: x-coordinate
        :param y: y-coordinate

        :return: True if within bounds, False otherwise
        """
        height, width = self.img_shape
        return 0 <= x < width and 0 <= y < height
    

    def find_intersection(self, line1, line2):
        """
        Find the intersection of two lines given in rho-theta representation.

        :param line1: [rho, theta] for the first line
        :param line2: [rho, theta] for the second line

        :return: (x, y) intersection coordinates or None if lines are parallel
        """

        theta1, rho1 = line1
        theta2, rho2 = line2

        # Line equations in Cartesian form: x * cos(theta) + y * sin(theta) = rho
        A = np.array([
            [np.cos(theta1), np.sin(theta1)],
            [np.cos(theta2), np.sin(theta2)]
        ])
        b = np.array([rho1, rho2])

        try:
            # Solve the system of equations to find intersection
            x, y = np.linalg.solve(A, b)
            
            return x, y
        except np.linalg.LinAlgError:
            # Lines are parallel, no intersection
            return None


    def populate_grid(self):
        """
        Populate the grid with the intersection points of positive and negative lines.

        :return :   N/A
        """
        for i, pos_line in enumerate(self.pos_lines):
            for j, neg_line in enumerate(self.neg_lines):
                intersection = self.find_intersection(pos_line, neg_line)
                if intersection is not None:
                    x, y = intersection
                    if not self._is_within_bounds(x, y):
                        intersection = (np.nan, np.nan)
                else:
                    intersection = (np.nan, np.nan)
                
                self.grid[i, j] = intersection
                self.num_intersections += 1
    

    def generate_template(self):
        """
        Create template patches using the grid intersections and the search patch for
        the consecutive frame.

        :param image    :   The image to crop, as a NumPy array.
        """
        height, width = self.img_shape

        for i in range(len(self.pos_lines) - 1):
            for j in range(len(self.neg_lines) - 1):
                
                # Crop based on bottom right node
                center = self.grid[i, j]
                bottom_right = self.grid[i, j+1]

                # If any corner is NaN, skip cropping this region
                if (
                    np.isnan(center).any() or
                    np.isnan(bottom_right).any()
                ):
                    continue

                rect_width  = abs(bottom_right[0] - center[0])
                rect_height = abs(bottom_right[1] - center[1])
                x_center, y_center = center
                
                # Ensure the coordinates are within the image boundaries
                x_min = max(0, int(x_center - rect_width / 2))
                x_max = min(width, int(x_center + rect_width / 2))
                y_min = max(0, int(y_center - rect_height / 2))
                y_max = min(height, int(y_center + rect_height / 2))

                self.template[i, j] = self.img[y_min:y_max, x_min:x_max]


    def generate_search_patch(self, window_scale=1.5, search_scale=2, alpha=1, beta=0.01):
        """
        Create search patches for the template matching algorithm by maximizing
        similarity between self.img and self.img2 using cross-correlation and minimizing distance.

        :param window_scale :   Window scaling constant
        :param search_scale :   Search region scaling constant
        :param alpha        :   Hyper-parameter for similarity weight
        :param beta         :   Hyper-parameter for distance weight
        """

        height, width = self.img_shape
        search_height, search_width = self.img2_shape

        for i in range(len(self.pos_lines) - 1):
            for j in range(len(self.neg_lines) - 1):

                # Crop based on bottom-right node
                center = self.grid[i, j]
                bottom_right = self.grid[i, j + 1]

                # Skip cropping if any corner is NaN
                if (
                    np.isnan(center).any() or
                    np.isnan(bottom_right).any()
                ):
                    continue

                # Calculate region dimensions
                rect_width = abs(bottom_right[0] - center[0])
                rect_height = abs(bottom_right[1] - center[1])
                x_center, y_center = center

                # Ensure the coordinates are within the image boundaries
                x_min = max(0, int(x_center - window_scale * rect_width / 2))
                x_max = min(width, int(x_center + window_scale * rect_width / 2))
                y_min = max(0, int(y_center - window_scale * rect_height / 2))
                y_max = min(height, int(y_center + window_scale * rect_height / 2))

                template = self.img[y_min:y_max, x_min:x_max]

                # Define the search region center and dimensions
                search_x_min = max(0, int(x_center - search_scale * rect_width / 2))
                search_x_max = min(search_width, int(x_center + search_scale * rect_width / 2))
                search_y_min = max(0, int(y_center - search_scale * rect_height / 2))
                search_y_max = min(search_height, int(y_center + search_scale * rect_height / 2))

                # Skip if the search region is smaller than the template
                if (
                    search_x_max - search_x_min < template.shape[1] or
                    search_y_max - search_y_min < template.shape[0]
                ):
                    continue

                best_score = -float('inf')
                best_center = None

                # Perform template matching using cv2.matchTemplate
                search_region = self.img2[search_y_min:search_y_max, search_x_min:search_x_max]
                match_result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)

                # Iterate over the match result to find the best match
                for y in range(match_result.shape[0]):
                    for x in range(match_result.shape[1]):

                        # Compute the correlation score
                        correlation = match_result[y, x]

                        # Compute the distance penalty
                        candidate_center = (search_x_min + x + template.shape[1] // 2,
                                            search_y_min + y + template.shape[0] // 2)
                        distance = (candidate_center[0] - x_center) ** 2 + (candidate_center[1] - y_center) ** 2

                        # Compute energy score
                        score = alpha * correlation - beta * distance

                        # Update the best score and its center
                        if score > best_score:
                            best_score = score
                            best_center = candidate_center

                # Store the best matching center and the region bounds
                if best_center:
                    self.search_patch[i, j] = np.array([
                        best_center[0] - template.shape[1] // 2,
                        best_center[1] - template.shape[0] // 2,
                        best_center[0] + template.shape[1] // 2,
                        best_center[1] + template.shape[0] // 2
                    ])


def skeletonize_img(image):
    blur = cv2.GaussianBlur(image, (5, 5), 0).astype('uint8')

    # Apply adaptive thresholding and Otsu's thresholding
    adaptive_thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, ot = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to Eliminate Noises
    kernel = np.ones((3, 3), np.uint8)
    adaptive_thresh = cv2.erode(adaptive_thresh, kernel=kernel)
    adaptive_thresh = cv2.dilate(adaptive_thresh, kernel=kernel)

    # Create a gradient mask to isolate specific regions
    grad_x = cv2.Sobel(ot, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(ot, cv2.CV_64F, 0, 1, ksize=5)
    grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradient_mask = (grad_magnitude == 0)
    dark_mask = (ot == 0)
    mask = gradient_mask & dark_mask

    adaptive_thresh[mask] = 0
    return adaptive_thresh, skeletonize(adaptive_thresh).astype(np.uint8)

# image_location = "Template_Matching_Test/Source/frame_2.png"
image_location = "Synthetic_Data/SNR_4/Set_0/Gaussian_Grid_Image_Set_0.png"
image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
img_shape = np.shape(image)
adaptive_thresh, skeleton = skeletonize_img(image=image)

# Show the images
# fig, axes = plt.subplots(2, 3, figsize=(20, 6))
# ax = axes.ravel()

# ax[0].imshow(image, cmap=cm.gray)
# ax[0].set_title('Input image')
# ax[0].set_axis_off()

# ax[1].imshow(adaptive_thresh, cmap=cm.gray)
# ax[1].set_title('Binary')
# ax[1].set_axis_off()

# ax[2].imshow(skeleton, cmap=cm.gray)
# ax[2].set_title('Skeleton Image')
# ax[2].set_axis_off()

# ax[3].imshow(image, cmap=cm.gray)
# ax[3].set_ylim((image.shape[0], 0))
# ax[3].set_axis_off()
# ax[3].set_title('Detected lines')

# ax[4].imshow(image, cmap=cm.gray)
# ax[4].set_ylim((image.shape[0], 0))
# ax[4].set_axis_off()
# ax[4].set_title('Detected Intersections')

# Hough transform parameters
tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=True)
h, theta, d = hough_line(skeleton, theta=tested_angles)
height, width = image.shape[:2]

# Store line parameters (angle and distance)
pos_lines = np.empty((0, 2), dtype=float)  # Array to store positive slope lines
neg_lines = np.empty((0, 2), dtype=float)  # Array to store negative slope lines

threshold = 0.2 * h.max()
num_peaks = 11 * 2   #--> User Defined

# Extract lines using hough_line_peaks
for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold=threshold,
                                            num_peaks=num_peaks)):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    slope = np.tan(angle + np.pi / 2)
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

    # if len(points_in_image) >= 2:
    #     points_in_image.sort()
    #     ax[3].plot([points_in_image[0][0], points_in_image[1][0]],
    #                [points_in_image[0][1], points_in_image[1][1]], color='red')


# dt_img_loc = "Template_Matching_Test/Target/frame_2_2us.png"
dt_img_loc = "Synthetic_Data/SNR_4/Set_0/Rotational_Flow_Image_Set_0.png"
dt_img = cv2.imread(dt_img_loc, cv2.IMREAD_GRAYSCALE)
dt_adaptive_thresh, dt_skeleton = skeletonize_img(image=dt_img)

grid_object_skel = GridStruct(pos_lines=pos_lines, neg_lines=neg_lines, img=skeleton, img2=dt_skeleton)
grid_object_img = GridStruct(pos_lines=pos_lines, neg_lines=neg_lines, img=image, img2= dt_img)

# ax[5].imshow(dt_img, cmap=cm.gray)
# ax[5].set_ylim((image.shape[0], 0))
# ax[5].set_axis_off()
# ax[5].set_title('Transformed Intersections')

for i in range(len(pos_lines) - 1):
    for j in range(len(neg_lines) - 1):

        # Crop based on bottom right node
        template_skel = grid_object_skel.template[i, j]
        template_img = grid_object_img.template[i,j]

        if template_skel is not None and grid_object_skel.search_patch[i, j] is not None:
            x_min, y_min, x_max, y_max = grid_object_skel.search_patch[i, j]

            search_region_skel = dt_skeleton[int(y_min):int(y_max), int(x_min):int(x_max)]
            search_region_img = dt_img[int(y_min):int(y_max), int(x_min):int(x_max)]
            img_region = dt_img[int(y_min):int(y_max), int(x_min):int(x_max)]
            w, h = template_skel.shape[::-1]

            # Ensure the search region is large enough for the template
            if search_region_skel.shape[0] < template_img.shape[0] or search_region_skel.shape[1] < template_img.shape[1]:
                print(f"Skipping grid cell ({i}, {j}) due to size mismatch.")
                continue  # Skip regions smaller than the template

            # Apply template matching using TM_CCORR_NORMED
            method = cv2.TM_CCORR_NORMED
            res1 = cv2.matchTemplate(search_region_skel, template_skel, method)
            res2 = cv2.matchTemplate(search_region_img, template_img, method)
            res = 0.75 * res1 + 0.25 * res2

            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # Determine the top-left corner
            top_left = max_loc  # From cv.minMaxLoc
            center_x = top_left[0] + w // 2
            center_y = top_left[1] + h // 2

            absolute_x = x_min + center_x
            absolute_y = y_min + center_y

            # Show the images
            fig, axes = plt.subplots(2, 3, figsize=(20, 6))
            ax = axes.ravel()

            ax[0].imshow(image[int(y_min):int(y_max), int(x_min):int(x_max)], cmap=cm.gray)
            ax[0].set_title('Source image')
            ax[0].set_axis_off()

            ax[1].imshow(template_skel, cmap=cm.gray)
            ax[1].set_title('Template')
            ax[1].set_axis_off()

            ax[2].imshow(search_region_skel, cmap=cm.gray)
            ax[2].scatter([center_x], [center_y], c='red', s=50, label='Detected Center')
            ax[2].set_title('Search Region - Skeleton')
            ax[2].set_axis_off()

            ax[3].imshow(img_region, cmap=cm.gray)
            ax[3].scatter([center_x], [center_y], c='red', s=50, label='Detected Center')
            ax[3].set_title('Search Region - Image')
            ax[3].set_axis_off()

            ax[4].imshow(image, cmap=cm.gray)
            ax[4].set_ylim((image.shape[0], 0))
            ax[4].set_axis_off()
            ax[4].set_title('Corr - Source')

            ax[5].imshow(dt_img, cmap=cm.gray)
            ax[5].set_ylim((image.shape[0], 0))
            ax[5].set_axis_off()
            ax[5].set_title('Corr - dt(Source)')

            # Plot source point
            point = grid_object_skel.grid[i, j]
            if point is not None or point is not (np.nan, np.nan):
                ax[4].scatter(point[0], point[1], color='green', s=5)

            # Plot the intersection on the original image
            ax[5].plot(absolute_x, absolute_y, 'ro', markersize=5)

            plt.tight_layout()
            plt.show()

# plt.tight_layout()
# plt.show()