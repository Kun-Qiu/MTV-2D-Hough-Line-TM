import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.Synthetic_Data.grid_image_generator import *
from data.Synthetic_Data.displace_image_generator import *
from data.Synthetic_Data.flow_pattern import poiseuille_flow


# def test_classic_pyramid(intersections):
#     displacement_range = np.arange(0, 51, 5) 
#     success_rates_no_pyr = []
#     success_rates_5_pyr = []

#     for disp_mag in displacement_range:
#         src = create_centered_grid(
#             image_size, fwhm, spacing, angle, 
#             line_intensity=0.5, num_lines=num_lines, snr=snr
#         ).astype(np.uint8)  
        
#         flow_field = poiseuille_flow(disp_mag, shape=src.shape)
#         ground_truth = np.zeros_like(intersections)
#         for idx in range(len(intersections)):  
#             x, y = intersections[idx, 0, :]
#             xi, yi = int(round(x)), int(round(y))
#             if 0 <= yi < 256 and 0 <= xi < 256:
#                 ground_truth[idx] = intersections[idx] + flow_field[yi, xi]
#             else:
#                 ground_truth[idx] = [np.nan, np.nan]
        
#         displaced_img = displace_image(src, flow_field).astype(np.uint8)  # Convert to uint8
#         criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
#         no_level, status_no, _ = cv2.calcOpticalFlowPyrLK(
#             prevImg=src, 
#             nextImg=displaced_img, 
#             prevPts=intersections, 
#             nextPts=None, 
#             winSize=(15, 15), 
#             maxLevel=0,  
#             criteria=criteria
#         )
        
#         five_level, status_five, _ = cv2.calcOpticalFlowPyrLK(
#             prevImg=src, 
#             nextImg=displaced_img, 
#             prevPts=intersections, 
#             nextPts=None, 
#             winSize=(15, 15), 
#             maxLevel=5,  # 5 pyramid levels
#             criteria=criteria
#         )

        
#         errors_no = np.linalg.norm(no_level - ground_truth, axis=2)
#         errors_5 = np.linalg.norm(five_level - ground_truth, axis=2)
        
#         success_no = np.mean(errors_no < 2) * 100
#         success_5 = np.mean(errors_5 < 2) * 100
            
#         success_rates_no_pyr.append(success_no)
#         success_rates_5_pyr.append(success_5)

#     # Create streamlined research figure
#     plt.figure(figsize=(8, 5), dpi=100)  # Reduced figure size
#     ax = plt.gca()

#     ax.plot(displacement_range, success_rates_no_pyr, 
#             color='#D62728', marker='o', markersize=6, 
#             linewidth=1.5, linestyle='--', label='Standard LK')

#     ax.plot(displacement_range, success_rates_5_pyr, 
#             color='#1F77B4', marker='^', markersize=6, 
#             linewidth=1.5, linestyle='-', label='Pyramidal LK')

#     max_gap_idx = np.argmax(np.array(success_rates_5_pyr) - np.array(success_rates_no_pyr))
#     x = displacement_range[max_gap_idx]

#     # Minimalist formatting
#     ax.set_xlabel('Displacement (pixels)', fontsize=10)
#     ax.set_ylabel('Accuracy (% <2px)', fontsize=10)

#     # Clean axes setup
#     ax.set_xlim(0, 50)
#     ax.set_ylim(0, 105)
#     ax.set_xticks(range(0, 51, 10))  # Fewer ticks
#     ax.set_yticks(range(0, 101, 20))
#     ax.grid(False)

#     # Efficient legend
#     ax.legend(loc='lower left', frameon=False, fontsize=8)

#     plt.tight_layout()
#     plt.show()

def test_classic_pyramid(intersections):
    displacement_range = np.arange(0, 51, 5)
    window_sizes = [10, 15, 20] 
    pyramid_levels = [0, 3, 5, 8]  
    
    # Prepare results storage
    results = np.zeros((len(window_sizes), len(pyramid_levels), len(displacement_range)))
    
    for i, w_x in enumerate(window_sizes):
        win_size = 2 * w_x + 1
        for j, max_level in enumerate(pyramid_levels):
            for k, disp_mag in enumerate(displacement_range):
                # Create test images and ground truth
                src = create_centered_grid(
                    image_size, fwhm, spacing, angle, 
                    line_intensity=0.5, num_lines=num_lines, snr=snr
                ).astype(np.uint8)
                
                flow_field = poiseuille_flow(disp_mag, shape=src.shape)
                ground_truth = np.zeros_like(intersections)
                for idx in range(len(intersections)):  
                    x, y = intersections[idx, 0, :]
                    xi, yi = int(round(x)), int(round(y))
                    if 0 <= yi < 256 and 0 <= xi < 256:
                        ground_truth[idx] = intersections[idx] + flow_field[yi, xi]
                    else:
                        ground_truth[idx] = [np.nan, np.nan]
                
                displaced_img = displace_image(src, flow_field).astype(np.uint8)
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
                
                # Calculate optical flow with current settings
                tracked_pts, status, _ = cv2.calcOpticalFlowPyrLK(
                    prevImg=src, 
                    nextImg=displaced_img, 
                    prevPts=intersections, 
                    nextPts=None, 
                    winSize=(win_size, win_size), 
                    maxLevel=max_level,
                    criteria=criteria
                )
                
                # Calculate success rate
                errors = np.linalg.norm(tracked_pts - ground_truth, axis=2)
                success_rate = np.mean(errors < 2)
                results[i, j, k] = success_rate
    
    for i, w_x in enumerate(window_sizes):
        win_size = 2 * w_x + 1
        print(f"\nWindow Size = {w_x} (winSize = {win_size}x{win_size})")
        for j, max_level in enumerate(pyramid_levels):
            print(f"- Pyramid Level {max_level}:")
            print("  ", end="")
            for k, disp_mag in enumerate(displacement_range):
                success_rate = results[i, j, k]
                print(f"({disp_mag}, {success_rate:.2f})", end=" ")
            print()  # Newline after each level
        
    # Visualization
    plt.figure(figsize=(12, 8))
    
    # Create a grid of subplots for each pyramid level
    n_cols = 3
    n_rows = int(np.ceil(len(pyramid_levels) / n_cols))
    
    for j, max_level in enumerate(pyramid_levels):
        plt.subplot(n_rows, n_cols, j+1)
        
        # Plot each window size's performance curve
        for i, w_x in enumerate(window_sizes):
            win_size = 2 * w_x + 1
            plt.plot(displacement_range, results[i, j, :], 
                     label=f'win={win_size}x{win_size}')
        
        plt.title(f'Pyramid Levels: {max_level}', fontsize=10)
        plt.xlabel('Displacement (pixels)', fontsize=8)
        plt.ylabel('Accuracy (% <2px)', fontsize=8)
        plt.ylim(0, 1)
        plt.xlim(0, 50)
        plt.grid(True, alpha=0.3)
        
        # Only show legend on first subplot to save space
        if j == 0:
            plt.legend(fontsize=6, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Parameters
    fwhm = 4
    spacing = 20
    angle = 60
    image_size = (256, 256)
    num_lines = 10
    snr = 4
    
    # Intersection points in proper format
    intersections = np.array([
        [28.598936, 128.04494],
        [38.744957, 145.70131],
        [48.6767, 162.9848],
        [58.5002, 180.07993],
        [68.55512, 197.57776],
        [78.576004, 215.01637],
        [88.38945, 232.094],
        [98.69556, 250.02896],
        [38.618053, 110.36013],
        [48.763744, 128.19568],
        [58.62515, 145.53146],
        [68.39909, 162.71352],
        [78.45307, 180.38783],
        [88.39347, 197.8625],
        [98.15732, 215.02678],
        [108.421135, 233.07],
        [118.33893, 250.50493],
        [48.53459, 92.856384],
        [58.761818, 110.72623],
        [68.63299, 127.97393],
        [78.43669, 145.10376],
        [88.57233, 162.81357],
        [98.51246, 180.18178],
        [108.3061, 197.294],
        [118.611534, 215.3005],
        [128.53896, 232.64648],
        [138.33762, 249.7675],
        [58.32584, 75.57379],
        [68.61341, 93.512726],
        [78.474556, 110.70809],
        [88.28816, 127.82055],
        [98.484085, 145.59967],
        [108.404274, 162.898],
        [118.207825, 179.99292],
        [128.53372, 197.9987],
        [138.45108, 215.29207],
        [148.25966, 232.39575],
        [68.14919, 58.23453],
        [78.50779, 76.22447],
        [88.36868, 93.350044],
        [98.202065, 110.427826],
        [108.46904, 128.25864],
        [118.378975, 145.46939],
        [128.20232, 162.52974],
        [138.55923, 180.51675],
        [148.47624, 197.73979],
        [158.3046, 214.80884],
        [78.30261, 40.312664],
        [88.671165, 58.46619],
        [98.471794, 75.62537],
        [108.265594, 92.77258],
        [118.542, 110.76477],
        [128.38155, 127.99209],
        [138.16527, 145.12166],
        [148.49043, 163.19922],
        [158.34743, 180.45708],
        [168.13617, 197.59546],
        [88.115395, 22.992052],
        [98.55508, 41.196228],
        [108.35547, 58.285656],
        [118.169075, 75.39812],
        [128.51663, 93.44166],
        [138.34593, 110.58149],
        [148.14949, 127.676414],
        [158.50572, 145.73508],
        [168.3624, 162.92264],
        [178.17096, 180.02632],
        [97.90802, 5.707018],
        [108.41888, 23.961388],
        [118.21896, 40.981354],
        [128.05234, 58.05913],
        [138.47112, 76.15357],
        [148.29007, 93.20631],
        [158.1134, 110.26665],
        [168.50067, 128.3064],
        [178.35695, 145.42392],
        [188.1853, 162.49298],
        [118.643555, 6.0960054],
        [128.3835, 23.148926],
        [138.17729, 40.296143],
        [148.6052, 58.55356],
        [158.35393, 75.6219],
        [168.13765, 92.75147],
        [178.49312, 110.88208],
        [188.28949, 128.0338],
        [198.07823, 145.17218],
        [138.23639, 5.8632264],
        [148.04999, 22.975689],
        [158.5492, 41.283653],
        [168.2876, 58.26499],
        [178.09116, 75.35992],
        [188.4777, 93.471466],
        [198.2737, 110.5532],
        [208.08226, 127.656876]
    ], dtype=np.float32).reshape(-1, 1, 2) 

    test_classic_pyramid(intersections)
    print("Test completed successfully.")

   