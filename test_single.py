from utility.py_import import plt, np, os, cv2
from src.Scipy_Hough_TM import HoughTM
from utility.image_utility import stereo_transform


if __name__ == "__main__":
    test_dir = "data/Synthetic_Data/Image/SNR_4"
    image_dir = "data/Synthetic_Data/Image/SNR_4"

    test_type = {
        "uniform": "uniform_flow.npy",
        "poiseuille": "poiseuille_flow.npy",
        "lamb_oseen": "lamb_oseen_flow.npy"
    }

    img_type = {
        "uniform": "displaced_uniform.png",
        "poiseuille": "displaced_poiseuille.png",
        "lamb_oseen": "displaced_lamb_oseen.png"
    }

    for i in range(1):
        src_path = os.path.join(image_dir, f"{i}/src.png")

        cdf_values = {}
        confidence_intervals = {}
        rmse_values = {}  

        for key, value in img_type.items():
            img_path = os.path.join(image_dir, f"{i}/{value}")
            solver = HoughTM(
                src_path, img_path, num_lines=10, fwhm=4, 
                temp_scale=0.67, uncertainty=3, num_interval=40, 
                verbose=False, max_level=5
                )

            solver.solve()
            solver.plot_intersections()
            solver.plot_fields(dt=1)

            valid_mask = ~np.isnan(solver.disp_field).any(axis=2)
            valid_field = solver.disp_field[valid_mask, :] 

            x_indices = valid_field[:, 0]
            y_indices = valid_field[:, 1]

            npy_file = os.path.join(test_dir, f"{i}/{test_type[key]}")
            ground_truth = np.load(npy_file)

            from scipy.interpolate import RegularGridInterpolator
            y_coords = np.arange(ground_truth.shape[0])
            x_coords = np.arange(ground_truth.shape[1])

            interp_dx = RegularGridInterpolator((y_coords, x_coords), ground_truth[..., 0], 
                                            method='linear', bounds_error=False, fill_value=np.nan)
            interp_dy = RegularGridInterpolator((y_coords, x_coords), ground_truth[..., 1], 
                                            method='linear', bounds_error=False, fill_value=np.nan)
            
            points = np.column_stack((y_indices, x_indices))
            gt_dx = interp_dx(points)
            gt_dy = interp_dy(points)
            extracted_gt = np.column_stack((gt_dx, gt_dy))
            
            valid_interp = ~np.isnan(extracted_gt).any(axis=1)
            errors = np.linalg.norm(valid_field[valid_interp, 2:] - extracted_gt[valid_interp], axis=1)
            rmse = np.sqrt(np.mean(errors ** 2)) 
            rmse_values[key] = rmse


        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        colors = ['blue', 'orange', 'green']
        for i, (flow_type, rmse) in enumerate(rmse_values.items()):
            ax.bar(flow_type, rmse, color=colors[i], label=flow_type)
            ax.text(flow_type, rmse, f"{rmse:.3f}", ha='center', va='bottom')

        ax.set_xlabel("Flow Type")
        ax.set_ylabel("RMSE")
        ax.set_title("Root Mean Square Error (RMSE)")
        ax.legend()
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()