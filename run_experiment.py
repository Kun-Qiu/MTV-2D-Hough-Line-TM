from utility.py_import import plt, np, os 
from src.Scipy_Hough_TM import HoughTM
import time
from tqdm import tqdm


if __name__ == "__main__":
    test_type = {
        # "uniform": "uniform_flow.npy",
        "poiseuille": "poiseuille_flow.npy",
        "lamb_oseen": "lamb_oseen_flow.npy"
    }

    img_type = {
        # "uniform": "displaced_uniform.png",
        "poiseuille": "displaced_poiseuille.png",
        "lamb_oseen": "displaced_lamb_oseen.png"
    }

    all_errors = {key: [] for key in test_type}
    rmse_data = {key: {} for key in test_type}
    # snrs = [2, 4, 8, 16]
    snrs =[4]
    num_runs = 2

    total_start = time.time()
    pbar = tqdm(total=len(snrs) * len(img_type) * num_runs, desc="Overall Progress")

    for snr in snrs:
        snr_errors = {key: [] for key in test_type}
        snr_rmses = {key: [] for key in test_type}
        
        for folder_num in range(num_runs):
            base_dir = f"data/Synthetic_Data/Image/SNR_{snr}/{folder_num}" 
            
            src_path = os.path.join(base_dir, "src.png")
            for key, value in img_type.items():
                img_path = os.path.join(base_dir, value)

                solver = HoughTM(
                    src_path, img_path, num_lines=10, fwhm=4, 
                    temp_scale=0.67, num_interval=30, uncertainty=3,
                    verbose=False, max_level=3, optimize=True
                )
                solver.solve()

                valid_mask = ~np.isnan(solver.disp_field).any(axis=2)
                valid_field = solver.disp_field[valid_mask, :] 
                
                x_indices = valid_field[:, 0]
                y_indices = valid_field[:, 1]

                npy_file = os.path.join(base_dir, test_type[key])
                ground_truth = np.load(npy_file)

                from scipy.interpolate import RegularGridInterpolator
                y_coords = np.arange(ground_truth.shape[0])
                x_coords = np.arange(ground_truth.shape[1])

                if ground_truth.ndim == 3:
                    interp_dx = RegularGridInterpolator((y_coords, x_coords), ground_truth[..., 0], 
                                                    method='linear', bounds_error=False, fill_value=np.nan)
                    interp_dy = RegularGridInterpolator((y_coords, x_coords), ground_truth[..., 1], 
                                                    method='linear', bounds_error=False, fill_value=np.nan)
                    
                    points = np.column_stack((y_indices, x_indices))
                    gt_dx = interp_dx(points)
                    gt_dy = interp_dy(points)
                    extracted_gt = np.column_stack((gt_dx, gt_dy))
                else:
                    interp = RegularGridInterpolator((y_coords, x_coords), ground_truth, 
                                                method='linear', bounds_error=False, fill_value=np.nan)
                    extracted_gt = interp(np.column_stack((y_indices, x_indices)))
                
                valid_interp = ~np.isnan(extracted_gt).any(axis=1)
                errors = np.linalg.norm(valid_field[valid_interp, 2:] - extracted_gt[valid_interp], axis=1)
                snr_errors[key].extend(errors)
                snr_rmses[key].append(np.sqrt(np.mean(errors ** 2)))

                pbar.set_postfix({"SNR": snr, "Folder": folder_num})
                pbar.update(1)

        for key in test_type:
            if snr_errors[key]:
                all_errors[key].extend(snr_errors[key])
                rmse_data[key][snr] = np.mean(snr_rmses[key])

    end_start = time.time()
    print(f"Total Time: {end_start-total_start}")
    print(f"RMSE Data: {rmse_data}")

    # Generate combined RMSE plot (same as before)
    plt.figure(figsize=(10, 6))
    markers = ['o', 's', 'D']
    for idx, (flow, data) in enumerate(rmse_data.items()):
        if data:  # Only plot if we have data
            sorted_snrs = sorted(data.keys())
            rmse_values = [data[snr] for snr in sorted_snrs]
            plt.plot(sorted_snrs, rmse_values, marker=markers[idx], linestyle='--', 
                     label=flow, markersize=8)
            
            # Annotate RMSE values
            for snr, rmse in zip(sorted_snrs, rmse_values):
                plt.text(snr, rmse, f"{rmse:.3f}", ha='center', va='bottom')

    plt.xlabel("SNR")
    plt.ylabel("RMSE")
    plt.title("Average RMSE Across Different SNR Levels (10 runs each)")
    plt.xticks(snrs)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()