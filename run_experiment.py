from utility.py_import import plt, np, os 
from src.Scipy_Hough_TM import HoughTM
import time

def compute_cdf(errors):
    """Compute the cumulative probability distribution function (CDF) of errors."""
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, cdf


if __name__ == "__main__":
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

    all_errors = {key: [] for key in test_type}
    rmse_data = {key: {} for key in test_type}
    snrs = [1, 2, 4, 8, 16]

    total_start = time.time()
    for snr in snrs:
        # Initialize storage for this SNR
        snr_errors = {key: [] for key in test_type}
        snr_rmses = {key: [] for key in test_type}
        
        # Process each of the 10 folders for this SNR
        for folder_num in range(10):
            base_dir = f"data/Synthetic_Data/Image/SNR_{snr}/{folder_num}" 
            
            src_path = os.path.join(base_dir, "src.png")
            for key, value in img_type.items():
                img_path = os.path.join(base_dir, value)

                solver = HoughTM(
                    src_path, img_path, num_lines=10, fwhm=4, 
                    temp_scale=0.67, num_interval=30, uncertainty=3,
                    verbose=False
                )
                solver.solve()
                
                valid_mask = ~np.isnan(solver.disp_field).any(axis=2)
                valid_field = solver.disp_field[valid_mask, :] 
                
                x_indices = valid_field[:, 0].astype(int)
                y_indices = valid_field[:, 1].astype(int)

                npy_file = os.path.join(base_dir, test_type[key])
                ground_truth = np.load(npy_file)
                extracted_gt = ground_truth[y_indices, x_indices]

                errors = np.linalg.norm(valid_field[:, 2:] - extracted_gt, axis=1)
                snr_errors[key].extend(errors)
                snr_rmses[key].append(np.sqrt(np.mean(errors ** 2)))

        # Compute averages for this SNR
        for key in test_type:
            if snr_errors[key]:  # Only if we have data
                all_errors[key].extend(snr_errors[key])
                rmse_data[key][snr] = np.mean(snr_rmses[key])

    end_start = time.time()
    print(f"Total Time: {end_start-total_start}")

    # Generate combined CDF plot (same as before)
    plt.figure(figsize=(10, 6))
    for key in test_type:
        if all_errors[key]:  # Only plot if we have data
            sorted_errors, cdf = compute_cdf(np.array(all_errors[key]))
            confidence_idx = np.searchsorted(cdf, 0.95)
            confidence_value = sorted_errors[confidence_idx]
            
            plt.plot(sorted_errors, cdf, label=f"{key} (95% CI: {confidence_value:.2f})")
    
    plt.xlabel("Displacement Error")
    plt.ylabel("Cumulative Probability")
    plt.title("Combined CDF of Displacement Errors (All SNRs)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

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