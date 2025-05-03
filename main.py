from utility.py_import import plt, np, os 
from src.Scipy_Hough_TM import HoughTM


def compute_cdf(errors):
    """ 
    Compute the cumulative probability distribution function (CDF) of errors.
    """
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, cdf


if __name__ == "__main__":
    test_dir = "data/Test"
    test_type = {
        "uniform": "displaced_uniform.npy",
        "poiseuille": "displaced_poiseuille.npy",
        "lamb_oseen": "displaced_lamb_oseen.npy"
    }

    image_dir = "data/Synthetic_Data/Image"
    src_path = os.path.join(image_dir, "src.png")
    img_type ={
        "uniform": "displaced_uniform.png",
        "poiseuille": "displaced_poiseuille.png",
        "lamb_oseen": "displaced_lamb_oseen.png"
    }

    cdf_values = {}
    confidence_intervals = {}
    rmse_values = {}  # New dictionary to store RMSE values

    for key, value in img_type.items():
        img_path = os.path.join(image_dir, value)
        solver = HoughTM(
            src_path, img_path, verbose=False, num_lines=10,
            temp_scale=0.67, window_scale=1.2, search_scale=2.0
            )

        solver.solve()
        solver.plot_intersections()
        solver.plot_fields(dt=1)

        valid_mask = ~np.isnan(solver.disp_field).any(axis=2)
        valid_field = solver.disp_field[valid_mask, :] 
        solver_dx, solver_dy = valid_field[:, 2], valid_field[:, 3]

        x_indices = valid_field[:, 0].astype(int)
        y_indices = valid_field[:, 1].astype(int)

        npy_file = os.path.join(test_dir, test_type[key])
        ground_truth = np.load(npy_file)
        extracted_gt = ground_truth[y_indices, x_indices]

        # Compute RMSE
        errors = np.linalg.norm(valid_field[:, 2:] - extracted_gt, axis=1)
        
        # Compute and store RMSE
        rmse = np.sqrt(np.mean(errors ** 2))  # RMSE calculation
        rmse_values[key] = rmse

        # Compute CDF
        sorted_errors, cdf = compute_cdf(errors)
        cdf_values[key] = (sorted_errors, cdf)
        
        # Find 95% confidence interval
        confidence_idx = np.searchsorted(cdf, 0.95)
        confidence_intervals[key] = sorted_errors[confidence_idx]

    # plt.figure(figsize=(8, 5))
    # for key, (sorted_errors, cdf) in cdf_values.items():
    #     plt.plot(sorted_errors, cdf, label=key)
    #     plt.axvline(confidence_intervals[key], color='k', linestyle='--', alpha=0.7,
    #                 label=f"{key} 95% CI: {confidence_intervals[key]:.3f}")
    
    # plt.xlabel("Displacement Error")
    # plt.ylabel("Cumulative Probability")
    # plt.title("CDF of Displacement Errors for Different Flow Types")
    # plt.legend()
    # plt.grid()

    # plt.figure(figsize=(8, 5))
    # plt.bar(rmse_values.keys(), rmse_values.values(), color=['blue', 'orange', 'green'])
    # plt.xlabel("Flow Type")
    # plt.ylabel("RMSE")
    # plt.title("RMSE of Displacement Errors for Different Flow Types")
    
    # # Add value labels on top of bars
    # for flow_type, value in rmse_values.items():
    #     plt.text(flow_type, value, f"{value:.3f}", ha='center', va='bottom')
    
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    # plt.tight_layout()
    # plt.show()
    # plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot CDF on the first subplot
    for key, (sorted_errors, cdf) in cdf_values.items():
        ax1.plot(sorted_errors, cdf, label=key)
        ax1.axvline(confidence_intervals[key], color='k', linestyle='--', alpha=0.7,
                    label=f"{key} 95% CI: {confidence_intervals[key]:.3f}")
    
    ax1.set_xlabel("Displacement Error")
    ax1.set_ylabel("Cumulative Probability")
    ax1.set_title("CDF of Displacement Errors")
    ax1.legend()
    ax1.grid()

    # Plot RMSE on the second subplot
    colors = ['blue', 'orange', 'green']
    for i, (flow_type, rmse) in enumerate(rmse_values.items()):
        ax2.bar(flow_type, rmse, color=colors[i], label=flow_type)
        ax2.text(flow_type, rmse, f"{rmse:.3f}", ha='center', va='bottom')
    
    ax2.set_xlabel("Flow Type")
    ax2.set_ylabel("RMSE")
    ax2.set_title("Root Mean Square Error (RMSE)")
    ax2.legend()
    ax2.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
