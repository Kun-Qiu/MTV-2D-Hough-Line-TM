from utility.py_import import plt, np, os 
from src.Scipy_Hough_TM import HoughTM
import time

def compute_cdf(errors):
    """ 
    Compute the cumulative probability distribution function (CDF) of errors.
    """
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    return sorted_errors, cdf


if __name__ == "__main__":
    test_dir = "data/Synthetic_Data/Image/SNR_4/2"
    image_dir = "data/Synthetic_Data/Image/SNR_4/2"

    test_type = {
        "uniform": "uniform_flow.npy",
        "poiseuille": "poiseuille_flow.npy",
        "lamb_oseen": "lamb_oseen_flow.npy"
    }

    img_type = {
        "uniform": "displaced_uniform.png"#,
        # "poiseuille": "displaced_poiseuille.png",
        # "lamb_oseen": "displaced_lamb_oseen.png"
    }
    src_path = os.path.join(image_dir, "src.png")
    uncertainties = np.arange(3, 4, 2)  # [1, 3, 5]
    num_intervals = np.arange(10, 36, 5)

    results = {
        flow_type: {
            'uncertainty': [],
            'num_intervals': [],
            'rmse': [],
            'runtime': []
        } 
        for flow_type in img_type.keys()
    }  

    total_start = time.time()
    for key, value in img_type.items():
        img_path = os.path.join(image_dir, value)
        npy_file = os.path.join(test_dir, test_type[key])
        ground_truth = np.load(npy_file)
        
        for uncertainty in uncertainties:
            for num_int in num_intervals:
                print(f"Processing {key} with uncertainty={uncertainty}, num_intervals={num_int}")
                
                start_time = time.time()
                solver = HoughTM(
                    src_path, img_path, num_lines=10, fwhm=4, 
                    temp_scale=0.7, uncertainty=3,
                    num_interval=num_int, verbose=False
                )
                
                solver.solve()
                runtime = time.time() - start_time
                
                valid_mask = ~np.isnan(solver.disp_field).any(axis=2)
                valid_field = solver.disp_field[valid_mask, :] 
                
                x_indices = valid_field[:, 0].astype(int)
                y_indices = valid_field[:, 1].astype(int)
                
                extracted_gt = ground_truth[y_indices, x_indices]
                errors = np.linalg.norm(valid_field[:, 2:] - extracted_gt, axis=1)
                rmse = np.sqrt(np.mean(errors ** 2))
                
                results[key]['uncertainty'].append(uncertainty)
                results[key]['num_intervals'].append(num_int)
                results[key]['rmse'].append(rmse)
                results[key]['runtime'].append(runtime)

    total_end = time.time()
    print(f"Total Runtime = {total_end - total_start}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot RMSE results
    for unc in uncertainties:
        mask = np.array(results['uniform']['uncertainty']) == unc
        x = np.array(results['uniform']['num_intervals'])[mask]
        y = np.array(results['uniform']['rmse'])[mask]
        sort_idx = np.argsort(x)
        ax1.plot(x[sort_idx], y[sort_idx], 'o-', label=f'Unc={unc}')
    
    ax1.set_title('RMSE vs Number of Intervals')
    ax1.set_xlabel('Number of Intervals')
    ax1.set_ylabel('RMSE')
    ax1.grid(True)
    ax1.legend()

    # Plot Runtime results
    for unc in uncertainties:
        mask = np.array(results['uniform']['uncertainty']) == unc
        x = np.array(results['uniform']['num_intervals'])[mask]
        y = np.array(results['uniform']['runtime'])[mask]
        sort_idx = np.argsort(x)
        ax2.plot(x[sort_idx], y[sort_idx], 'o-', label=f'Unc={unc}')
    
    ax2.set_title('Runtime vs Number of Intervals')
    ax2.set_xlabel('Number of Intervals')
    ax2.set_ylabel('Runtime (seconds)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()