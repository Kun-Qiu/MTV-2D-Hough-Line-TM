from utility.py_import import plt, np, os 
from src.Scipy_Hough_TM import HoughTM
from utility.image_utility import stereo_transform


if __name__ == "__main__":
    test_dir = "data/Synthetic_Data/Image/SNR_4"
    image_dir = "data/Synthetic_Data/Image/SNR_4"

    test_type = {
        "uniform": "uniform_flow.npy",
        "poiseuille": "poiseuille_flow.npy"
    }

    img_type = {
        "uniform": "displaced_stereo_uniform.png",
        "poiseuille": "displaced_stereo_poiseuille.png"
    }

    for i in range(10):
        src_path = os.path.join(image_dir, f"{i}/stereo_src.png")

        cdf_values = {}
        confidence_intervals = {}
        rmse_values = {}  

        for key, value in img_type.items():
            img_path = os.path.join(image_dir, f"{i}/{value}")
            solver = HoughTM(
                src_path, img_path, num_lines=10, fwhm=4, 
                temp_scale=0.67, uncertainty=3, num_interval=35, 
                verbose=False, max_level=5
                )

            solver.solve()
            # solver.plot_intersections()
            # solver.plot_fields(dt=1)

            valid_mask = ~np.isnan(solver.disp_field).any(axis=2)
            valid_field = solver.disp_field[valid_mask, :] 
            solver_dx, solver_dy = valid_field[:, 2], valid_field[:, 3]

            x_indices = valid_field[:, 0].astype(int)
            y_indices = valid_field[:, 1].astype(int)

            npy_file = os.path.join(test_dir, f"{i}/{test_type[key]}")
            ground_truth = np.load(npy_file)
            extracted_gt = ground_truth[y_indices, x_indices]
            errors = np.linalg.norm(valid_field[:, 2:] - extracted_gt, axis=1)
            
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