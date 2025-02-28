import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from src.Scipy_Hough_TM import HoughTM


def compute_rmse(predicted, ground_truth):
    """ Compute the Root Mean Square Error (RMSE) between two displacement fields. """
    return np.sqrt(np.mean((predicted - ground_truth) ** 2))

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

    rmse_values = {}
    for key, value in img_type.items():
        img_path = os.path.join(image_dir, value)
        solver = HoughTM(src_path, img_path, num_lines=11,
                         temp_scale=0.67, window_scale=1.2, search_scale=2)

        solver.solve()
        solver.plot_intersections()
        solver.plot_fields(dt=1)

        valid_mask = ~np.isnan(solver.disp_field).any(axis=2)  # Ensures valid_mask has shape (M,)
        valid_field = solver.disp_field[valid_mask, :]  # Now correctly indexed
        solver_dx, solver_dy = valid_field[:, 2], valid_field[:, 3]

        x_indices = valid_field[:, 0].astype(int)
        y_indices = valid_field[:, 1].astype(int)

        npy_file = os.path.join(test_dir, test_type[key])
        ground_truth = np.load(npy_file)
        extracted_gt = ground_truth[y_indices, x_indices]

        # Compute RMSE
        rmse = compute_rmse(valid_field[:, 2:], extracted_gt)
        rmse_values[key] = rmse

    # Plot RMSE values
    plt.figure(figsize=(8, 5))
    plt.bar(rmse_values.keys(), rmse_values.values(), color=['blue', 'green', 'red'])
    plt.xlabel("Flow Type")
    plt.ylabel("RMSE")
    plt.title("RMSE for Different Flow Types")
    plt.ylim(0, max(rmse_values.values()) * 1.2)  # Add some padding
    plt.show()