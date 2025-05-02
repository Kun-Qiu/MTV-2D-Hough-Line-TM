from src.T0_grid_struct import T0GridStruct
from src.dT_grid_struct import DTGridStruct
from src.parametric_X import ParametricX
from src.parametric_opt import ParameterOptimizer
# from utility.image_utility import stereo_transform
from utility.py_import import np, plt


src_path = "data/Synthetic_Data/Image/src.png"
tar_path = "data\Synthetic_Data\Image\displaced_uniform.png"
# path = "stereo_test.png"
grid_T0 = T0GridStruct((10, 10), src_path, num_lines=10, threshold=0.2, density=10)
grid_dT = DTGridStruct(grid_T0, tar_path, down_scale=4, window_scale=1.2, 
                       search_scale=2.0, rotate_range=45)

# plt.figure(figsize=(10, 10))
# plt.imshow(grid_dT.image, cmap='gray')
        
xs, ys = [], []
for i in range(grid_T0.shape[0]):
    for j in range(grid_T0.shape[1]):

        if grid_T0.params[i, j] is not None:
            ang1, ang2, leg_len = grid_T0.params[i, j]        
        else:
            continue

        if grid_dT.params[i, j] is not None:
            ang1_dt, ang2_dt, leg_len_dt = grid_dT.params[i, j]        
        else:
            continue

        if not np.any(grid_dT.grid[i,j] is None):
            x, y = grid_dT.grid[i, j]
            
            xs.append(x)
            ys.append(y)
            parametricX_obj = ParametricX((int(round(x)), int(round(y))), 
                                      (ang1_dt, ang2_dt, 0.5, 4, leg_len_dt),
                                      image=grid_dT.image)
            optimizer = ParameterOptimizer(parametricX_obj, lock_angle=False, verbose=True)
            optimizer.quad_optimize()
            optimizer.visualize()


# plt.scatter(xs, ys, color='red', s=10, marker='x', label='Intersections')
# plt.title("Grid Intersections")
# plt.axis('off')
# plt.show()