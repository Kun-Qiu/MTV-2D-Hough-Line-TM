# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# cython_build/ParametricOPT/ParametricOpt.pyx

from utility.py_import import np, plt
from libc.math cimport atan, fabs, INFINITY
from cython cimport boundscheck, wraparound

cimport numpy as np

cdef extern from * nogil:
    """
    #include <cmath>
    #include <algorithm>
    """
    double c_sqrt "sqrt"(double) nogil
    double c_atan "atan"(double) nogil

@boundscheck(False)
@wraparound(True)
cdef class ParameterOptimizer:
    def __init__(
            self, 
            ParametricX parametric_X,
            double uncertainty = 3.0,
            int num_interval = 20,
            int generation = 3,
            int shrnk_factor = 2,
            bint lock_angle = False,
            bint verbose = False
            ):

        self.parametric_X = parametric_X
        self.uncertainty = uncertainty
        self.num_interval = num_interval
        self.generation = generation
        self.shrnk_factor = shrnk_factor
        self.lock_angle = lock_angle
        self.verbose = verbose

        cdef:
            tuple shape = self.parametric_X.shape
            int NRPos = <int>np.ceil(self.num_interval / 2.0)
        
        self.rad[0] = self.uncertainty * 2
        self.rad[1] = self.uncertainty * 2
        self.rad[2] = atan(self.uncertainty / shape[4])  
        self.rad[3] = atan(self.uncertainty / shape[4])
        self.rad[4] = np.minimum(shape[2], 1 - shape[2]) / 2
        self.rad[5] = 0.75 * shape[2]
        
        self.n_rad[0] = NRPos
        self.n_rad[1] = NRPos
        self.n_rad[2] = self.num_interval
        self.n_rad[3] = self.num_interval
        self.n_rad[4] = self.num_interval
        self.n_rad[5] = self.num_interval


    @staticmethod
    cdef str format_verbose(np.ndarray arr):
        return ', '.join([f'{x:.2f}' for x in arr])


    cdef double[::1] linspace(self, double start, double stop, int num) nogil:
        cdef: 
            double[::1] arr
            double step
            Py_ssize_t i
        
        with gil:
            arr = np.empty(num, dtype=np.float64)

        if num == 1:
            arr[0] = start
            return arr
        
        step = (stop - start) / (num - 1)
        for i in range(num):
            arr[i] = start + i * step
        return arr
    

    cdef double correlate(self, double* param, ParametricX X_obj) nogil:
        cdef:
            Py_ssize_t i, n = params.shape[0]
            double[::1] corrs
            double* params_ptr
            double corr = -1.0
        
        with gil:
            corrs = np.empty(n, dtype=np.float64)

        for i in range(n):
            params_ptr = &params[i, 0]
            temp_opt._correlate(params_ptr, &corr)
            corrs[i] = corr
        return corrs


    cdef void quad_fit_1D(self, double[::1] values, double[::1] corrs, double* opt_x, double* a_coeff) nogil:
    cdef:
        Py_ssize_t i, max_idx = 0
        bint homogeneous = True
        double first_val = values[0]
        double x, y, x2, x3, x4
        double sum_x4 = 0.0, sum_x3 = 0.0, sum_x2 = 0.0, sum_x = 0.0, sum_1 = 0.0
        double sum_x2y = 0.0, sum_xy = 0.0, sum_y = 0.0
        double det, det_a, det_b, det_c, a, b, c, optimal

        # Check homogeneity
        for i in range(1, values.shape[0]):
            if values[i] != first_val:
                homogeneous = False
                break

        if homogeneous or values.shape[0] < 3:
            for i in range(1, corrs.shape[0]):
                if corrs[i] > corrs[max_idx]:
                    max_idx = i
            opt_x[0] = values[max_idx]
            a_coeff[0] = 0.0
            return

        # Compute sums for quadratic fit
        sum_1 = values.shape[0]
        for i in range(values.shape[0]):
            x = values[i]
            y = corrs[i]
            x2 = x * x
            x3 = x2 * x
            x4 = x3 * x
            sum_x4 += x4
            sum_x3 += x3
            sum_x2 += x2
            sum_x += x
            sum_x2y += x2 * y
            sum_xy += x * y
            sum_y += y

        # Compute determinant of the matrix
        det = (sum_x4 * (sum_x2 * sum_1 - sum_x * sum_x) 
            - sum_x3 * (sum_x3 * sum_1 - sum_x * sum_x2) 
            + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2))

        if fabs(det) < 1e-12:
            # Singular matrix, find max index
            max_idx = 0
            for i in range(1, corrs.shape[0]):
                if corrs[i] > corrs[max_idx]:
                    max_idx = i
            opt_x[0] = values[max_idx]
            a_coeff[0] = 0.0
            return

        # Compute coefficients using Cramer's rule
        det_a = (sum_x2y * (sum_x2 * sum_1 - sum_x * sum_x) 
                - sum_x3 * (sum_xy * sum_1 - sum_x * sum_y) 
                + sum_x2 * (sum_xy * sum_x - sum_x2 * sum_y))
        det_b = (sum_x4 * (sum_xy * sum_1 - sum_x * sum_y) 
                - sum_x2y * (sum_x3 * sum_1 - sum_x * sum_x2) 
                + sum_x2 * (sum_x3 * sum_y - sum_xy * sum_x2))
        det_c = (sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy) 
                - sum_x3 * (sum_x3 * sum_y - sum_x * sum_x2y) 
                + sum_x2y * (sum_x3 * sum_x - sum_x2 * sum_x2))

        a = det_a / det
        b = det_b / det
        c = det_c / det

        if a >= 0:
            # Find max index
            max_idx = 0
            for i in range(1, corrs.shape[0]):
                if corrs[i] > corrs[max_idx]:
                    max_idx = i
            opt_x[0] = values[max_idx]
            a_coeff[0] = a
        else:
            optimal = -b / (2 * a)
            if optimal < values[0]:
                optimal = values[0]
            elif optimal > values[values.shape[0] - 1]:
                optimal = values[values.shape[0] - 1]
            
            opt_x[0] = optimal
            a_coeff[0] = a

    
    cdef void quad_fit_2D(self, double[::1] x_vals, double[::1] y_vals, double[:, ::1] corr_matrix, double* opt_x, double* opt_y) nogil:
        cdef:
            Py_ssize_t i, j, max_i = 0, max_j = 0
            double max_val = -INFINITY
            double x, y, x2, x3, x4, y2, y3, y4
            double sum_x4, sum_x3, sum_x2, sum_x, sum_1_x
            double sum_x2y, sum_xy, sum_y_x
            double sum_y4, sum_y3, sum_y2, sum_y, sum_1_y
            double sum_y2x, sum_yx, sum_x_y
            double det, det_a, det_b, det_c, a, b, c, optimal

        # Find maximum correlation index
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if corr_matrix[i, j] > max_val:
                    max_val = corr_matrix[i, j]
                    max_i = i
                    max_j = j

        # X-direction fit ---------------------------------------------------------
        sum_x4 = 0.0
        sum_x3 = 0.0
        sum_x2 = 0.0
        sum_x = 0.0
        sum_1_x = x_vals.shape[0]
        sum_x2y = 0.0
        sum_xy = 0.0
        sum_y_x = 0.0

        for j in range(x_vals.shape[0]):
            x = x_vals[j]
            y = corr_matrix[max_i, j]
            x2 = x * x
            x3 = x2 * x
            x4 = x3 * x
            sum_x4 += x4
            sum_x3 += x3
            sum_x2 += x2
            sum_x += x
            sum_x2y += x2 * y
            sum_xy += x * y
            sum_y_x += y

        det = (sum_x4 * (sum_x2 * sum_1_x - sum_x * sum_x) 
            - sum_x3 * (sum_x3 * sum_1_x - sum_x * sum_x2) 
            + sum_x2 * (sum_x3 * sum_x - sum_x2 * sum_x2))

        if fabs(det) < 1e-12:
            opt_x[0] = x_vals[max_j]
        else:
            det_a = (sum_x2y * (sum_x2 * sum_1_x - sum_x * sum_x) 
                    - sum_x3 * (sum_xy * sum_1_x - sum_x * sum_y_x) 
                    + sum_x2 * (sum_xy * sum_x - sum_x2 * sum_y_x))
            det_b = (sum_x4 * (sum_xy * sum_1_x - sum_x * sum_y_x) 
                    - sum_x2y * (sum_x3 * sum_1_x - sum_x * sum_x2) 
                    + sum_x2 * (sum_x3 * sum_y_x - sum_xy * sum_x2))
            det_c = (sum_x4 * (sum_x2 * sum_y_x - sum_x * sum_xy) 
                    - sum_x3 * (sum_x3 * sum_y_x - sum_x * sum_x2y) 
                    + sum_x2y * (sum_x3 * sum_x - sum_x2 * sum_x2))

            a = det_a / det
            b = det_b / det
            c = det_c / det

            if a < 0:
                optimal = -b / (2 * a)
                if optimal < x_vals[0]:
                    optimal = x_vals[0]
                elif optimal > x_vals[x_vals.shape[0] - 1]:
                    optimal = x_vals[x_vals.shape[0] - 1]
                opt_x[0] = optimal
            else:
                opt_x[0] = x_vals[max_j]

        # Y-direction fit ---------------------------------------------------------
        sum_y4 = 0.0
        sum_y3 = 0.0
        sum_y2 = 0.0
        sum_y = 0.0
        sum_1_y = y_vals.shape[0]
        sum_y2x = 0.0
        sum_yx = 0.0
        sum_x_y = 0.0

        for i in range(y_vals.shape[0]):
            y = y_vals[i]
            x_val = corr_matrix[i, max_j]
            y2 = y * y
            y3 = y2 * y
            y4 = y3 * y
            sum_y4 += y4
            sum_y3 += y3
            sum_y2 += y2
            sum_y += y
            sum_y2x += y2 * x_val
            sum_yx += y * x_val
            sum_x_y += x_val

        det = (sum_y4 * (sum_y2 * sum_1_y - sum_y * sum_y) 
            - sum_y3 * (sum_y3 * sum_1_y - sum_y * sum_y2) 
            + sum_y2 * (sum_y3 * sum_y - sum_y2 * sum_y2))

        if fabs(det) < 1e-12:
            opt_y[0] = y_vals[max_i]
        else:
            det_a = (sum_y2x * (sum_y2 * sum_1_y - sum_y * sum_y) 
                    - sum_y3 * (sum_yx * sum_1_y - sum_y * sum_x_y) 
                    + sum_y2 * (sum_yx * sum_y - sum_y2 * sum_x_y))
            det_b = (sum_y4 * (sum_yx * sum_1_y - sum_y * sum_x_y) 
                    - sum_y2x * (sum_y3 * sum_1_y - sum_y * sum_y2) 
                    + sum_y2 * (sum_y3 * sum_x_y - sum_yx * sum_y2))
            det_c = (sum_y4 * (sum_y2 * sum_x_y - sum_y * sum_yx) 
                    - sum_y3 * (sum_y3 * sum_x_y - sum_y * sum_y2x) 
                    + sum_y2x * (sum_y3 * sum_y - sum_y2 * sum_y2))

            a = det_a / det
            b = det_b / det
            c = det_c / det

            if a < 0:
                optimal = -b / (2 * a)
                if optimal < y_vals[0]:
                    optimal = y_vals[0]
                elif optimal > y_vals[y_vals.shape[0] - 1]:
                    optimal = y_vals[y_vals.shape[0] - 1]
                
                opt_y[0] = optimal
            else:
                opt_y[0] = y_vals[max_i]


    cpdef void quad_optimize(self):
        cdef:
            int num_params, max_steps
            int corr_idx, ang_idx, p_idx
            double[::1] x_vals, y_vals, ang_vals, p_vals
            double[6] cur_rad
            double* cur_param
            double[:, ::1] params_batch

            double[::1] corr, corr_result
            double opt_x, opt_y, start_x, stop_x, start_y, stop_y
            double best_da, best_dp, a_coeff, corr_val = -1.0
            Py_ssize_t xi, yi, idx, nx, ny, best_idx, G, i, j
            
        num_params = 0
        for i in range(2, 6):  # From index 2 to 5
            if self.rad[i] > 0 and self.n_rad[i] > 0:
                num_params += 1
        
        max_steps = (self.generation * (num_params + 1)) + 1
        corr = np.full(max_steps, np.nan)    
            
        self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
        corr[0] = corr_val

        for G in range(self.generation):
            for i in range(6):
                cur_rad[i] = self.rad[i] / (self.shrnk_factor ** G)
            corr_idx = (G * (num_params + 1)) + 1

            if cur_rad[0] > 1e-9 and cur_rad[1] > 1e-9:
                cur_param = self.parametric_X.get_params_ptr()
                print("before", (cur_param[0], cur_param[1]))
                
                x_vals = self.linspace(
                    cur_param[0] - cur_rad[0],
                    cur_param[0] + cur_rad[0] + 1e-8,
                    <int>(2*(self.n_rad[0])+1)
                    )
                
                y_vals = self.linspace(
                    cur_param[1] - cur_rad[1],
                    cur_param[1] + cur_rad[1] + 1e-8,
                    <int>(2*(self.n_rad[1])+1)
                    )
                
                nx, ny = x_vals.shape[0], y_vals.shape[0]
                
                params_batch = np.tile(self.parametric_X.get_params(), ((nx * ny), 1))
                
                for i in range(nx):
                    for j in range(ny):
                        idx = i * ny + j
                        params_batch[idx, 0] = x_vals[i]
                        params_batch[idx, 1] = y_vals[j]

                grid_corrs = np.asarray(
                    self.correlate_batch(params_batch, self.parametric_X)
                    ).reshape(nx, ny)

                self.quad_fit_2D(x_vals, y_vals, grid_corrs, &opt_x, &opt_y)
                print("after", (opt_x, opt_y))
                self.parametric_X.update_params([0, 1], [opt_x, opt_y])
                self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val) 
                corr[corr_idx] = corr_val               
                corr_idx += 1

            # Vectorized angle optimization
            if self.lock_angle:
                ang_vals = self.linspace(
                    -cur_rad[2], cur_rad[2], 
                    <int>(2*(self.n_rad[2])+1)
                    )

                params_batch = np.tile(self.parametric_X.get_params(), (len(ang_vals), 1))
                idx = 0
                for yi in range(ny):
                    for xi in range(nx):
                        idx = i * ny + j
                        params_batch[idx, 2] += ang_vals[xi]
                        params_batch[idx, 3] += ang_vals[yi]
                        idx += 1
                
                ang_corrs = self.correlate_batch(params_batch, self.parametric_X)
                self.quad_fit_1D(ang_vals, ang_corrs, &best_da, &a_coeff)
                
                self.parametric_X.update_params(
                    [2, 3],
                    [self.parametric_X.get_params_ptr()[2] + best_da, self.parametric_X.get_params_ptr()[2] + best_da]
                    )
                
                self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                corr[corr_idx] = corr_val
                corr_idx += 1
            else:
                for ang_idx in range(2, 4):
                    ang_vals = self.linspace(
                        -cur_rad[ang_idx], cur_rad[ang_idx], 
                        <int>(2*(self.n_rad[ang_idx])+1)
                        )

                    params_batch = np.tile(self.parametric_X.get_params(), (len(ang_vals), 1))
                    
                    idx = 0
                    for yi in range(ny):
                        for xi in range(nx):
                            params_batch[idx, ang_idx] += ang_vals[xi]
                    
                    ang_corrs = self.correlate_batch(params_batch, self.parametric_X) 
                    self.quad_fit_1D(ang_vals, ang_corrs, &best_da, &a_coeff)

                    self.parametric_X.update_params([ang_idx], [self.parametric_X.get_params()[ang_idx] + best_da])
                
                    self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                    corr[corr_idx] = corr_val                    
                    corr_idx += 1

            for p_idx in range(4, 6):
                p_vals = self.linspace(
                    -cur_rad[p_idx], cur_rad[p_idx], 
                    <int>(2*(self.n_rad[p_idx])+1)
                    )

                params_batch = np.tile(self.parametric_X.get_params(), (len(p_vals), 1))
                
                idx = 0
                for yi in range(ny):
                    for xi in range(nx):
                        params_batch[idx, p_idx] += p_vals[xi]
                
                p_corrs = self.correlate_batch(params_batch, self.parametric_X)
                self.quad_fit_1D(p_vals, p_corrs, &best_dp, &a_coeff)
                
                self.parametric_X.update_params([p_idx], [self.parametric_X.get_params()[p_idx] + best_dp])
                self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                corr[corr_idx] = corr_val
                corr_idx += 1

        # if self.verbose:
        #     print(f"Optimized parameters: {self.format_verbose(self.parametric_X.get_params())}")
        # return corr


    # Visualization method remains commented as in original code

    # cpdef void visualize(self):
    #     cdef:
    #         np.ndarray img = self.parametric_X.image
    #         np.ndarray template
    #         (np.ndarray, tuple) template_info = self.parametric_X.get_parametric_X()
        
    #     if img is None:
    #         raise ValueError("No image available for visualization")
    #     template, (min_col, min_row) = template_info
        
    #     fig = plt.figure(figsize=(15, 7))
    #     ax1 = plt.subplot(121)
    #     plt.imshow(img, cmap='gray')
        
    #     cdef list extent = [
    #         min_col - 0.5, min_col + template.shape[1] - 0.5,
    #         min_row + template.shape[0] - 0.5, min_row - 0.5
    #     ]
    #     plt.imshow(template, cmap='viridis', alpha=0.7, extent=extent)
    #     plt.scatter(
    #         min_col + template.shape[1]/2, 
    #         min_row + template.shape[0]/2,
    #         c='cyan', marker='o', s=100,
    #         edgecolors='red', linewidth=1
    #     )
    #     plt.title("Template Overlay on Image")
        
    #     ax2 = plt.subplot(122)
    #     plt.imshow(template, cmap='viridis', 
    #             extent=[min_col, min_col + template.shape[1],
    #                     min_row + template.shape[0], min_row])
    #     plt.colorbar(label='Template Intensity')
    #     plt.title("Template Only")
    #     plt.xlabel("X Position")
    #     plt.ylabel("Y Position")

    #     plt.tight_layout()
    #     plt.show()