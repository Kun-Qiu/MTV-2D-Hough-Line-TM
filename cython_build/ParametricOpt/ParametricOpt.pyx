# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# cython_build/ParametricOpt/ParametricOpt.pyx

from utility.py_import import np, plt
from libc.math cimport atan, fabs, INFINITY

cimport numpy as np
cimport cython

cdef extern from * nogil:
    """
    #include <cmath>
    #include <algorithm>
    """
    double c_sqrt "sqrt"(double) nogil
    double c_atan "atan"(double) nogil

@cython.boundscheck(False)
@cython.wraparound(False)
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
        cdef:
            tuple shape = self.parametric_X.shape
            int NRPos = <int>np.ceil(self.num_interval / 2.0)

        self.uncertainty = uncertainty
        self.num_interval = num_interval
        self.generation = generation
        self.shrnk_factor = shrnk_factor
        self.lock_angle = lock_angle
        self.verbose = verbose
        
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
    

    cdef double[::1] correlate_batch(self, double[:, ::1] params, ParametricX temp_opt) nogil:
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
            double[:] values_view = values
            double[:] corrs_view = corrs

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

        with gil:
            try:
                values_np = np.asarray(values)
                corrs_np = np.asarray(corrs)
                
                coeffs = np.polyfit(values_np, corrs_np, 2)
                a = coeffs[0]
                
                if a >= 0: 
                    max_idx = np.argmax(corrs_np)
                    opt_x[0] = values_np[max_idx]
                    a_coeff[0] = a
                else:
                    optimal = -coeffs[1] / (2 * coeffs[0])
                    
                    if optimal < values_np[0]:
                        optimal = values_np[0]
                    elif optimal > values_np[-1]:
                        optimal = values_np[-1]
                    
                    opt_x[0] = optimal
                    a_coeff[0] = a
                    
            except np.linalg.LinAlgError:
                max_idx = np.argmax(corrs_np)
                opt_x[0] = values_np[max_idx]
                a_coeff[0] = 0.0

    
    cdef void quad_fit_2D(self, double[::1] x_vals, double[::1] y_vals, double[:, ::1] corr_matrix, double* opt_x, double* opt_y) nogil:
        cdef:
            Py_ssize_t i, j, max_i = 0, max_j = 0
            double max_val = -INFINITY
            double[::1] row_vals, col_vals

        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if corr_matrix[i, j] > max_val:
                    max_val = corr_matrix[i, j]
                    max_i = i
                    max_j = j

        # X-direction fit -----------------------------------------------------
        with gil:
            try:
                row_np = np.asarray(corr_matrix[max_i, :])
                x_np = np.asarray(x_vals)
                
                x_coeffs = np.polyfit(x_np, row_np, 2)
                a_x = x_coeffs[0]
                
                if a_x < 0:  # Valid maximum
                    opt_x_candidate = -x_coeffs[1] / (2 * x_coeffs[0])
                    if opt_x_candidate >= x_np[0] and opt_x_candidate <= x_np[-1]:
                        opt_x[0] = opt_x_candidate
                    else:
                        opt_x[0] = x_np[max_j]
                else:
                    opt_x[0] = x_np[max_j]
                    
            except np.linalg.LinAlgError:
                opt_x[0] = x_np[max_j]

        # Y-direction fit -----------------------------------------------------
        with gil:
            try:
                col_np = np.asarray(corr_matrix[:, max_j])
                y_np = np.asarray(y_vals)
                
                y_coeffs = np.polyfit(y_np, col_np, 2)
                a_y = y_coeffs[0]
                
                if a_y < 0: 
                    opt_y_candidate = -y_coeffs[1] / (2 * y_coeffs[0])
                    if opt_y_candidate >= y_np[0] and opt_y_candidate <= y_np[-1]:
                        opt_y[0] = opt_y_candidate
                    else:
                        opt_y[0] = y_np[max_i]
                else:
                    opt_y[0] = y_np[max_i]
                    
            except np.linalg.LinAlgError:
                opt_y[0] = y_np[max_i]


    cpdef void quad_optimize(self):
        cdef:
            int num_params, max_steps, corr_idx, G, i, p_idx
            double[::1] x_vals, y_vals, delta_vals
            double[:] corr
            double[:, ::1] params_batch, grid_corrs
            double opt_val,opt_x, opt_y, corr_val = -1.0
            double[6] cur_rad
            double* current_params
            ParametricX temp_opt
            
        num_params = 0
        for i in range(2, 6):  # From index 2 to 5
            if self.rad[i] > 0 and self.n_rad[i] > 0:
                num_params += 1

        max_steps = (self.generation * (num_params + 1)) + 1
        corr = np.full(max_steps, np.nan)
            
        temp_opt = ParametricX(
            center=self.parametric_X.center,
            shape=self.parametric_X.shape,
            image=self.parametric_X.image
            )
        
        current_params = self.parametric_X.get_params_ptr()
        self.parametric_X._correlate(current_params, &corr_val)
        corr[0] = corr_val

        for G in range(self.generation):
            # Update current radii
            for i in range(6):
                cur_rad[i] = self.rad[i] / (self.shrnk_factor ** G)
            corr_idx = (G * (num_params + 1)) + 1

            # Position optimization (x,y)
            if cur_rad[0] > 1e-9 and cur_rad[1] > 1e-9:
                x_vals = np.linspace(
                    current_params[0] - cur_rad[0],
                    current_params[0] + cur_rad[0] + 1e-8,
                    2*self.n_rad[0]+1
                )
                y_vals = np.linspace(
                    current_params[1] - cur_rad[1],
                    current_params[1] + cur_rad[1] + 1e-8,
                    2*self.n_rad[1]+1
                )
                
                # Create parameter grid
                xx, yy = np.meshgrid(x_vals, y_vals)
                params_batch = np.tile(np.asarray(self.parametric_X.get_params()), (xx.size, 1))
                params_batch[:, 0] = xx.ravel()
                params_batch[:, 1] = yy.ravel()

                # Get correlations and find optimal
                grid_corrs = np.asarray(self.correlate_batch(params_batch, temp_opt)).reshape(xx.shape)
                opt_x, opt_y = self.parametric_X.center
                self.quad_fit_2D(x_vals, y_vals, grid_corrs, &opt_x, &opt_y)
                
                # Update parameters
                self.parametric_X.update_params([0, 1], [opt_x, opt_y])
                self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                corr[corr_idx] = corr_val
                corr_idx += 1

            # Angle optimization
            if self.lock_angle:
                delta_vals = np.linspace(-cur_rad[2], cur_rad[2], 2*self.n_rad[2]+1)
                params_batch = np.tile(np.asarray(self.parametric_X.get_params()), (len(delta_vals), 1))
                
                # Explicit assignment to avoid memoryview issues
                for i in range(len(delta_vals)):
                    params_batch[i, 2] = current_params[2] + delta_vals[i]
                    params_batch[i, 3] = current_params[3] + delta_vals[i]
                
                ang_corrs = self.correlate_batch(params_batch, temp_opt)
                best_da, a_coeff = 0.0, 0.0
                self.quad_fit_1D(delta_vals, ang_corrs, &best_da, &a_coeff)
                
                new_ang = current_params[2] + best_da
                self.parametric_X.update_params([2, 3], [new_ang, new_ang])
                self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                corr[corr_idx] = corr_val
                corr_idx += 1
            else:
                for p_idx in [2, 3]:
                    if cur_rad[p_idx] > 1e-9 and self.n_rad[p_idx] > 0:
                        delta_vals = np.linspace(-cur_rad[p_idx], cur_rad[p_idx], 2*self.n_rad[p_idx]+1)
                        params_batch = np.tile(np.asarray(self.parametric_X.get_params()), (len(delta_vals), 1))
                        
                        for i in range(len(delta_vals)):
                            params_batch[i, p_idx] = current_params[p_idx] + delta_vals[i]
                        
                        p_corrs = self.correlate_batch(params_batch, temp_opt)
                        best_dp, a_coeff = 0.0, 0.0
                        self.quad_fit_1D(delta_vals, p_corrs, &best_dp, &a_coeff)
                        
                        new_val = current_params[p_idx] + best_dp
                        self.parametric_X.update_params([p_idx], [new_val])
                        self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                        corr[corr_idx] = corr_val
                        corr_idx += 1

            # Parameter optimization (p4, p5)
            for p_idx in [4, 5]:
                if cur_rad[p_idx] > 1e-9 and self.n_rad[p_idx] > 0:
                    delta_vals = np.linspace(-cur_rad[p_idx], cur_rad[p_idx], 2*self.n_rad[p_idx]+1)
                    params_batch = np.tile(np.asarray(self.parametric_X.get_params()), (len(delta_vals), 1))
                    
                    for i in range(len(delta_vals)):
                        params_batch[i, p_idx] = current_params[p_idx] + delta_vals[i]
                    
                    p_corrs = self.correlate_batch(params_batch, temp_opt)
                    best_dp, a_coeff = 0.0, 0.0
                    self.quad_fit_1D(delta_vals, p_corrs, &best_dp, &a_coeff)
                    
                    new_val = current_params[p_idx] + best_dp
                    self.parametric_X.update_params([p_idx], [new_val])
                    self.parametric_X._correlate(self.parametric_X.get_params_ptr(), &corr_val)
                    corr[corr_idx] = corr_val
                    corr_idx += 1