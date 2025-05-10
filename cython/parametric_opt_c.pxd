## parameter_optimizer.pxd
## Header

cimport numpy as np
from libc.math cimport atan
from src.parametric_X cimport ParametricX

cdef class ParameterOptimizer:
    cdef:
        ParametricX parametric_X
        float uncertainty
        int num_interval, generation, shrnk_factor
        bint lock_angle, verbose
        
        float rad[6]
        int n_rad[6]
    
    cpdef np.ndarray quad_optimize(self)
    
    cdef tuple _quad_fit_1D(self, np.ndarray values, np.ndarray corrs)
    
    cdef tuple _quad_fit_2D(
        self, 
        np.ndarray x_vals, 
        np.ndarray y_vals, 
        np.ndarray corr_matrix, 
        tuple x_lim, 
        tuple y_lim
    )
    
    cdef np.ndarray __correlate_batch(self, np.ndarray params_array, ParametricX temp_opt)
    
    cpdef void visualize(self)
    
    ## end parametric_opt.pxd