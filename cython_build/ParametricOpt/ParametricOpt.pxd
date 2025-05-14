# cython_build/ParametricOpt/ParametricOpt.pxd
# cython: language_level=3

cimport numpy as np
from ParametricX cimport ParametricX

cdef class ParameterOptimizer:
    
    cdef:
        ParametricX parametric_X
        double uncertainty
        int num_interval, generation, shrnk_factor
        bint lock_angle, verbose
        
        double[6] rad
        int[6] n_rad
    
    # C Functions
    @staticmethod
    cdef str format_verbose(np.ndarray arr)

    cdef double[::1] linspace(self, double start, double stop, int num) nogil

    cdef double[::1] correlate_batch(self, double[:, ::1] params, ParametricX temp_opt) nogil
    
    cdef void quad_fit_1D(self, double[::1] values, double[::1] corrs, double* opt_x, double* a_coeff) nogil

    cdef void quad_fit_2D(self, double[::1] x_vals, double[::1] y_vals, double[:, ::1] corr_matrix, double* opt_x, double* opt_y) nogil

    # Python functions

    cpdef void quad_optimize(self)

    # cpdef void visualize(self)