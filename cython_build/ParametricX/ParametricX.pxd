# cython_build/ParametricX/ParametricX.pxd
# cython: language_level=3
cimport numpy as np

cdef class ParametricX:
    # Class Variables
    cdef public tuple center
    cdef public tuple shape
    cdef public object image
    cdef double[7] c_params

    # Callable by Cython only 
    cdef void validate_inputs(self) except *
    
    cdef void parametric_template(
        self, 
        double* params, 
        double[:, ::1] template, 
        int* min_col, 
        int* min_row
        )

    cdef void _correlate(self, double* params, double* corr)

    # Callable by Python and Cython
    cpdef dict correlate(self, np.ndarray params)
    
    cpdef void visualize(self)

    cpdef np.ndarray get_params(self)

    cpdef void update_params(self, object i, object vals)

    cpdef tuple get_parametric_X(self)