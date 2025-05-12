# cython_build/parametric_x/parametric_X_c.pxd
# cython: language_level=3
cimport numpy as np

cdef class ParametricX:
    # Class Variables
    cdef public tuple center
    cdef public tuple shape
    cdef public object image
    cdef double[7] c_params

    # Callable by Cython only 
    cdef void _validate_inputs(self) except *
    
    cdef tuple _parametric_template(self, double* params)

    cdef dict _correlate(self, double* params)

    # Callable by Python and Cython
    cpdef dict correlate(self, np.ndarray params)
    
    cpdef void visualize(self)

    cpdef np.ndarray get_params(self)

    cpdef void update_params(self, np.ndarray i, np.ndarray vals)

    cpdef tuple get_parametric_X(self)