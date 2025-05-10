# cython: language_level=3
cimport numpy as np

cdef class ParametricX:
    # Class Variables
    cdef public tuple center
    cdef public tuple shape
    cdef public object image
    cdef double[7] c_params
    
    # Class Functions

    # Callable by Cython only 
    cdef void _validate_inputs(self) except *
    
    cdef tuple _parametric_template(self, double* params)

    cdef dict _correlate(self, double* params)
    
    @staticmethod
    cdef double[:, :] _rotation_matrix(double angle, bint ccw=*)

    cdef np.ndarray get_parametric_X(self, double* params)
    
    # Callable by Python and Cython
    cpdef dict correlate(self, list params)
    
    cpdef void visualize(self)