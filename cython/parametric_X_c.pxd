# cython: language_level=3
cimport numpy as np

cdef class ParametricX:
    # Class Variables
    cdef public tuple center
    cdef public tuple shape
    cdef public object image
    cdef double[7] params_arr
    cdef double[:] params_view
    
    # Class Functions
    cdef void _validate_inputs(self) except *
    
    cdef tuple _parametric_template(self, double[:] params=*)
    
    @staticmethod
    cdef np.ndarray _rotation_matrix(double angle, bint ccw=*)
    
    cpdef dict correlate(self, list params)
    
    cpdef np.ndarray get_parametric_X(self, list params=*)
    
    cpdef void visualize(self)