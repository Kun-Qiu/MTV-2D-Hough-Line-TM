# parametric_x.pxd

cimport numpy as np

cdef class ParametricX:
    cdef tuple center
    cdef tuple shape
    cdef object image
    cdef np.ndarray params

    cdef void _validate_inputs(self) except *
    
    cdef tuple _parametric_template(self, list params=*)
    
    @staticmethod
    cdef np.ndarray _rotation_matrix(double angle, bint counter_clockwise=*)

    cpdef dict correlate(self, list params)
    
    cpdef np.ndarray get_parametric_X(self, list params=*)
    
    cpdef void visualize(self)