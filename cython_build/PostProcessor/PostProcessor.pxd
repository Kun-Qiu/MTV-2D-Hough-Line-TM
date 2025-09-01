# cython_build/PostProcessor/PostProcessor.pxd
# cython: language_level=3

cimport numpy as np

cdef class PostProcessor:
    
    cdef:
        int n_rows, n_cols
        double[:, ::1] mean_x, mean_y, mean_w, M2_x, M2_y, M2_w
        long count
    
    # C Functions
    cdef void c_update(self, double[:, ::1] in_x, double[:, ::1] in_y, double[:, ::1] in_w) nogil

    cdef void process_std(self, double[:, ::1] out_x, double[:, ::1] out_y, double[:, ::1] out_w) nogil

    # Python functions
    cpdef tuple get_std(self)

    cpdef tuple get_mean(self)

    cpdef void update(self, double[:, :] in_x, double[:, :] in_y, double[:, :] in_w)