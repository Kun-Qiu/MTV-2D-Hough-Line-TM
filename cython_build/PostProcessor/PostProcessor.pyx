# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: language = c++
# cython_build/PostProcessor/PostProcessor.pyx

from cython cimport boundscheck, wraparound
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

@boundscheck(False)
@wraparound(False)
cdef class PostProcessor:

    def __init__(self, int n_rows, int n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.count = 0
        
        # Initialize arrays with the specified size
        self.mean_x = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.mean_y = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.mean_w = np.zeros((n_rows, n_cols), dtype=np.float64)
        self.M2_x = np.zeros((n_rows, n_cols), dtype=np.float64) 
        self.M2_y = np.zeros((n_rows, n_cols), dtype=np.float64) 
        self.M2_w = np.zeros((n_rows, n_cols), dtype=np.float64)


    cdef void c_update(self, double[:, ::1] in_x, double[:, ::1] in_y, double[:, ::1] in_w) nogil:
        cdef: 
            int i, j
            double delta_x, delta_y, delta_w
            double current_mean_x, current_mean_y, current_mean_w
            double new_mean_x, new_mean_y, new_mean_w
            double M2_update_x, M2_update_y, M2_update_w
            long local_count = self.count  # Cache count locally

        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if local_count == 0:
                    with gil:
                        self.mean_x[i, j] = in_x[i, j]
                        self.mean_y[i, j] = in_y[i, j]
                        self.mean_w[i, j] = in_w[i, j]
                else:
                    with gil:
                        # Read and calculate everything nogil
                        current_mean_x = self.mean_x[i, j]
                        current_mean_y = self.mean_y[i, j]
                        current_mean_w = self.mean_w[i, j]
                    
                    delta_x = in_x[i, j] - current_mean_x
                    delta_y = in_y[i, j] - current_mean_y
                    delta_w = in_w[i, j] - current_mean_w
                    
                    new_mean_x = current_mean_x + delta_x / (local_count + 1)
                    new_mean_y = current_mean_y + delta_y / (local_count + 1)
                    new_mean_w = current_mean_w + delta_w / (local_count + 1)
                    
                    M2_update_x = delta_x * (in_x[i, j] - new_mean_x)
                    M2_update_y = delta_y * (in_y[i, j] - new_mean_y)
                    M2_update_w = delta_w * (in_w[i, j] - new_mean_w)
                    
                    # Single gil block for all writes
                    with gil:
                        self.mean_x[i, j] = new_mean_x
                        self.mean_y[i, j] = new_mean_y
                        self.mean_w[i, j] = new_mean_w
                        self.M2_x[i, j] += M2_update_x
                        self.M2_y[i, j] += M2_update_y
                        self.M2_w[i, j] += M2_update_w
        
        # Final count update
        with gil:
            self.count += 1
        

    cdef void process_std(self, double[:, ::1] out_x, double[:, ::1] out_y, double[:, ::1] out_w) nogil:

        cdef: 
            int i, j
            double variance_x, variance_y, variance_w
        
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                if self.count > 1:
                    variance_x = self.M2_x[i, j] / (self.count - 1)
                    variance_y = self.M2_y[i, j] / (self.count - 1)
                    variance_w = self.M2_w[i, j] / (self.count - 1)
                    
                    with gil:
                        out_x[i, j] = sqrt(variance_x)
                        out_y[i, j] = sqrt(variance_y)
                        out_w[i, j] = sqrt(variance_w)
                else:
                    out_x[i, j] = 0.0
                    out_y[i, j] = 0.0
                    out_w[i, j] = 0.0


    cpdef tuple get_std(self):
        cdef: 
            np.ndarray[np.float64_t, ndim=2] std_x = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            np.ndarray[np.float64_t, ndim=2] std_y = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            np.ndarray[np.float64_t, ndim=2] std_w = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
            
            double[:, ::1] std_x_view = std_x
            double[:, ::1] std_y_view = std_y
            double[:, ::1] std_w_view = std_w
            
        self.process_std(std_x_view, std_y_view, std_w_view)
        return (std_x, std_y, std_w)


    cpdef tuple get_mean(self):
        return (np.asarray(self.mean_x), np.asarray(self.mean_y), np.asarray(self.mean_w))

    
    cpdef void update(self, double[:, :] in_x, double[:, :] in_y, double[:, :] in_w):
        cdef: 
            double[:, ::1] in_x_contig = np.ascontiguousarray(in_x, dtype=np.float64)
            double[:, ::1] in_y_contig = np.ascontiguousarray(in_y, dtype=np.float64)
            double[:, ::1] in_w_contig = np.ascontiguousarray(in_w, dtype=np.float64)
        
        # Call the internal method with contiguous memoryviews
        self.c_update(in_x_contig, in_y_contig, in_w_contig)


