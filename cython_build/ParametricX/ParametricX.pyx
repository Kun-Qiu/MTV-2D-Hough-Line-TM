# cython: language_level=3
# distutils: language = c++

from utility.py_import import plt, np

cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, exp, isnan, INFINITY, fmax, fmin, round, floor, ceil
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef class ParametricX:
    def __init__(self, tuple center, tuple shape, object image):
        self.center = center
        self.shape = shape 
        self.image = np.asarray(image, dtype=np.float64)
        self.image_view = self.image
        self.validate_inputs()
        
        self.c_params[0] = center[0]  # x
        self.c_params[1] = center[1]  # y
        self.c_params[2] = shape[0]   # ang1
        self.c_params[3] = shape[1]   # ang2
        self.c_params[4] = shape[2]   # rel_intens
        self.c_params[5] = shape[3]   # lin_wid
        self.c_params[6] = shape[4]   # leg_len


    cdef void validate_inputs(self) except *:
        """Validate input parameters."""
        if len(self.center) != 2:
            raise ValueError("center must be a tuple of (x, y)")
        if len(self.shape) != 5:
            raise ValueError("shape needs 5 params: (ang1, ang2, rel_intens, lin_wid, leg_len)")
        
        ang1, ang2, rel_intens, lin_wid, leg_len = self.shape
        if not (-2*np.pi <= ang1 <= 2*np.pi):
            raise ValueError("ang1 must be between -2π and 2π")
        if not (-2*np.pi <= ang2 <= 2*np.pi):
            raise ValueError("ang2 must be between -2π and 2π")
        if not (0 <= rel_intens <= 1):
            raise ValueError("rel_intens must be in [0, 1]")
        if lin_wid <= 0 or leg_len <= 0:
            raise ValueError("lin_wid and leg_len must be positive")


    cdef void parametric_template(self, double* params, double[:, ::1] template, int* min_col, int* min_row) nogil:
        cdef:
            double center_x = params[0], center_y = params[1], 
            double ang1 = params[2], ang2 = params[3]
            double rel_intens = params[4], lin_wid = params[5], leg_len = params[6]
            
            int size = template.shape[0] 
            double half_size = (size - 1) / 2
            double c1 = cos(ang1), s1 = sin(ang1)
            double c2 = cos(ang2), s2 = sin(ang2)
            double sigma = lin_wid / (2 * sqrt(2 * log(2)))
            double sigma_sq = sigma * sigma
            double inv_2_sigma_sq = 1.0 / (2 * sigma_sq)
            double xx, yy, rot1, rot2, leg1, leg2 
            Py_ssize_t i, j
        
        for i in range(size):
            for j in range(size):
                xx = j - half_size
                yy = i - half_size
                
                rot1 = c1 * xx + s1 * yy
                rot2 = c2 * xx - s2 * yy

                leg1 = exp(-rot1 * rot1 * inv_2_sigma_sq)
                leg2 = exp(-rot2 * rot2 * inv_2_sigma_sq)
                template[i,j] = rel_intens * leg1 + (1 - rel_intens) * leg2

        min_col[0] = <int>(center_x - half_size)
        min_row[0] = <int>(center_y - half_size)
    

    cdef double bilinear_interpolate(self, double x, double y) nogil:
        cdef:
            double[:, ::1] image = self.image_view
            int width = image.shape[1], height = image.shape[0]
            int x0 = <int>fmax(0, fmin(width - 2, <int>floor(x)))
            int y0 = <int>fmax(0, fmin(height - 2, <int>floor(y)))
            int x1 = x0 + 1
            int y1 = y0 + 1
            double dx = x - x0, dy = y - y0
            double val00, val01, val10, val11

        val00 = image[y0, x0]
        val01 = image[y0, x1]
        val10 = image[y1, x0]
        val11 = image[y1, x1]

        return (val00 * (1-dx) * (1-dy) +
                val01 * dx * (1-dy) +
                val10 * (1-dx) * dy +
                val11 * dx * dy)


    cdef void _correlate(self, double* params, double* corr) nogil:
        cdef:
            double[:, ::1] template
            double center_x = params[0], center_y = params[1], leg_len = params[6]
            int size = 2 * (<int>params[6]) + 1
            double half_size = (size - 1) / 2.0
            double sum_t = 0.0, sum_i = 0.0, sum_tt = 0.0, sum_ii = 0.0, sum_ti = 0.0
            int count = 0

            double t_mean = 0.0, t_std = 0.0, i_mean = 0.0, i_std = 0.0, covar = 0.0
            int min_col, min_row
            Py_ssize_t i, j
        
        with gil:
            template = np.zeros((size, size), dtype=np.float64)
        
        self.parametric_template(params, template, &min_col, &min_row)
        for i in range(size):
            for j in range(size):
                x = center_x + j - half_size
                y = center_y + i - half_size
                
                t_val = template[i, j]
                i_val = self.bilinear_interpolate(x, y)
                    
                sum_t = sum_t + t_val
                sum_i = sum_i + i_val
                sum_tt = sum_tt + t_val * t_val
                sum_ii = sum_ii + i_val * i_val
                sum_ti = sum_ti + t_val * i_val
                count = count + 1

        if count < 2:
            corr[0] = -INFINITY
            return

        t_mean = sum_t / count
        i_mean = sum_i / count
        t_std = sqrt((sum_tt - sum_t * t_mean) / count)
        i_std = sqrt((sum_ii - sum_i * i_mean) / count)
        covar = (sum_ti - sum_t * i_mean) / count

        corr[0] = covar / (t_std * i_std + 1e-9)


    cdef double* get_params_ptr(self) nogil:
        return &self.c_params[0]


    cpdef tuple get_parametric_X(self):
        cdef: 
            double* params = &self.c_params[0]
            double leg_len = params[6]
            int half_len = <int>round(leg_len / 2)
            int size = 2 * half_len
            np.ndarray template = np.zeros((size, size), dtype=np.float64)
            int min_col, min_row
        
        self.parametric_template(params, template, &min_col, &min_row)
        return template, (min_col, min_row)


    cpdef void visualize(self):
        template, _ = self.get_parametric_X()
        plt.figure(figsize=(10, 5))
        plt.imshow(template, cmap='hot', alpha=0.5)
        plt.title('Parametric X Template')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    cpdef dict correlate(self, np.ndarray params):
        cdef: 
            double[7] param_arr
            double* p 
            double corr = -1.0
            double[:, ::1] image_view = self.image
        
        for i in range(7):
            param_arr[i] = params[i]
        p = &param_arr[0]
        
        with nogil:
            self._correlate(p, &corr)
        return {'correlation': corr}

    
    cpdef np.ndarray get_params(self):
        return np.array(self.c_params)


    cpdef void update_params(self, object indices, object values):
        cdef np.ndarray i = np.asarray(indices, dtype=np.int32)
        cdef np.ndarray v = np.asarray(values, dtype=np.float64)
        cdef Py_ssize_t idx

        for idx in range(i.shape[0]):
            self.c_params[i[idx]] = v[idx]
