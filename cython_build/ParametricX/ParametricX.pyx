# cython: language_level=3
# distutils: language = c++

from utility.py_import import plt, np

cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, exp, isnan, INFINITY, fmax, fmin, round
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
            double* p = params 
            double x = p[0], y = p[1], ang1 = p[2], ang2 = p[3]
            double rel_intens = p[4], lin_wid = p[5], leg_len = p[6]
            int size = template.shape[0] 
            int image_width = self.image_view.shape[1], image_height = self.image_view.shape[0]
            int half_len = size // 2
            double c1 = cos(ang1), s1 = sin(ang1)
            double c2 = cos(ang2), s2 = sin(ang2)
            double sigma = lin_wid / (2 * sqrt(2 * log(2)))
            double sigma_sq = sigma * sigma
            double xx, yy, rot1, rot2, leg1, leg2 
            Py_ssize_t i, j
        
        for i in range(size):
            for j in range(size):
                xx = j - half_len  
                yy = i - half_len
                rot1 = c1 * xx + s1 * yy
                rot2 = c2 * xx - s2 * yy
                leg1 = exp(-(rot1 * rot1) / (2 * sigma_sq))
                leg2 = exp(-(rot2 * rot2) / (2 * sigma_sq))
                template[i,j] = rel_intens * leg1 + (1 - rel_intens) * leg2
        
        min_col[0] = <int>fmax(fmin(x - half_len, image_width), 0)
        min_row[0] = <int>fmax(fmin(y - half_len, image_height), 0)
    

    cdef void _correlate(self, double* params, double* corr) nogil:
        cdef:
            double[:, ::1] template, img_patch
            int size = <int>(2 * round(params[6] / 2))
            double t_mean = 0.0, t_std = 0.0, i_mean = 0.0, i_std = 0.0
            double sum_t = 0.0, sum_i = 0.0, sum_tt = 0.0, sum_ii = 0.0, sum_ti = 0.0
            double temp
            int min_col, min_row, t_height, t_width
            int image_width = self.image_view.shape[1], image_height = self.image_view.shape[0]
            Py_ssize_t i, j, n
        
        with gil:
            template = np.zeros((size, size), dtype=np.float64)

        self.parametric_template(params, template, &min_col, &min_row)
        t_height, t_width = template.shape[0], template.shape[1]
        
        if template.shape[0] == 0 or template.shape[1] == 0:
            corr[0] = -INFINITY
            return

        if (min_row < 0 or min_row + t_height > image_height or
            min_col < 0 or min_col + t_width > image_width):
            corr[0] = -INFINITY
            return

        img_patch = self.image_view[
            min_row:min_row + t_height, 
            min_col:min_col + t_width
            ]

        n = t_height * t_width
        for i in range(t_height):
            for j in range(t_width):
                temp = template[i, j]
                sum_t += temp
                sum_tt += temp * temp
                
                temp = img_patch[i, j]
                sum_i += temp
                sum_ii += temp * temp
                
        t_mean = sum_t / n
        i_mean = sum_i / n
        t_std = sqrt((sum_tt - sum_t * t_mean) / n)
        i_std = sqrt((sum_ii - sum_i * i_mean) / n)
        
        for i in range(t_height):
            for j in range(t_width):
                norm_t = (template[i, j] - t_mean) / (t_std + 1e-9)
                norm_i = (img_patch[i, j] - i_mean) / (i_std + 1e-9)
                sum_ti += norm_t * norm_i
        
        corr[0] = sum_ti / (n - 1) if not isnan(sum_ti) else -INFINITY


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