# cython: language_level=3
# distutils: language = c++

from utility.py_import import plt, np

cimport numpy as np
from libc.math cimport sqrt, log, cos, sin, exp
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef class ParametricX:
    def __init__(self, tuple center, tuple shape, object image):
        self.center = center
        self.shape = shape 
        self.image = image
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


    cdef tuple parametric_template(self, double* params):
        cdef:
            double* p = params 
            double x = p[0], y = p[1], ang1 = p[2], ang2 = p[3]
            double rel_intens = p[4], lin_wid = p[5], leg_len = p[6]
            int half_len = <int>round(leg_len / 2)
            int size = 2 * half_len
            double c1 = cos(ang1), s1 = sin(ang1)
            double c2 = cos(ang2), s2 = sin(ang2)
            double sigma = lin_wid / (2 * sqrt(2 * log(2)))
            double sigma_sq = sigma * sigma
            double xx, yy, rot1, rot2, leg1, leg2 
            double[:, ::1] template = np.zeros((size, size), dtype=np.float64)
            int min_col = 0, min_row = 0
            Py_ssize_t i, j
        
        for i in range(size):
            for j in range(size):
                xx = j - half_len  
                yy = i - half_len
                rot1 = c1 * xx + s1 * yy
                rot2 = c2 * xx - s2 * yy  # CCW rotation
                leg1 = exp(-(rot1 * rot1) / (2 * sigma_sq))
                leg2 = exp(-(rot2 * rot2) / (2 * sigma_sq))
                template[i,j] = rel_intens * leg1 + (1 - rel_intens) * leg2
        
        min_col = <int>max(min(x - half_len, self.image.shape[1]), 0)
        min_row = <int>max(min(y - half_len, self.image.shape[0]), 0)

        return np.asarray(template), (min_col, min_row)
    

    cdef dict _correlate(self, double* params):
        cdef: 
            dict result = {'correlation': -np.inf}
            np.ndarray template, img_patch
            tuple coords
            double t_mean, t_std, i_mean, i_std, corr
            int min_col, min_row, t_height, t_width
            Py_ssize_t n

        template, coords = self.parametric_template(params)
        min_col, min_row = coords[0], coords[1]
        t_height, t_width = template.shape[0], template.shape[1]

        img_patch = self.image[
            min_row:min_row + t_height, 
            min_col:min_col + t_width
            ]

        t_mean = np.mean(template)
        t_std = np.std(template)
        i_mean = np.mean(img_patch)
        i_std = np.std(img_patch)
        
        cdef np.ndarray t_norm = (template - t_mean) / (t_std + 1e-9)
        cdef np.ndarray i_norm = (img_patch - i_mean) / (i_std + 1e-9)
        
        n = t_norm.size
        corr = np.sum(t_norm * i_norm) / (n - 1)
        
        result['correlation'] = corr if not np.isnan(corr) else -np.inf
        return result


    cpdef void visualize(self):
        cdef double* params = &self.c_params[0]
        cdef np.ndarray template 

        template, _ = self._parametric_template(params)
        plt.figure(figsize=(10, 5))
        plt.imshow(template, cmap='hot', alpha=0.5)
        plt.title('Parametric X Template')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


    cpdef tuple get_parametric_X(self):
        cdef double* params = &self.c_params[0]
        return self._parametric_template(params)


    cpdef dict correlate(self, np.ndarray params):
        cdef double[7] param_arr
        cdef double* p 
        
        for i in range(7):
            param_arr[i] = params[i]
        p = &param_arr[0]

        return self._correlate(p)

    
    cpdef np.ndarray get_params(self):
        return np.array(self.c_params)


    cpdef void update_params(self, object indices, object values):
        cdef np.ndarray i
        cdef np.ndarray v
        cdef Py_ssize_t idx
        
        i = np.asarray(indices, dtype=np.int32)
        v = np.asarray(values, dtype=np.float64)

        for idx in range(i.shape[0]):
            self.c_params[i[idx]] = v[idx]