# cython: language_level=3
# distuils: language = c++

from utility.py_import import plt, np, convolve2d

cimport numpy as np
from libc.math cimport exp, sqrt, log, cos, sin
from cython cimport boundscheck, wraparound

@boundscheck(False)
@wraparound(False)
cdef class ParametricX:
    cdef:
        public tuple center
        public tuple shape
        public object image
        double[7] params
        double[:] params_view

    def __init__(self, tuple center, tuple shape, object image=None):
        self.center = center
        self.shape = shape
        self.image = image
        self._validate_inputs()
        
        # Initialize params array
        self.params[:] = [
            self.center[0],  # x
            self.center[1],  # y
            self.shape[0],   # ang1
            self.shape[1],   # ang2
            self.shape[2],   # rel_intens
            self.shape[3],   # lin_wid
            self.shape[4]    # leg_len
        ]
        self.params_view = self.params

    cdef void _validate_inputs(self) except *:
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

    @staticmethod
    cdef double[:, :] _rotation_matrix(double angle, bint ccw=False):
        """Create 2D rotation matrix."""
        cdef double c = cos(angle), s = sin(angle)
        cdef double[:, :] R = np.empty((2, 2), dtype=np.float64)
        
        if ccw:
            R[0,0], R[0,1] = c, -s
            R[1,0], R[1,1] = s, c
        else:
            R[0,0], R[0,1] = c, s
            R[1,0], R[1,1] = -s, c
        
        return np.asarray(R)

    cdef tuple _parametric_template(self, double[:] params=None):
        """Generate X-shaped template from parameters."""
        cdef double[7] local_params
        cdef double* p = (
            &self.params[0] if params is None or params.shape[0] != 7 else &local_params[0]
            )
        
        if p != &self.params[0]:
            for i in range(7): local_params[i] = params[i]

        cdef double x = p[0], y = p[1], ang1 = p[2], ang2 = p[3]
        cdef double rel_intens = p[4], lin_wid = p[5], leg_len = p[6]
        cdef int half_len = int(round(leg_len / 2))

        # Create coordinate grid
        cdef np.ndarray xx, yy
        xx, yy = np.meshgrid(
            np.arange(-half_len, half_len), 
            np.arange(-half_len, half_len)
            )
        
        # Rotate coordinates
        cdef np.ndarray coords = np.stack([xx.ravel(), yy.ravel()]).T
        cdef np.ndarray rot1 = coords @ ParametricX._rotation_matrix(ang1)
        cdef np.ndarray rot2 = coords @ ParametricX._rotation_matrix(ang2, ccw=True)
        
        # Create Gaussian legs
        cdef double sigma = lin_wid / (2 * sqrt(2 * log(2)))
        cdef np.ndarray leg1 = exp(-rot1[:, 0]**2 / (2 * sigma**2))
        cdef np.ndarray leg2 = exp(-rot2[:, 0]**2 / (2 * sigma**2))
        cdef np.ndarray template = (rel_intens * leg1 + (1 - rel_intens) * leg2).reshape(
            (xx.shape[0], xx.shape[1])
            )
        
        # Calculate image coordinates
        cdef int min_col = 0, min_row = 0
        if self.image is not None:
            min_col = int(np.clip(x - half_len, 0, self.image.shape[1]))
            min_row = int(np.clip(y - half_len, 0, self.image.shape[0]))

        return template, (min_col, min_row)
    

    cpdef dict correlate(self, list params):
        """Correlate template with image region."""
        cdef dict result = {
            'correlation': -np.inf, 
            'background': 0.0, 
            'noise': np.nan, 
            'difference': None
        }
        
        cdef np.ndarray template
        cdef tuple coords
        template, coords = self._parametric_template(params)
        cdef int min_col, min_row 
        cdef Py_ssize_t t_height, t_width
        
        min_col, min_row = coords[0], coords[1]
        t_height, t_width = template.shape[0], template.shape[1]

        # Check if template fits in image
        if (self.image is None or 
            self.image.shape[0] < min_row + t_height or 
            self.image.shape[1] < min_col + t_width):
            return result

        cdef np.ndarray img_patch = self.image[
            min_row:min_row + t_height, 
            min_col:min_col + t_width
            ]

        if img_patch.shape != template.shape:
            return result

        # Normalize and correlate
        cdef double t_mean = np.mean(template), t_std = np.std(template)
        cdef double i_mean = np.mean(img_patch), i_std = np.std(img_patch)
        
        cdef np.ndarray t_norm = (template - t_mean) / (t_std + 1e-9)
        cdef np.ndarray i_norm = (img_patch - i_mean) / (i_std + 1e-9)
        
        cdef double corr = np.corrcoef(t_norm.ravel(), i_norm.ravel())[0, 1]
        
        result.update({
            'correlation': corr if not np.isnan(corr) else -np.inf,
            'background': ((1 - t_mean) / t_std * i_std) + i_mean,
            'difference': (i_norm - t_norm) * i_std
        })

        cdef np.ndarray local_mean, kernel
        # Calculate noise if possible
        if t_height > 2 and t_width > 2:
            kernel = np.ones((3, 3)) / 9
            local_mean = convolve2d(result['difference'], kernel, mode='valid')
            result['noise'] = 5 * np.std(result['difference'][1:-1, 1:-1] - local_mean)

        return result


    cpdef np.ndarray get_parametric_X(self, list params=None):
        """Get the X template."""
        cdef np.ndarray template
        template, _ = self._parametric_template(params)
        return template


    cpdef void visualize(self):
        """Visualize the template."""
        cdef np.ndarray template = self.get_parametric_X()
        plt.figure(figsize=(10, 5))
        plt.imshow(template, cmap='hot', alpha=0.5)
        plt.title('Parametric X Template')
        plt.axis('off')
        plt.tight_layout()
        plt.show()