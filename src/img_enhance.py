import numpy as np 
import cv2
from dataclasses import dataclass
import matplotlib.pyplot as plt
from skimage.util import img_as_float, img_as_ubyte


@dataclass
class SingleShotEnhancer:
    avg_shot     : np.ndarray
    single_shot  : np.ndarray

    def __post_init__(self):
        self.__avg_shot    = img_as_float(self.avg_shot).astype(np.float32)
        self.__single_shot = img_as_float(self.single_shot).astype(np.float32)


    def __frequency_filter(self, img: np.ndarray, low_cutoff: float = 0.45, 
                       high_cutoff: float = 0.55) -> np.ndarray:
        if img.max() > 1.0:
            img = img_as_float(img)
        
        fft_avg = np.fft.fft2(self.avg_shot)
        fft_avg_shift = np.fft.fftshift(fft_avg)
        mag_avg = np.abs(fft_avg_shift)
        
        low_threshold = np.percentile(mag_avg, low_cutoff * 100)
        high_threshold = np.percentile(mag_avg, high_cutoff * 100)
        mask = np.ones_like(mag_avg, dtype=np.float32)
        
        """
        # Option 1: Binary mask (simple but can cause ringing)
        # mask = (mag_avg >= low_threshold) & (mag_avg <= high_threshold)
        """
    
        # Option 2: Smooth transition (better for reducing ringing)
        mask = np.clip((mag_avg - low_threshold) / (high_threshold - low_threshold), 0, 1)
        
        fft_single = np.fft.fft2(img)
        fft_single_shift = np.fft.fftshift(fft_single)
        fft_filtered_shift = fft_single_shift * mask.astype(np.complex64)
        
        # Inverse FFT
        fft_filtered = np.fft.ifftshift(fft_filtered_shift)
        filtered_image = np.fft.ifft2(fft_filtered)
        filtered_image = np.abs(filtered_image)
        
        return np.clip(filtered_image, 0, 1)

    
    def __guided_filter(self, img:np.ndarray, radius:int=15, 
                        eps:float=1e-6, strength:float=1.0
                        ) -> np.ndarray:
        """
        Apply guided filter to the input image using the average shot as guidance.

        :param img: Description
        :type img: np.ndarray
        :param radius: Description
        :type radius: int
        :param eps: Description
        :type eps: float
        :param strength: Description
        :type strength: float
        :return: Description
        :rtype: ndarray
        """
        # Apply guided filter
        if len(self.__avg_shot.shape) == 2: 
            filtered = cv2.ximgproc.guidedFilter(
                guide=self.__avg_shot,
                src=img,
                radius=radius,
                eps=eps
            )
        else:  # Color
            filtered = np.zeros_like(self.__single_shot)
            for i in range(3):
                filtered[:, :, i] = cv2.ximgproc.guidedFilter(
                    guide=self.__avg_shot[:, :, i],
                    src=img[:, :, i],
                    radius=radius,
                    eps=eps
                )
        
        if strength < 1.0:
            filtered = strength * filtered + (1 - strength) * self.__single_shot
        return np.clip(filtered, 0, 1)


    def filter(self) -> np.ndarray:
        """
        Enhance the single shot image using guided filtering and frequency domain filtering.
        """

        filtered_image = self.__guided_filter(self.__single_shot, radius=20, eps=1e-6, strength=0.5)
        filtered_image = self.__frequency_filter(filtered_image, low_cutoff=0.4, high_cutoff=0.7)
        return img_as_ubyte(filtered_image)


def match_histograms(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """
    Match the histogram of the source image to that of the reference image.
    """
    src_hist = cv2.calcHist([src], [0], None, [512], [0, 256])
    ref_hist = cv2.calcHist([ref], [0], None, [512], [0, 256])
    
    src_hist /= src_hist.sum()
    ref_hist /= ref_hist.sum()
    
    src_cdf = np.cumsum(src_hist)
    ref_cdf = np.cumsum(ref_hist)
    
    # Create lookup table with interpolation for smoother matching
    lookup_table = np.zeros(256, dtype=np.uint8)
    
    for i in range(256):
        
        cdf_val = src_cdf[i * 2] 
        idx = np.searchsorted(ref_cdf, cdf_val)
        if idx > 0 and idx < len(ref_cdf) - 1:
            # Interpolate between adjacent values
            lower_val = (idx - 1) / 2
            upper_val = idx / 2
            weight = (cdf_val - ref_cdf[idx-1]) / (ref_cdf[idx] - ref_cdf[idx-1] + 1e-10)
            lookup_table[i] = np.clip(lower_val * (1 - weight) + upper_val * weight, 0, 255)
        else:
            lookup_table[i] = np.clip(idx / 2, 0, 255)
    
    # Apply the lookup table
    matched = lookup_table[src].astype(np.uint8)
    src_grad = cv2.Laplacian(src, cv2.CV_64F)
    matched_grad = cv2.Laplacian(matched, cv2.CV_64F)
    
    # Only preserve gradients where they're strong in the original
    grad_mask = np.abs(src_grad) > np.percentile(np.abs(src_grad), 75)
    enhanced = matched.copy().astype(np.float64)
    enhanced[grad_mask] = 0.7 * enhanced[grad_mask] + 0.3 * src[grad_mask].astype(np.float64)
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def blob_enhance_tophat(img, se_radius=7):
    gray = img if img.ndim==2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*se_radius+1, 2*se_radius+1))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, se)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(tophat)


if __name__ == "__main__":
    source = "data\experimental_data\Source\Ref1_06042025.png"
    target = "data\experimental_data\Source\Ref1_06042025_0000.png"

    source2 = "data\experimental_data\Target\Run2_06042025.png"
    target2 = "data\experimental_data\Target\Run2_06042025_0000.png"

    avg_source = cv2.imread(source, cv2.IMREAD_GRAYSCALE)
    single_source = cv2.imread(target, cv2.IMREAD_GRAYSCALE)

    enhancer_source = SingleShotEnhancer(avg_shot=avg_source, single_shot=single_source)
    enhanced = enhancer_source.filter()

    plt.figure(figsize=(18, 6))

    # Original Single Shot
    plt.subplot(1, 3, 1)
    plt.imshow(single_source, cmap='gray')
    plt.axis('off')

    # Enhanced Image
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced, cmap='gray')
    plt.axis('off')

    # Averaged Reference
    plt.subplot(1, 3, 3)
    plt.imshow(avg_source, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


