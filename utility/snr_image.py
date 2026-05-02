import numpy as np
from utility.tif_reader import tifReader

def estimate_snr(image_array, smooth_image=None):
    """
    Estimate SNR of an image using a reference smooth image.
    
    image_array : 2D numpy array (float32) - the noisy image
    smooth_image: 2D numpy array (float32) - reference clean/smooth image
                  if None, will fallback to median filter
    """
    sp = np.max(image_array)
    
    if smooth_image is not None:
        noise = image_array - smooth_image
        sigma = np.std(noise)
    else:
        from scipy.ndimage import median_filter
        smooth = median_filter(image_array, size=5)
        noise = image_array - smooth
        sigma = np.std(noise)
        
    snr = sp / (4 * sigma)
    return snr

if __name__ == "__main__":
    # June 4th experiment data
    # tif_paths = [
    #     r"HTV_Data\x=0\Ref1.tif",
    #     r"HTV_Data\x=0\1gs\Run1.tif",
    #     r"HTV_Data\x=0\1gs\Run2.tif",
    #     r"HTV_Data\x=0\1gs\Run3.tif",
    #     r"HTV_Data\x=0\2gs\Run4.tif",
    #     r"HTV_Data\x=0\2gs\Run5.tif",
    #     r"HTV_Data\x=0\2gs\Run6.tif",
    #     r"HTV_Data\x=10\Ref2.tif",
    #     r"HTV_Data\x=10\1gs\Run7.tif",
    #     r"HTV_Data\x=10\1gs\Run8.tif",
    #     r"HTV_Data\x=10\1gs\Run9.tif",
    #     r"HTV_Data\x=10\2gs\Run10.tif",
    #     r"HTV_Data\x=10\2gs\Run11.tif",
    #     r"HTV_Data\x=10\2gs\Run12.tif",
    #     r"HTV_Data\x=20\Ref3.tif",
    #     r"HTV_Data\x=20\1gs\Run13.tif",
    #     r"HTV_Data\x=20\1gs\Run14.tif",
    #     r"HTV_Data\x=20\2gs\Run17.tif",
    #     r"HTV_Data\x=20\2gs\Run18.tif"
    # ]

    #June 12 Experiment data
    # tif_paths = [
    #     r"HTV_Data\FRA\x=0\Ref1.tif",
    #     r"HTV_Data\FRA\x=0\Run1.tif",
    #     r"HTV_Data\FRA\x=0\Run4.tif",
    #     r"HTV_Data\FRA\x=0\Run10.tif",
    #     r"HTV_Data\FRA\x=25\Ref3.tif",
    #     r"HTV_Data\FRA\x=25\Run25.tif",
    #     r"HTV_Data\FRA\x=25\Run28.tif",
    #     r"HTV_Data\FRA\x=25\Run29.tif",
    #     r"HTV_Data\FRA\x=25\Run32.tif",
    #     r"HTV_Data\FRA\x=25\Run36.tif",
    #     r"HTV_Data\FRA\x=50\Ref2.tif",
    #     r"HTV_Data\FRA\x=50\Run13.tif",
    #     r"HTV_Data\FRA\x=50\Run16.tif",
    #     r"HTV_Data\FRA\x=50\Run17.tif",
    #     r"HTV_Data\FRA\x=50\Run20.tif",
    #     r"HTV_Data\FRA\x=50\Run24.tif"
    #     ]

    tif_paths = [
        r"HTV_Data\Stereo\A\Ref2.tif",
        r"HTV_Data\Stereo\A\Run4.tif",
        r"HTV_Data\Stereo\A\Ref4.tif",
        r"HTV_Data\Stereo\A\Run19.tif",
        r"HTV_Data\Stereo\B\Ref2.tif",
        r"HTV_Data\Stereo\B\Run4.tif",
        r"HTV_Data\Stereo\B\Ref4.tif",
        r"HTV_Data\Stereo\B\Run19.tif"
        ]
    
    for tif_path in tif_paths:
        tif = tifReader(tif_path)

        min_snr = np.inf
        max_snr = 0
        total_snr = 0

        avg_image = tif.average()      # Averaged image
        length = tif.length
        for i in range(length):
            image = tif.get_image(i)       # Single-shot image
            snr = estimate_snr(image, smooth_image=avg_image)
            total_snr += snr

            if snr < min_snr:
                min_snr = snr
            if snr > max_snr:
                max_snr = snr

    avg_snr = total_snr / length  # Calculate average SNR
    print(f"SNR: min={min_snr:.2f}, max={max_snr:.2f}, avg={avg_snr:.2f}")
