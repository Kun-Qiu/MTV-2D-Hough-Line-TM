from utility.py_import import plt, np, os, cv2
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar
from skimage.util import img_as_float


def compute_rotation_angle(image1, image2):
    """
    Compute rotation angle (in degrees) between two images using phase correlation in polar coordinates.
    """
    image1 = img_as_float(image1)
    image2 = img_as_float(image2)

    radius = min(image1.shape) // 2
    polar1 = warp_polar(image1, radius=radius, channel_axis=-1)
    polar2 = warp_polar(image2, radius=radius, channel_axis=-1)

    shift, _, _ = phase_cross_correlation(polar1, polar2, normalization=None)
    angle_shift = shift[0]
    angle_deg = angle_shift * (360 / polar1.shape[0])
    
    return angle_deg


if __name__ == "__main__":
    test_dir = "data/Synthetic_Data/Image/SNR_4"
    image_dir = "data/Synthetic_Data/Image/SNR_4"

    img_type = {
        "lamb_oseen": "displaced_lamb_oseen.png"
    }

    for i in range(1):
        src_path = os.path.join(image_dir, f"{i}/src.png") 

        for key, value in img_type.items():
            img_path = os.path.join(image_dir, f"{i}/{value}")

            original_img = plt.imread(src_path)
            dewarped_img = plt.imread(img_path)

            rotation_angle = compute_rotation_angle(original_img, dewarped_img)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            
            # Display original image
            ax1.imshow(original_img, cmap='gray')
            ax1.set_title('Original Image')
            ax1.axis('off')
            
            # Display dewarped image
            ax2.imshow(dewarped_img, cmap='gray')
            ax2.set_title(f'Dewarped Image\nRotation: {rotation_angle:.2f}°')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()