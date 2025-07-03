import os
import numpy as np
import tifffile
from skimage import io
from PIL import Image

def merge_tiff(folder_path, output_filename='merged.tif'):
    image_paths = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(folder_path, f))
    
    if not image_paths:
        raise ValueError("No valid image files found in the folder")

    first_img = io.imread(image_paths[0])
    if len(first_img.shape) == 3 and first_img.shape[2] in [3, 4]: 
        first_img = np.dot(first_img[...,:3], [0.2989, 0.5870, 0.1140])
    h, w = first_img.shape
    dtype = first_img.dtype

    num_images = len(image_paths)
    stack = np.zeros((num_images, h, w), dtype=dtype)
    stack[0,:,:] = first_img

    for i, path in enumerate(image_paths[1:], 1):
        try:
            img = io.imread(path)
            if len(img.shape) == 3 and img.shape[2] in [3, 4]:
                img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140]).astype(dtype)
            if img.shape != (h, w):
                img = Image.fromarray(img).resize((w, h))
                img = np.array(img)
            stack[i,:,:] = img
        except Exception as e:
            print(f"Could not read {path}: {str(e)}")
            stack[i,:,:] = 0

    tifffile.imwrite(
        os.path.join(folder_path, output_filename),
        stack,
        photometric='minisblack',
        compression='zlib'
    )
    
    print(f"Saved {num_images} images to {output_filename} (dimensions: {stack.shape})")

if __name__ == "__main__":

    base_dir = os.path.dirname(os.path.abspath(__file__))

    for snr in [2, 4, 8, 16]:
        # Construct full paths using the base directory
        # src_dir = os.path.join(base_dir, "src", f"SNR_{snr}")
        uniform_dir = os.path.join(base_dir, "uniform", f"SNR_{snr}")
        poiseuille_dir = os.path.join(base_dir, "poiseuille", f"SNR_{snr}")
        # lamb_oseen_dir = os.path.join(base_dir, "lamb_oseen", f"SNR_{snr}")
        
        # Output files should also go in the base directory
        # merge_tiff(src_dir, os.path.join(base_dir, f"src_{snr}.tif"))
        merge_tiff(uniform_dir, os.path.join(base_dir, f"uniform_{snr}.tif"))
        merge_tiff(poiseuille_dir, os.path.join(base_dir, f"poiseuille_{snr}.tif"))
        # merge_tiff(lamb_oseen_dir, os.path.join(base_dir, f"lamb_oseen_{snr}.tif"))