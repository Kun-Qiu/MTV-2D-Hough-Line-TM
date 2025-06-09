from utility.py_import import plt, np, os 
from src.T0_grid_struct import T0GridStruct
from data.Synthetic_Data.grid_image_generator import *
import time

if __name__ == "__main__":  
    fwhm       = 4          # Full width at half maximum for the Gaussian lines
    spacing    = 20         # Reduced spacing for denser lines
    angle      = 45         # Angle for intersecting lines
    image_size = (256, 256) # Size of the image
    num_lines  = 10         # Number of lines

    
    src = create_centered_grid(
        image_size, fwhm, spacing, angle, 
        line_intensity=0.5, num_lines=num_lines, snr=4
    )

    plt.imsave("src.png", src, cmap='gray')

    src_path = "src.png"
    solver = T0GridStruct(
        (10, 10), 
        src_path, 
        num_lines=10,
        solve_uncert=False,
        density=1
    )