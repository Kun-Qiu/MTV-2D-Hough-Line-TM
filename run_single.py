from src.Scipy_Hough_TM import HoughTM
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from cython_build.PostProcessor import PostProcessor
import argparse
from utility.tif_reader import tifReader


##################################
##### USER DEFINED VARIABLES #####
##################################

WIN_SIZE = (61, 61)
MAX_LEVEL = 5
EPSILON = 0.001
ITERATION = 100
HOUGH_THRESHOLD = 0.2
HOUGH_DENSITY = 10

##################################


def parser_setup():
    parser = argparse.ArgumentParser(
        description="Hybrid Analysis Method for Single Time Frame",
        epilog="Usage: python run_single.py Ref1.tif Run1.tif 9 11 10 1 1e-6 0.000039604 --num 3 --filter True"
    )

    parser.add_argument("ref", required=True, type=str, help="Path to the reference image")
    parser.add_argument("mov", required=True, type=str, help="Path to the moving image")
    parser.add_argument("line", nargs=2, required=True, type=int, help="Number of lines")
    parser.add_argument("slope", nargs=2, required=True, type=int, help="Slope of lines")
    parser.add_argument("dt", required=True, type=float, help="Delay time between reference and moving image")
    parser.add_argument("pix_world", required=True, type=float, help="Conversion factor for pixel to world coordinates")
    parser.add_argument("--num", default=0, type=int, help="Number of images to process")
    parser.add_argument("--filter", default=False, type=bool, help="Boolean to filter single shot")
    return parser


def solver_setup(ref, mov, num_lines, slope_thresh, ref_avg=None, mov_avg=None):
    """
    Parameter setup for the solver
    """ 
    solver = HoughTM(
        ref=ref,
        ref_avg=ref_avg,
        mov=mov,
        mov_avg=mov_avg,
        num_lines=num_lines,
        slope_thresh=slope_thresh
    )

    solver.set_optical_flow_params(
        win_size=WIN_SIZE, 
        max_level=MAX_LEVEL, 
        epsilon=EPSILON, 
        iteration=ITERATION
        )

    solver.set_hough_params(
        density=HOUGH_DENSITY, 
        threshold=HOUGH_THRESHOLD
        )

    return solver


if __name__ == "__main__":
    parser = parser_setup()
    args = parser.parse_args()

    print("Pre-Processing reference & moving image...")
    
    ref_tif, mov_tif = tifReader(args.ref), tifReader(args.mov)

    min_length = min(ref_tif.length, mov_tif.length)
    num_lines = (args.line[0], args.line[1])
    slope_thresh = (args.slope[0], args.slope[1])
    ref_avg, mov_avg = (ref_tif.average(), mov_tif.average()) if args.filter else (None, None)
    
    img_shape = np.shape(ref_avg)
    processor = PostProcessor(img_shape[0], img_shape[1])

    print("Reference and moving image processed.")

    skip = [5, 6, 10, 11, 45]
    num = (min_length if args.num==0 else args.num)

    for i in tqdm(range(num), desc="Processing images"):
        ref, mov = ref_tif.get_image(i), mov_tif.get_image(i)
        solver = solver_setup(
            ref, mov, num_lines, 
            slope_thresh, ref_avg, mov_avg
            )
        solver.solve()
        solver.plot_fields(dt=args.dt, pix_to_world=args.pix_world)

        if (i + 1) in skip:
            continue            

        sol_field = solver.get_fields(dt=args.dt, pix_to_world=args.pix_world)
        processor.update(sol_field[..., 4], sol_field[..., 5], sol_field[..., 6])

    print("Processing & Plotting Mean / RMS Fields")

    vx_mean, vy_mean, vort_mean = processor.get_mean()
    vx_std, vy_std, vort_std = processor.get_std()

    # Average Magnitude
    valid_points = np.array([solver.disp_field[i, j][:2] for i, j in solver.valid_ij])
    X = np.round(valid_points[:, 0]).astype(int)
    Y = np.round(valid_points[:, 1]).astype(int)

    vel_mean_mag = np.sqrt(vx_mean**2 + vy_mean**2)
    Vx = vx_mean[Y, X]
    Vy = vy_mean[Y, X]

    Vel_mag = vel_mean_mag[Y, X]
    unit_Vx = Vx / Vel_mag
    unit_Vy = Vy / Vel_mag

    # Velocity Magnitude Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    mag_plot = ax.imshow(vel_mean_mag, cmap='RdBu_r', origin='upper')
    ax.quiver(X, Y, unit_Vx, unit_Vy, color='red', scale=20)
    cbar = fig.colorbar(mag_plot, ax=ax, format='%.1e')
    ax.set_title("Magnitude (m/s)")


    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Mean Velocity Plots
    im0 = axs[0, 0].imshow(vx_mean, cmap='jet')
    axs[0, 0].set_title('Mean U (m/s)')
    plt.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(vy_mean, cmap='jet')
    axs[0, 1].set_title('Mean V (m/s)')
    plt.colorbar(im1, ax=axs[0, 1])

    im2 = axs[0, 2].imshow(vort_mean, cmap='jet')
    axs[0, 2].set_title('Mean ω​​ (1/s)')
    plt.colorbar(im2, ax=axs[0, 2])

    # RMS Velocity Plots
    im3 = axs[1, 0].imshow(vx_std, cmap='jet')
    axs[1, 0].set_title('U\' (m/s)')
    plt.colorbar(im3, ax=axs[1, 0])

    im4 = axs[1, 1].imshow(vy_std, cmap='jet')
    axs[1, 1].set_title('V\' (m/s)')
    plt.colorbar(im4, ax=axs[1, 1])

    im5 = axs[1, 2].imshow(vort_std, cmap='jet')
    axs[1, 2].set_title('ω​​\' (1/s)')
    plt.colorbar(im5, ax=axs[1, 2])

    plt.tight_layout()
    plt.show()