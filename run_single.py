from src.Scipy_Hough_TM import HoughTM
import matplotlib.pyplot as plt
import numpy as np
import os 
from tqdm import tqdm
from cython_build.PostProcessor import PostProcessor
import argparse
from utility.tif_reader import tifReader


##################################
##### USER DEFINED VARIABLES #####
##################################

WIN_SIZE = (61, 61)
MAX_LEVEL = 3
EPSILON = 0.01
ITERATION = 100
HOUGH_THRESHOLD = 0.2
HOUGH_DENSITY = 10
X_ORIGIN, Y_ORIGIN = 0, 0
X_SHIFT, Y_SHIFT = 50, 0
OUTPUT_FOLDER = "plots"
##################################


def parser_setup():
    parser = argparse.ArgumentParser(
        description="Hybrid Analysis Method for Single Time Frame",
        epilog="Usage: python run_single.py Ref1.tif Run1.tif 9 11 1e-6 0.000039604 --thresh 10 1 --interp 0 --num 3 --filter True"
    )

    parser.add_argument("ref", type=str, help="Path to the reference image")
    parser.add_argument("mov",  type=str, help="Path to the moving image")
    parser.add_argument("line", nargs=2, type=int, help="Number of lines")
    parser.add_argument("dt", type=float, help="Delay time between reference and moving image")
    parser.add_argument("pix_world", type=float, help="Conversion factor for pixel to world coordinates")
    parser.add_argument("--thresh", nargs=2, type=int, help="Slope threshold for Hough transform")
    parser.add_argument("--interp", type=int, default=2, help="Interpolation method")
    parser.add_argument("--num", default=0, type=int, help="Number of images to process")
    parser.add_argument("--filter", default=False, type=bool, help="Boolean to filter single shot")
    return parser


def solver_setup(ref, mov, num_lines, slope_thresh, interp, ref_avg=None, mov_avg=None):
    """
    Parameter setup for the solver
    """ 
    solver = HoughTM(
        ref=ref,
        ref_avg=ref_avg,
        mov=mov,
        mov_avg=mov_avg,
        num_lines=num_lines,
        interp=interp,
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
    ref_avg, mov_avg = (ref_tif.average(), mov_tif.average()) if args.filter else (None, None)
    
    img_shape = np.shape(ref_avg)
    processor = PostProcessor(img_shape[0], img_shape[1])

    print("Reference and moving image processed.")

    num = (min_length if args.num==0 else args.num)
    interpolation = args.interp
    slope_thresh = args.thresh if args.thresh else (10, 1)  
    
    for i in tqdm(range(num), desc="Processing images"):
        ref, mov = ref_tif.get_image(i), mov_tif.get_image(i)
        solver = solver_setup(
            ref, mov, num_lines, slope_thresh,
            interpolation, ref_avg, mov_avg
            )
        
        solver.solve()   
        solver.disp_field[:, :, 2:4] += np.array([X_SHIFT, Y_SHIFT])   
        sol_field = solver.get_fields(dt=args.dt, pix_to_world=args.pix_world)
        processor.update(sol_field[..., 4], sol_field[..., 5], sol_field[..., 6])

    print("Processing & Plotting Mean / RMS Fields")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    vx_mean, vy_mean, vort_mean = processor.get_mean()
    vx_std, vy_std, vort_std = processor.get_std()
    height, width = vx_mean.shape
    
    # Average Magnitude
    valid_points = np.array([solver.disp_field[i, j][:2] for i, j in solver.valid_ij])
    X = np.round(valid_points[:, 0])
    Y = np.round(valid_points[:, 1])

    within_bounds_mask = (
        (X >= 0) & (X < width) & 
        (Y >= 0) & (Y < height) &
        np.isfinite(X) & np.isfinite(Y)
        )
    
    valid_points = valid_points[within_bounds_mask]
    X = X[within_bounds_mask].astype(int)
    Y = Y[within_bounds_mask].astype(int)
    
    # Create mask
    X_full, Y_full = np.meshgrid(np.arange(vx_mean.shape[1]), np.arange(vx_mean.shape[0]))
    if X_ORIGIN != 0 and Y_ORIGIN != 0:
        X_shift_full = X_full - X_ORIGIN
        Y_shift_full = Y_full - Y_ORIGIN
        X_shift = X - X_ORIGIN
        Y_shift = Y - Y_ORIGIN
    else:
        X_shift_full = X_full
        Y_shift_full = Y_full
        X_shift = X
        Y_shift = Y

    mask = np.zeros_like(vx_mean, dtype=bool)
    mask[Y, X] = True

    vel_mean_mag = np.sqrt(vx_mean**2 + vy_mean**2)
    Vx = vx_mean[Y, X]
    Vy = -vy_mean[Y, X]

    Vel_mag = vel_mean_mag[Y, X]
    Vel_max_mag = np.nanmax(Vel_mag) if len(Vel_mag) > 0 else 1.0
    unit_Vx = Vx / Vel_max_mag
    unit_Vy = Vy / Vel_max_mag
     
    # In term of physical coordinates # Convert the x, y to mm 
    X_shift_full_physical = X_shift_full * args.pix_world 
    Y_shift_full_physical = Y_shift_full * args.pix_world 
    X_shift_physical = X_shift * args.pix_world 
    Y_shift_physical = Y_shift * args.pix_world 

    # Calculate extent in physical coordinates
    extent_physical = [
        np.min(X_shift_full_physical),
        np.max(X_shift_full_physical),  
        np.max(Y_shift_full_physical),  
        np.min(Y_shift_full_physical)   
        ]

    # List to store all figures for displaying at the end
    figures = []
    fig_titles = []

    # 1. Velocity Magnitude Plot
    fig1, ax = plt.subplots(figsize=(8, 6))
    mag_plot = ax.imshow(
        vel_mean_mag, 
        cmap='RdBu_r', 
        origin='upper',
        extent=extent_physical
        )
    ax.quiver(X_shift_physical, Y_shift_physical, unit_Vx, unit_Vy, color='black', scale=20)
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    cbar = fig1.colorbar(mag_plot, ax=ax, format='%.1e')
    ax.set_title("Velocity Magnitude (m/s)")

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'velocity_magnitude.png'), dpi=300, bbox_inches='tight')
    figures.append(fig1)
    fig_titles.append("Velocity Magnitude")

    # 2. Mean U Plot
    fig2, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    im0 = ax.imshow(vx_mean, cmap='jet', extent=extent_physical)
    ax.set_title('Mean U (m/s)')
    plt.colorbar(im0, ax=ax, shrink=0.8)

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'mean_u.png'), dpi=300, bbox_inches='tight')
    figures.append(fig2)
    fig_titles.append("Mean U")

    # 3. Mean V Plot
    fig3, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    im1 = ax.imshow(vy_mean, cmap='jet', extent=extent_physical)
    ax.set_title('Mean V (m/s)')
    plt.colorbar(im1, ax=ax, shrink=0.8)
    
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'mean_v.png'), dpi=300, bbox_inches='tight')
    figures.append(fig3)
    fig_titles.append("Mean V")

    # 4. Mean Vorticity Plot
    fig4, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    im2 = ax.imshow(vort_mean, cmap='jet', extent=extent_physical)
    ax.set_title('Mean ω (1/s)')
    plt.colorbar(im2, ax=ax, shrink=0.8)
    
    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")   
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'mean_vorticity.png'), dpi=300, bbox_inches='tight')
    figures.append(fig4)
    fig_titles.append("Mean Vorticity")

    # 5. RMS U Plot
    fig5, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    im3 = ax.imshow(vx_std, cmap='jet', extent=extent_physical)
    ax.set_title('RMS U\' (m/s)')
    plt.colorbar(im3, ax=ax, shrink=0.8)

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'rms_u.png'), dpi=300, bbox_inches='tight')
    figures.append(fig5)
    fig_titles.append("RMS U")

    # 6. RMS V Plot
    fig6, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    im4 = ax.imshow(vy_std, cmap='jet', extent=extent_physical)
    ax.set_title('RMS V\' (m/s)')
    plt.colorbar(im4, ax=ax, shrink=0.8)

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'rms_v.png'), dpi=300, bbox_inches='tight')
    figures.append(fig6)
    fig_titles.append("RMS V")

    # 7. RMS Vorticity Plot
    fig7, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(y=0, color='black', linewidth=1, linestyle='-', alpha=0.8)
    im5 = ax.imshow(vort_std, cmap='jet', extent=extent_physical)
    ax.set_title('RMS ω\' (1/s)')
    plt.colorbar(im5, ax=ax, shrink=0.8)

    ax.set_xticks(ax.get_xticks())
    ax.set_yticks(ax.get_yticks())
    ax.set_xticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_xticks()])
    ax.set_yticklabels([f'{float(tick)*1000:.1f}' for tick in ax.get_yticks()])
    ax.set_xlabel("Streamwise Direction (mm)")
    ax.set_ylabel("Spanwise Direction from Jet (mm)")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, 'rms_vorticity.png'), dpi=300, bbox_inches='tight')
    figures.append(fig7)
    fig_titles.append("RMS Vorticity")

    print(f"All plots have been saved to '{OUTPUT_FOLDER}' folder")
    plt.show()