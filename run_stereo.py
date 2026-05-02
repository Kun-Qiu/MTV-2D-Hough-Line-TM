from src.Scipy_Hough_TM import HoughTM
from utility.py_import import np, os
from utility.tif_reader import tifReader
from cython_build.PostProcessor import PostProcessor
from tqdm import tqdm
import argparse
import csv


##################################
##### USER DEFINED VARIABLES #####
##################################

WIN_SIZE = (32, 32)
MAX_LEVEL = 3
EPSILON = 0.01
ITERATION = 100
HOUGH_THRESHOLD = 0.2
HOUGH_DENSITY = 10
X_SHIFT, Y_SHIFT = 0, 0
OUTPUT_FOLDER = r"HTV_Data\Stereo\A\Recirculation_Region"
##################################


def parser_setup():
    """
    Argument parser setup
    """
    parser = argparse.ArgumentParser(
        description="Hybrid Analysis Method for Stereo Configuration",
        epilog="Usage: python run_stereo.py HTV_Data/Stereo/A/Ref2.tif HTV_Data/Stereo/A/Run4.tif 11 11 --thresh 10 1 --num 3"
    )

    parser.add_argument("ref", type=str, help="Path to the reference image")
    parser.add_argument("mov",  type=str, nargs='+', help="Path to the list moving image")
    parser.add_argument("line", nargs=2, type=int, help="Number of lines")
    parser.add_argument("--thresh", nargs=2, type=int, help="Slope threshold for Hough transform")
    parser.add_argument("--interp", type=int, default=0, help="Interpolation method")
    parser.add_argument("--num", default=0, type=int, help="Number of images to process")
    parser.add_argument("--filter", action="store_true", help="Boolean to filter single shot")
    parser.add_argument("--m", action="store_true", help="Manual correction")
    return parser


def solver_setup(ref, mov, num_lines, slope_thresh, interp, verbose, ref_avg=None, mov_avg=None):
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
        slope_thresh=slope_thresh,
        verbose=verbose
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
    
    ref_tif = tifReader(args.ref)
    ref_avg = ref_tif.average() if args.filter else None

    mov_tif_mat, mov_avg_mat = [], []    
    for mov_tif in args.mov:
        # Read moving image(s)
        mov_tif_obj = tifReader(mov_tif)
        mov_tif_mat.append(mov_tif_obj)
        mov_avg_mat.append(mov_tif_obj.average() if args.filter != None else None) 

    min_length = min(ref_tif.length, mov_tif_mat[0].length)
    num_lines = (args.line[0], args.line[1])
    interpolation = args.interp
    slope_thresh = args.thresh

    intersection_processor = PostProcessor(args.line[0], args.line[1])

    print("Reference and moving image processed.")

    num = (min_length if args.num==0 else args.num)
    verbose = args.m
    prev_ij = None

    idx_range = range(num)
    # idx_range = range(100, 200)
    for i in tqdm(idx_range, desc="Processing images"): 
        ref, mov = ref_tif.get_image(i), mov_tif_mat[0].get_image(i)

        if args.num == 1:
            ref, mov = ref_tif.average_indicies(10, 60), mov_tif_mat[0].average_indicies(10, 60)

        # Setup solver for each image pair
        solver = solver_setup(
            ref, mov, num_lines, slope_thresh, 
            interpolation, verbose, ref_avg, mov_avg_mat[0]
            )
        
        if len(mov_tif_mat) > 1:
            # Handle sequence of moving images
            mov_sequence = [mov_tif.get_image(i) for mov_tif in mov_tif_mat[1:]]
            solver.sequence_solver(mov_sequence, mov_avg_mat[1:], args.filter) 
        
        if prev_ij is not None:
            # Propogate the valid ij from previous manual correction
            solver.set_valid_ij(prev_ij)
        
        # Solve for displacement field
        solver.solve()
        solver.plot_intersections()

        if verbose and prev_ij is None:
            # After first manual correction, set verbose to False for next iterations
            verbose = False
            prev_ij = solver.get_valid_ij()

        solver.disp_field[..., 2:4] += np.array([X_SHIFT, Y_SHIFT])
        sol_field = solver.get_fields()

        # Update post-processor with current solution field
        intersection_fields = solver.disp_field[..., 2:4]
        intersection_processor.update(
            intersection_fields[..., 0], 
            intersection_fields[..., 1],
            np.zeros_like(intersection_fields[..., 0])
            )

    print("Processing & Plotting Mean / RMS Fields")

    # Fields for non-interpolated fields
    vx_mean_int, vy_mean_int, _ = intersection_processor.get_mean()
    vx_std_int, vy_std_int, _ = intersection_processor.get_std() 

    valid_points = np.array([
        solver.disp_field[i, j][:2] for i, j in solver.valid_ij
        ])

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    csv_path = os.path.join(OUTPUT_FOLDER, "intersection_measurement.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "x (pix)", "y (pix)", 
            "u (pix/s)", "v (pix/s)", 
            "u_rms (pix/s)", "v_rms (pix/s)"
            ])
        
        x_intersection = (valid_points[:, 0]) 
        y_intersection = (valid_points[:, 1])
        for x, y, u, v, ur, vr in zip(
            x_intersection, y_intersection, 
            vx_mean_int[tuple(zip(*solver.valid_ij))], vy_mean_int[tuple(zip(*solver.valid_ij))], 
            vx_std_int[tuple(zip(*solver.valid_ij))], vy_std_int[tuple(zip(*solver.valid_ij))]
            ):
            writer.writerow([x, y, u, v, ur, vr])

    print(f"Displacement fields saved to {csv_path}.")

