from src.Scipy_Hough_TM import HoughTM


if __name__ == "__main__":
    source = "data\experimental_data\Source\Ref1_06042025_0000.png"
    source_avg = "data\experimental_data\Source\Ref1_06042025.png"
    target = "data\experimental_data\Target\Run1_06042025_0000.png"
    target_avg = "data\experimental_data\Target\Run1_06042025.png"

    target_seq = [
        "data\experimental_data\Target\Run2_06042025_0000.png",
        "data\experimental_data\Target\Run3_06042025_0000.png"
        ]
    target_avg_seq = [
        "data\experimental_data\Target\Run2_06042025.png",
        "data\experimental_data\Target\Run3_06042025.png"
        ]
    
    solver = HoughTM(
        path_ref=source,
        path_ref_avg=source_avg,
        path_mov=target,
        path_mov_avg=target_avg,
        num_lines=(9, 11),
        slope_thresh=(10, 1)
    )

    winSize = (61, 61)
    maxLevel = 5
    epsilon = 0.001
    iteration = 100
    solver.set_optical_flow_params(win_size=winSize, max_level=maxLevel, epsilon=epsilon, iteration=iteration)

    hough_threshold = 0.2
    solver.set_hough_params(density=10, threshold=hough_threshold)

    solver.sequence_solve(target_seq, target_avg_seq)
    solver.plot_intersections()
    solver.plot_fields(dt=3e-6, pix_to_world=0.000039604)
    # solver.plot_fields(dt=1, pix_to_world=1)