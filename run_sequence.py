from src.Scipy_Hough_TM import HoughTM
import cv2
import matplotlib.pyplot as plt


if __name__ == "__main__":
    source = cv2.imread("data\experimental_data\Source\Ref1_06042025_0000.png", cv2.IMREAD_GRAYSCALE)
    source_avg = cv2.imread("data\experimental_data\Source\Ref1_06042025.png", cv2.IMREAD_GRAYSCALE)
    target = cv2.imread("data\experimental_data\Target\Run1_06042025_0000.png", cv2.IMREAD_GRAYSCALE)
    target_avg = cv2.imread("data\experimental_data\Target\Run1_06042025.png", cv2.IMREAD_GRAYSCALE)

    target_seq = [
        cv2.imread("data\experimental_data\Target\Run2_06042025_0000.png", cv2.IMREAD_GRAYSCALE)#,
        # cv2.imread("data\experimental_data\Target\Run3_06042025_0000.png", cv2.IMREAD_GRAYSCALE)
        ]
    target_avg_seq = [
        cv2.imread("data\experimental_data\Target\Run2_06042025.png", cv2.IMREAD_GRAYSCALE)#,
        # cv2.imread("data\experimental_data\Target\Run3_06042025.png", cv2.IMREAD_GRAYSCALE)
        ]
    
    solver = HoughTM(
        ref=source,
        ref_avg=source_avg,
        mov=target,
        mov_avg=target_avg,
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
    # solver.plot_intersections()
    solver.plot_fields(dt=2e-6, pix_to_world=0.000039604)