from utility.py_import import plt, np, os, cv2
from src.Scipy_Hough_TM import HoughTM


if __name__ == "__main__":
    test_dir = "data/Experimental_Data/Source/frame_2.png"
    image_dir = "data/Experimental_Data/Target/frame_2_2us.png"

    solver = HoughTM(
        test_dir, image_dir, num_lines=11, fwhm=4, 
        temp_scale=0.67, uncertainty=1, num_interval=35, 
        verbose=False
        )

    solver.solve()
    solver.plot_intersections()
    solver.plot_fields(dt=1)