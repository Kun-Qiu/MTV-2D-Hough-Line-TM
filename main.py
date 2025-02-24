from Scipy_Hough_TM import HoughTM
import numpy as np

if __name__ == "__main__":
    image_location = "Synthetic_Data/Image/src.png"
    # image_location = "Experimental_Data/Source/frame_1.png"

    # dt_img_loc = "Experimental_Data/Target/frame_1_2us.png"
    dt_img_loc = "Synthetic_Data/Image/displaced_uniform.png"
    # dt_img_loc = "Synthetic_Data/Image/displaced_poiseuille.png"
    # dt_img_loc = "Synthetic_Data/Image/displaced_lamb_oseen.png"

    solver = HoughTM(image_location, dt_img_loc, num_lines=11)
    solver.solve()
    solver.plot_velocity_field()
