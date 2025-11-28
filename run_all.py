from utility.py_import import plt, np, os, cv2
from src.Scipy_Hough_TM import HoughTM
from utility.tif_reader import tifReader


def main(src, tar):
    src = tifReader(src)
    tar = tifReader(tar)
    max_size = max(src.get_tif_size(), tar.get_tif_size())

    avg_src = src.average_all_tif()
    avg_tar = tar.average_all_tif()

    for i in range(max_size):
        print(f"Processing frame {i+1}/{max_size}")
        cur_src = src.get_image(i)
        cur_tar = tar.get_image(i)

        solver = HoughTM(
            path_ref=cur_src,
            path_ref_avg=avg_src,
            path_mov=cur_tar,
            path_mov_avg=avg_tar,
            num_lines=(9, 11),
            slope_thresh=(10, 1)
        )

        solver.solve()
        solver.plot_intersections()
        solver.plot_fields(dt=1e-6, pix_to_world=0.000039604)


if __name__ == "__main__":

    source = r"data\experimental_data\Source\Ref1_06042025.tif"
    target = r"data\experimental_data\Target\Run2_06042025.tif"

    main(source, target)

    