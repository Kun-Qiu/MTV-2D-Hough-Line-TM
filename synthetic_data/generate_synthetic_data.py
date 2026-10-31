import os
import sys
import json
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

# When run as a script, put the project root on the path so the synthetic_data package resolves
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synthetic_data.src import flow_pattern as fp
from synthetic_data.src.grid_pattern import create_centered_grid, grid_intensity, add_noise, to_uint8
from synthetic_data.src.displace_grid import inverse_map, stereo_source_coords, stereo_flow


def preview_grid():
    """
    Displays a single noisy grid image for a quick visual check of the grid parameters.
    """
    fwhm        = 4            # Full width at half maximum for the Gaussian lines
    spacing     = 10           # Reduced spacing for denser lines
    angle       = 150          # Angle for intersecting lines
    image_size  = (256, 256)   # Size of the image
    num_lines   = 5            # Number of lines
    snr         = 8            # SNR value

    # Generate the grid image
    image = create_centered_grid(
        image_size, fwhm, spacing, angle,
        line_intensity=0.5, num_lines=num_lines, snr=snr,
        rng=np.random.default_rng(0)
        )

    plt.figure(figsize=(6, 6))
    plt.title("Centered Gaussian Grid Image")
    plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()


def generate_dataset(snrs=(4,), num_lines=9, angle=90):
    """
    Generates source / displaced grid image pairs (plain and stereo) with their ground-truth
    displacement fields for every SNR level into ./Image/SNR_<snr>/<set>/.

    :param snrs         :   SNR levels to generate
    :param num_lines    :   Number of lines in each direction
    :param angle        :   Angle for intersecting lines (degrees)
    """
    fwhm           = 10             # Full width at half maximum for the Gaussian lines
    spacing        = 30             # Reduced spacing for denser lines
    image_size     = (512, 512)     # Size of the image
    line_intensity = 0.5            # Peak line intensity above background, range(0, 1)
    background     = 30             # Background intensity, keeps noise from being clipped
    num_sets       = 10
    seed           = 0

    pwd = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(pwd, "Image")
    os.makedirs(base_dir, exist_ok=True)

    # Every image is rendered analytically at its exact source coordinates (no resampling), and
    # noise is added independently to every saved frame so frames never share a noise realisation.
    h, w = image_size
    center = ((w - 1) / 2, (h - 1) / 2)

    def grid(X, Y):
        return grid_intensity(X, Y, fwhm, spacing, angle, center,
                              line_intensity, num_lines, background)

    X, Y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    stereo_x, stereo_y = stereo_source_coords(image_size)
    clean_src = grid(X, Y)
    clean_stereo_src = grid(stereo_x, stereo_y)
    signal = 255.0 * line_intensity

    for j, snr in enumerate(snrs):
        snr_dir = os.path.join(base_dir, f"SNR_{snr}")
        os.makedirs(snr_dir, exist_ok=True)

        params = dict(
            fwhm=fwhm, spacing=spacing, angle=angle, image_size=image_size,
            num_lines=num_lines, line_intensity=line_intensity, background=background,
            snr=snr, num_sets=num_sets, seed=seed
            )
        with open(os.path.join(snr_dir, "params.json"), "w") as f:
            json.dump(params, f, indent=4)

        for i in range(num_sets):
            iter_dir = os.path.join(snr_dir, f"{i}")
            os.makedirs(iter_dir, exist_ok=True)

            # Flow fields depend only on (seed, set) so they are identical across SNR levels
            flow_rng = np.random.default_rng([seed, i])
            noise_rng = np.random.default_rng([seed, j, i])

            def save_noisy(name, clean):
                noisy = add_noise(clean, snr, signal=signal, rng=noise_rng)
                cv2.imwrite(os.path.join(iter_dir, name), to_uint8(noisy))

            save_noisy("src.png", clean_src)
            save_noisy("stereo_src.png", clean_stereo_src)

            flow_fields = {
                "poiseuille": fp.poiseuille_flow(
                    flow_rng.uniform(-30, 30), shape=image_size,
                    filename=os.path.join(iter_dir, "poiseuille_flow"),
                    show=False
                ),
                "uniform": fp.uniform_flow(
                    flow_rng.uniform(-10, 10), flow_rng.uniform(-10, 10), shape=image_size,
                    filename=os.path.join(iter_dir, "uniform_flow"),
                    show=False
                ),
                "lamb_oseen": fp.lamb_oseen_vortex(
                    scale=flow_rng.uniform(200, 300), shape=image_size,
                    filename=os.path.join(iter_dir, "lamb_oseen_flow"),
                    show=False
                )
            }

            for flow_name, flow_field in flow_fields.items():
                displaced = grid(*inverse_map(flow_field))

                # Stereo view of the displaced world, not a displacement of the stereo view
                displaced_stereo = grid(*inverse_map(flow_field, points=(stereo_x, stereo_y)))
                np.save(os.path.join(iter_dir, f"{flow_name}_flow_stereo.npy"),
                        stereo_flow(flow_field))

                save_noisy(f"displaced_{flow_name}.png", displaced)
                save_noisy(f"displaced_stereo_{flow_name}.png", displaced_stereo)

    print("All images have been saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic MTV grid image generator")
    parser.add_argument("--preview", action="store_true",
                        help="show a single grid image instead of generating the dataset")
    parser.add_argument("--snr", type=float, nargs="+", default=[4],
                        help="SNR levels to generate (default: 4)")
    parser.add_argument("--num-lines", type=int, default=9,
                        help="number of lines in each direction (default: 9)")
    parser.add_argument("--angle", type=float, default=90,
                        help="angle between intersecting lines in degrees (default: 90)")
    args = parser.parse_args()

    if args.preview:
        preview_grid()
    else:
        snrs = [int(s) if s.is_integer() else s for s in args.snr]  # keeps folder names like SNR_4
        generate_dataset(snrs=snrs, num_lines=args.num_lines, angle=args.angle)
