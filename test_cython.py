from utility.py_import import cv2, np, plt
from cython_build.parametric_x.parametric_X import ParametricX
from src.parametric_X import ParametricX as PX2
import time as t

def benchmark(iterations_list, fwhm=4):
    image_dir = os.path.abspath("data/Synthetic_Data/Image/SNR_1/0/displaced_lamb_oseen.png")
    img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    
    cython_times = []
    python_times = []
    speed_ratios = []
    
    for n_iter in iterations_list:
        # Benchmark Cython
        start = t.time()
        for _ in range(n_iter):
            parameter_X = ParametricX(
                (110, 123), 
                (np.pi/6, np.pi/6, 0.5, fwhm, 38*0.7), 
                img
            )
        end = t.time()
        cython_time = end - start
        cython_times.append(cython_time)
        
        # Benchmark Python
        start = t.time()
        for _ in range(n_iter):
            parameter_Y = PX2(
                (110, 123), 
                (np.pi/6, np.pi/6, 0.5, fwhm, 38*0.7), 
                img
            )
        end = t.time()
        python_time = end - start
        python_times.append(python_time)
        
        # Calculate speed ratio
        speed_ratio = python_time / cython_time
        speed_ratios.append(speed_ratio)
        
        print(f"Iterations: {n_iter:<10} | Cython: {cython_time:.6f}s | Python: {python_time:.6f}s | Ratio: {speed_ratio:.2f}x")
    
    return cython_times, python_times, speed_ratios

def plot_results(iterations_list, cython_times, python_times, speed_ratios):
    plt.figure(figsize=(12, 5))
    
    # Plot execution times
    plt.subplot(1, 2, 1)
    plt.plot(iterations_list, cython_times, 'b-o', label='Cython')
    plt.plot(iterations_list, python_times, 'r-o', label='Python')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('Execution time (s)')
    plt.title('Execution Time Comparison')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    # Plot speed ratio
    plt.subplot(1, 2, 2)
    plt.plot(iterations_list, speed_ratios, 'g-o')
    plt.xscale('log')
    plt.xlabel('Number of iterations')
    plt.ylabel('Speed Ratio (Python/Cython)')
    plt.title('Speed Ratio Across Iterations')
    plt.grid(True, which="both", ls="--")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage of the ParameterOptimizer 
    # with a ParametricX instance
    
    import os
    # Define the range of iterations to test
    image_dir = os.path.abspath("data/Synthetic_Data/Image/SNR_1/0/displaced_lamb_oseen.png")
    img = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE)
    # iterations_list = [1000000]

    # Run benchmarks
    # cython_times, python_times, speed_ratios = benchmark(iterations_list)

    # Plot results
    # plot_results(iterations_list, cython_times, python_times, speed_ratios)

    # Visualize one of the templates
    parameter_X = ParametricX(
        (110, 123), 
        (np.pi/6, np.pi/6, 0.5, 4, 38*0.7), 
        img
    )
    params = list([110, 123, np.pi/6, np.pi/6, 0.5, 4, 38*0.7])
    print(parameter_X.correlate(params))
    parameter_X.visualize()