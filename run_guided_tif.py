#!/usr/bin/env python3
"""
Command line interface for HoughTM class
"""

import argparse
import sys

from utility.tif_reader import tifReader


def main():
    parser = argparse.ArgumentParser(description='Run Hough Transform-based Template Matching')
    
    # Required arguments
    parser.add_argument('--ref', required=True, help='Path to reference image (.tif)')
    parser.add_argument('--mov', required=True, help='Path to moving image (.tif)')
    parser.add_argument('--num-lines', required=True, type=int, nargs=2, 
                       help='Number of lines to detect (int int)')
    parser.add_argument('--slope-thresh', required=True, type=float, nargs=2,
                       help='Slope threshold (float float)')
    
    # Optional arguments
    parser.add_argument('--optimize', action='store_true', help='Enable template optimization')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    try:
        from src.Scipy_Hough_TM import HoughTM
        from utility.tif_reader import tifReader
    except ImportError:
        print("Error: Could not import required class.")
        sys.exit(1)
    
    source_tif = tifReader(args.ref)
    moving_tif = tifReader(args.mov)

    print("TIFF files loaded successfully.")
    print(source_tif.get_tif_size(), moving_tif.get_tif_size())

    # # Create and run HoughTM instance
    # hough_tm = HoughTM(
    #     path_ref=source_tif,
    #     path_mov=moving_tif,
    #     num_lines=tuple(args.num_lines),
    #     slope_thresh=tuple(args.slope_thresh),
    #     optimize=args.optimize,
    #     verbose=args.verbose,
    #     path_ref_avg=args.ref_avg,
    #     path_mov_avg=args.mov_avg
    # )
    
    # # Add your processing logic here
    # # For example:
    # # result = hough_tm.process()
    # # print(f"Result: {result}")
    
    # print("HoughTM instance created successfully!")
    # print(f"Reference: {args.ref}")
    # print(f"Moving: {args.mov}")
    # print(f"Number of lines: {args.num_lines}")
    # print(f"Slope threshold: {args.slope_thresh}")

if __name__ == '__main__':
    main()