__author__ = "Kun Qiu"
__credits__ = ["Kun Qiu", "Marc C. Ramsey", "Robert W. Pitz"]
__maintainer__ = "Kun Qiu"
__email__ = "qiukun1234@gmail.com"

"""
This Python script is a direct transcription and adaptation of Mark Ramsey's original MATLAB 
implementation as described in the publication:

    "Template Matching for Improved Accuracy in Molecular Tagging Velocimetry"
    Marc C. Ramsey and Robert W. Pitz

All algorithmic methods, logic, and workflows closely follow the techniques outlined in 
the original work, with adjustments made only for syntax, language-specific practices, and 
minor optimization for Python execution.

Citation:
Ramsey, M. C., & Pitz, R. W. (Year). Template Matching for Improved Accuracy in Molecular 
Tagging Velocimetry. [Publication details].

This script is intended for academic, research, and educational purposes.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from scipy.signal import convolve2d
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import warnings
import matplotlib.pyplot as plt
