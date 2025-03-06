import sys
import os

# Construct the full path to the directory containing the package
project_path = '/Users/hofmannc/git/apolune'

# Add the directory to sys.path
sys.path.append(project_path)

import numpy as np
from numba import jit

# JIT-decorated function
@jit('float64(float64[::1], float64[::1], float64[::1])', nopython=True, nogil=True, fastmath=True)
def fun(a, b, c):
    # Example computation
    return np.sum(a * b + c)

# Main function
@jit(nopython=True, nogil=True, fastmath=True)
def main(X):
    a = X[0, :]  # Slice row 0
    b = X[1, :]  # Slice row 1
    c = X[2, :]  # Slice row 2
    
    print(a.flags['C_CONTIGUOUS'])  # Should be True for C-contiguous arrays
    print(a.flags['F_CONTIGUOUS'])  # Should be False (unless explicitly column-major)
    
    result = fun(a, b, c)  # Call JIT-compiled `fun`
    return result

# Example usage
X = np.random.rand(3, 1000)  # 2D array with 3 rows
result = main(X)
print("Result:", result)

