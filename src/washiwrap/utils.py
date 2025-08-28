import math
import numpy as np

def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n == 0 else v / n

def rotation2d(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s],[s, c]], dtype=float)

def reflect_along_unit_axis(u: np.ndarray) -> np.ndarray:
    ux, uy = u
    return np.array([[2*ux*ux - 1, 2*ux*uy],
                     [2*ux*uy,     2*uy*uy - 1]], dtype=float)
