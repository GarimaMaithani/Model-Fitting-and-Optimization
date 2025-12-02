# -*- coding: utf-8 -*-
import numpy as np

def pflat(X: np.ndarray) -> np.ndarray:
    """
    Projects homogeneous coordinates to 2D/3D by dividing by the last row.
    """
    return X / X[-1]
