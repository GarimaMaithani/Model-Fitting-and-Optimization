# -*- coding: utf-8 -*-
import numpy as np

def pflat(X: np.ndarray) -> np.ndarray:
    return X / X[-1]

def pextend(X: np.ndarray) -> np.ndarray:
    return np.vstack([X, np.ones((1, X.shape[1]))])

def project_points(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """P: 3x4, X: 4xN -> 3xN normalized"""
    x = P @ X
    return x / x[2]


