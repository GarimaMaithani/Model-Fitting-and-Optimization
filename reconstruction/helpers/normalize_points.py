# -*- coding: utf-8 -*-
import numpy as np

def normalize_points(points):
    d, n = points.shape
    centroid = np.mean(points[:d-1], axis=1)
    shifted = points[:d-1] - centroid[:, None]
    avg_dist = np.mean(np.sqrt(np.sum(shifted**2, axis=0)))
    scale = np.sqrt(2) / avg_dist

    T = np.eye(d)
    T[:d-1, :d-1] *= scale
    T[:d-1, -1] = -scale * centroid

    points_h = np.vstack((shifted * scale, np.ones(n)))
    return points_h, T

