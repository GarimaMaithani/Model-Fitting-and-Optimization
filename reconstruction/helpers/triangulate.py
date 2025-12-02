# -*- coding: utf-8 -*-
import numpy as np

def triangulate(x1, x2, P1, P2):
    """
    x1,x2: Nx2 (normalized) points; P1,P2: 3x4
    returns: Nx3 points in Euclidean coords
    """
    N = x1.shape[0]
    X = np.zeros((N,4))
    for i in range(N):
        u1,v1 = x1[i]; u2,v2 = x2[i]
        A = np.vstack([
            u1*P1[2]-P1[0],
            v1*P1[2]-P1[1],
            u2*P2[2]-P2[0],
            v2*P2[2]-P2[1]
        ])
        _,_,Vt = np.linalg.svd(A)
        Xi = Vt[-1]; Xi /= Xi[-1]
        X[i] = Xi
    return X[:, :3]

