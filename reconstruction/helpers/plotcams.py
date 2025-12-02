# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def null_space(A, tol=1e-12):
    u, s, vh = np.linalg.svd(A)
    rank = (s > tol).sum()
    return vh[rank:].T

def plotcams(P_list, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    c, v = [], []
    for P in P_list:
        C = null_space(P).flatten()
        C /= C[-1]
        c.append(C)
        vi = np.sign(np.linalg.det(P[:, :3])) * P[2, :3]
        vi = vi / np.linalg.norm(vi)
        v.append(vi)

    c = np.array(c).T
    v = np.array(v).T
    ax.quiver(c[0], c[1], c[2], v[0], v[1], v[2], color='r')
    return ax


