# -*- coding: utf-8 -*-
import numpy as np
from scipy.linalg import expm

# so(3) basis
Ba = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]], float)  # rot z
Bb = np.array([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], float)  # rot y
Bc = np.array([[0, 0, 0], [0, 0, 1], [0, -1, 0]], float)  # rot x

def apply_se3_update(P_list, U4, d):
    """
    Mirrors MATLAB ComputeStr.m behavior.
    d: stacked [point updates (3*N-1); camera updates (6*M)] (first coord fixed)
    """
    n_pts = U4.shape[1]
    m_cams = len(P_list)
    dpoint = np.zeros((3, n_pts))
    dpoint.flat[1:] = d[:3*n_pts-1]                 # keep first element zero as in MATLAB
    dcam = np.zeros((6, m_cams))
    dcam.flat[:] = d[3*n_pts-1:]
    # update points
    Unew = np.vstack([U4[:3] + dpoint, np.ones((1, n_pts))])
    # update cameras
    Pnew = []
    for i, P in enumerate(P_list):
        R0 = P[:, :3]
        t0 = P[:, 3]
        rot = expm(Ba*dcam[0,i] + Bb*dcam[1,i] + Bc*dcam[2,i]) @ R0
        t  = t0 + dcam[3:6, i]
        Pnew.append(np.hstack([rot, t.reshape(3,1)]))
    return Pnew, Unew

