# -*- coding: utf-8 -*-
import numpy as np
from .reprojection import project_points, pextend
from .se3 import apply_se3_update

def compute_reprojection_error(P_list, U4, u_meas):
    r_list = []
    for i, P in enumerate(P_list):
        x = project_points(P, U4)
        r = (x[:2] - u_meas[i])            # 2xN
        r_list.append(r.reshape(-1,1))
    r_full = np.vstack(r_list).ravel()
    res_nonneg = (r_full**2)
    return r_full, res_nonneg

def numeric_jacobian(P_list, U4, u_meas, eps=1e-6):
    """
    Parameters: [point_delta(3*N-1); cam_delta(6*M)]
    """
    n_pts = U4.shape[1]
    m_cams = len(P_list)
    dim = (3*n_pts - 1) + 6*m_cams
    r0, _ = compute_reprojection_error(P_list, U4, u_meas)
    J = np.zeros((r0.size, dim))
    for k in range(dim):
        e = np.zeros(dim); e[k] = eps
        Pn, Un = apply_se3_update(P_list, U4, e)
        r1, _ = compute_reprojection_error(Pn, Un, u_meas)
        J[:, k] = (r1 - r0) / eps
    return r0, J

def linearize_reproj_err(P_list, U4, u_meas):
    return numeric_jacobian(P_list, U4, u_meas)

def update_solution(delta, P_list, U4):
    return apply_se3_update(P_list, U4, delta)

