# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye as speye
from reconstruction.helpers.ba_linearize import linearize_reproj_err, update_solution, compute_reprojection_error

def run(P1_original, P2_original, X3_best, inliers, x1, x2, iters=10, lamb=1e-2):
    P = [P1_original, P2_original]
    U = np.vstack([X3_best[inliers].T, np.ones((1, len(inliers)))])
    u_meas = {0: x1[:, inliers], 1: x2[:, inliers]}

    current = P[:]
    values = []
    for it in range(iters):
        r, J = linearize_reproj_err(current, U, u_meas)
        g = J.T @ r
        H = J.T @ J + lamb * np.eye(J.shape[1])
        delta = -np.linalg.solve(H, g)
        current, U = update_solution(delta, current, U)
        r_new, res = compute_reprojection_error(current, U, u_meas)
        val = np.sqrt(np.sum(r_new**2))
        values.append(val)
        print(f"Iteration {it+1}: Objective value = {val:.6f}")

    plt.figure(); plt.plot(range(1, iters+1), values, '-o')
    plt.xlabel("Iteration"); plt.ylabel("Objective Value"); plt.title("LM Objective")
    # RMS
    N = u_meas[0].shape[1]*2
    rms1 = np.sqrt(np.sum(res[:N]) / u_meas[0].shape[1])
    rms2 = np.sqrt(np.sum(res[N:]) / u_meas[1].shape[1])
    print(f"Final RMS error 1: {rms1:.6f}")
    print(f"Final RMS error 2: {rms2:.6f}")
    plt.show()
    return current, U

if __name__ == "__main__":
    print("Call run(...) with data prepared by robust_essential_matrix_estimation.py")

