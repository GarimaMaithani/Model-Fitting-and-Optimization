# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from reconstruction.helpers.ba_linearize import linearize_reproj_err, update_solution, compute_reprojection_error

def run(P1_original, P2_original, X3_best, inliers, x1, x2, iters=10, gamma=1e-10):
    P = [P1_original, P2_original]
    U = np.vstack([X3_best[inliers].T, np.ones((1, len(inliers)))])  # 4xN
    u_meas = {0: x1[:, inliers], 1: x2[:, inliers]}

    current = P[:]
    objective = []

    for it in range(iters):
        r, J = linearize_reproj_err(current, U, u_meas)
        grad = J.T @ r
        delta = -gamma * grad
        current, U = update_solution(delta, current, U)
        _, res = compute_reprojection_error(current, U, u_meas)
        val = np.sqrt(np.sum(res))
        objective.append(val)
        print(f"Iteration {it+1}: Objective value = {val:.6f}")

    plt.figure(); plt.plot(range(1, iters+1), objective, '-o')
    plt.xlabel("Iteration"); plt.ylabel("Objective Value"); plt.title("Steepest Descent Objective")
    # final RMS split per view
    r_final, res_final = compute_reprojection_error(current, U, u_meas)
    N = u_meas[0].shape[1]*2
    rms1 = np.sqrt(np.sum(res_final[:N]) / u_meas[0].shape[1])
    rms2 = np.sqrt(np.sum(res_final[N:]) / u_meas[1].shape[1])
    print(f"Final RMS error 1: {rms1:.6f}")
    print(f"Final RMS error 2: {rms2:.6f}")
    plt.show()
    return current, U

if __name__ == "__main__":
    print("This script is meant to be called from robust_essential_matrix_estimation.py once you have P1,P2,X,inliers.")

