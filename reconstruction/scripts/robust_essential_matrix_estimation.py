# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from reconstruction.helpers.fivepoint_solver import fivepoint_solver
from reconstruction.helpers.triangulate import triangulate
from reconstruction.helpers.reprojection import pextend

def run():
    data = loadmat("reconstruction/data/compEx2data.mat")
    x = [xi for xi in data["x"][0]]     # each 2xN
    K = data["K"]

    num_iter = 100
    best_inliers = []
    best_E = None
    best_num = 0
    thresh = 5.0

    for _ in range(num_iter):
        idx = np.random.choice(x[0].shape[1], 5, replace=False)
        x1n = np.vstack([x[0][:, idx], np.ones(5)])
        x2n = np.vstack([x[1][:, idx], np.ones(5)])
        x1n = np.linalg.inv(K) @ x1n
        x2n = np.linalg.inv(K) @ x2n
        Elist = fivepoint_solver(x1n, x2n)
        for E in Elist:
            F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
            x11 = np.vstack([x[0], np.ones((1, x[0].shape[1]))])
            x22 = np.vstack([x[1], np.ones((1, x[1].shape[1]))])
            l = F @ x11
            l = l / np.sqrt(l[0]**2 + l[1]**2)
            d = np.abs(np.sum(l * x22, axis=0))
            inliers = np.where(d < thresh)[0]
            if inliers.size > best_num:
                best_num = inliers.size
                best_inliers = inliers
                best_E = E

    print(f"Number of inliers: {best_num}")
    # Camera extraction
    U,S,Vt = np.linalg.svd(best_E)
    if np.linalg.det(U @ Vt) < 0: Vt = -Vt
    W = np.array([[0,-1,0],[1,0,0],[0,0,1]])
    R1 = U @ W  @ Vt
    R2 = U @ W.T @ Vt
    t  = U[:,2]
    cam_solutions = [np.hstack([R1,  t.reshape(3,1)]),
                     np.hstack([R1, -t.reshape(3,1)]),
                     np.hstack([R2,  t.reshape(3,1)]),
                     np.hstack([R2, -t.reshape(3,1)])]

    x1h = pextend(x[0])    # 3xN
    x2h = pextend(x[1])
    x1n_full = (np.linalg.inv(K) @ x1h)[:2].T
    x2n_full = (np.linalg.inv(K) @ x2h)[:2].T

    counts = []
    tris   = []
    for P2 in cam_solutions:
        P1 = np.hstack([np.eye(3), np.zeros((3,1))])
        X = triangulate(x1n_full, x2n_full, P1, P2)     # Nx3
        X2 = (P2[:,:3] @ X.T + P2[:,3:4]).T
        count = np.sum((X[:,2] > 0) & (X2[:,2] > 0))
        counts.append(count); tris.append(X)
    best_idx = int(np.argmax(counts))
    Xbest = tris[best_idx]

    # Reprojection & RMS plots
    P1o = K @ np.hstack([np.eye(3), np.zeros((3,1))])
    P2o = K @ cam_solutions[best_idx]
    Xh = np.hstack([Xbest, np.ones((Xbest.shape[0],1))]).T
    x2_proj = (P2o[:,:3] @ Xbest.T + P2o[:,3:4])
    x2_proj /= x2_proj[2]
    x1_proj = (P1o[:,:3] @ Xbest.T)
    x1_proj /= x1_proj[2]

    plt.figure(); plt.scatter(x[0][0], x[0][1], s=6, c='g'); plt.scatter(x1_proj[0], x1_proj[1], s=6, c='b')
    plt.title("Image 1: measured vs projected")
    plt.figure(); plt.scatter(x[1][0], x[1][1], s=6, c='r'); plt.scatter(x2_proj[0], x2_proj[1], s=6, c='b')
    plt.title("Image 2: measured vs projected")

    err1 = x[0][:,best_inliers] - x1_proj[:2, best_inliers]
    err2 = x[1][:,best_inliers] - x2_proj[:2, best_inliers]
    plt.figure(); plt.hist(np.abs(err1).sum(axis=0), bins=100); plt.title("Reproj error hist (img1)")
    plt.figure(); plt.hist(np.abs(err2).sum(axis=0), bins=100); plt.title("Reproj error hist (img2)")

    rms1 = np.sqrt(np.sum((x[0][:,best_inliers]-x1_proj[:2,best_inliers])**2)/x1_proj.shape[1])
    rms2 = np.sqrt(np.sum((x[1][:,best_inliers]-x2_proj[:2,best_inliers])**2)/x2_proj.shape[1])
    print(f"RMS error 1: {rms1:.4f} px")
    print(f"RMS error 2: {rms2:.4f} px")
    plt.show()

if __name__ == "__main__":
    run()
