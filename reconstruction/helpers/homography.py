# -*- coding: utf-8 -*-
import numpy as np
import cv2

def estimate_homography_dlt(ptsA, ptsB):
    """ptsA, ptsB: Nx2 arrays; returns 3x3 H (A->B)"""
    n = ptsA.shape[0]
    A = []
    for i in range(n):
        x,y = ptsA[i]; u,v = ptsB[i]
        A.append([-x,-y,-1, 0,0,0, x*u, y*u, u])
        A.append([ 0, 0, 0,-x,-y,-1, x*v, y*v, v])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3,3)
    return H / H[-1,-1]

def ransac_homography(xA, xB, num_iter=1000, thresh=5.0):
    """xA,xB: 2xN arrays. Returns best H and inlier mask."""
    xA_t = xA.T; xB_t = xB.T
    N = xA_t.shape[0]
    best_in = 0; best_H = None; best_mask = None
    for _ in range(num_iter):
        idx = np.random.choice(N, 4, replace=False)
        H = estimate_homography_dlt(xA_t[idx], xB_t[idx])
        pts = np.vstack([xA, np.ones((1, N))])
        proj = H @ pts
        proj /= proj[2]
        d = np.sqrt(np.sum((proj[:2] - xB)**2, axis=0))
        inliers = d < thresh
        count = inliers.sum()
        if count > best_in:
            best_in = count; best_H = H; best_mask = inliers
    return best_H, best_mask

def warp_and_stitch(A, B, H):
    """Warp A into B's frame and stitch."""
    hA,wA = A.shape[:2]; hB,wB = B.shape[:2]
    cornersA = np.array([[0,0],[wA,0],[wA,hA],[0,hA]], dtype=float)
    cornersB = np.array([[0,0],[wB,0],[wB,hB],[0,hB]], dtype=float)
    ones = np.ones((4,1))
    cA = np.hstack([cornersA, ones])
    cAt = (H @ cA.T).T; cAt = cAt[:, :2] / cAt[:, 2:3]
    all_pts = np.vstack([cAt, cornersB])
    min_xy = np.floor(all_pts.min(axis=0)).astype(int)
    max_xy = np.ceil(all_pts.max(axis=0)).astype(int)
    tx, ty = -min_xy[0], -min_xy[1]
    T = np.array([[1,0,tx],[0,1,ty],[0,0,1]], float)
    size = (max_xy[0]-min_xy[0], max_xy[1]-min_xy[1])
    A_warp = cv2.warpPerspective(A, T@H, size)
    B_warp = cv2.warpPerspective(B, T@np.eye(3), size)
    mask = (A_warp>0)
    out = B_warp.copy()
    out[mask] = A_warp[mask]
    return out

