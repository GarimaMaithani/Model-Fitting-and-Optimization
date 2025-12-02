# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from reconstruction.helpers.drawface import drawface

def run():
    D = loadmat("reconstruction/data/compEx4data.mat")
    A = D["A"]  # 2m x n
    k = 7
    m, n = A.shape
    c = np.random.randn(2, k, m//2)
    B = np.random.randn(1, n, k)
    X = np.zeros_like(A)
    for i in range(m//2):
        X[2*i:2*i+2, :] = (c[:,:,i] @ B.reshape(k, n)).reshape(2, n) + A[2*i:2*i+2, :1].repeat(n, axis=1)

    rel_err = np.linalg.norm(A - X, 'fro') / np.linalg.norm(A, 'fro')
    print(f"Relative error: {rel_err:.6f}")

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); drawface(A[0], A[1], 'b'); plt.title("Original Face")
    plt.subplot(1,2,2); drawface(X[0], X[1], 'r'); plt.title("Estimated Face")
    plt.tight_layout()

    # SVD baseline demo
    mean_rows = A.mean(axis=1, keepdims=True)
    U,S,Vt = np.linalg.svd(A - mean_rows @ np.ones((1, A.shape[1])), full_matrices=False)
    U_best = A @ np.linalg.pinv(Vt)
    _ = U_best  # just to mirror the MATLAB line; not used further

    plt.show()

if __name__ == "__main__":
    run()

