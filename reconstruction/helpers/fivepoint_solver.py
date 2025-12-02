# -*- coding: utf-8 -*-
import numpy as np

def rref(A, tol=1e-12):
    A = A.astype(float).copy()
    m,n = A.shape
    r = 0
    piv = []
    for c in range(n):
        pivrow = None
        for r2 in range(r, m):
            if abs(A[r2,c]) > tol:
                pivrow = r2; break
        if pivrow is None: continue
        if pivrow != r: A[[r,pivrow]] = A[[pivrow,r]]
        A[r] = A[r]/A[r,c]
        for r2 in range(m):
            if r2 != r:
                A[r2] -= A[r2,c]*A[r]
        piv.append((r,c))
        r += 1
        if r == m: break
    return A

def fivepoint_solver(x1n, x2n):
    """x1n, x2n: 3x5 homogeneous normalized."""
    M = np.zeros((5,9))
    for i in range(5):
        xx = np.outer(x1n[:,i], x2n[:,i]).reshape(-1)
        M[i,:] = xx
    # null space (9x4)
    U,S,Vt = np.linalg.svd(M)
    Evec = Vt[-4:].T
    E = [Evec[:,k].reshape(3,3).T for k in range(4)]

    coeffs = np.zeros((9,64))
    mons = np.zeros((4,64), int)
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                mons[:,idx] += 1
                mons[j,idx] += 0  # already counted
                mons[k,idx] += 0
                newC = 2*E[i]@E[j].T@E[k] - np.trace(E[i]@E[j].T)*E[k]
                coeffs[:,idx] = newC.reshape(-1)
                idx += 1
    det_coeffs = np.zeros((1,64))
    idx = 0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                det_coeffs[0,idx] = (
                    E[i][0,0]*E[j][1,1]*E[k][2,2] +
                    E[i][0,1]*E[j][1,2]*E[k][2,0] +
                    E[i][0,2]*E[j][1,0]*E[k][2,1] -
                    E[i][0,0]*E[j][1,2]*E[k][2,1] -
                    E[i][0,1]*E[j][1,0]*E[k][2,2] -
                    E[i][0,2]*E[j][1,1]*E[k][2,0]
                )
                idx += 1
    coeffs = np.vstack([coeffs, det_coeffs])
    # deduplicate monomials
    monsT = mons.T
    uniq, I, J = np.unique(monsT, axis=0, return_index=True, return_inverse=True)
    mons_small = mons[:, I]
    coeffs_small = np.zeros((10, uniq.shape[0]))
    for r in range(coeffs.shape[0]):
        coeffs_small[r,:] = np.bincount(J, weights=coeffs[r,:], minlength=uniq.shape[0])
    # set x1 = 1 => drop row 0
    mons_small = mons_small[1:,:]
    red = rref(coeffs_small)
    # build action matrix for x4
    Mx4 = np.zeros((10,10))
    mon_basis = mons_small[:,10:]
    for i in range(mon_basis.shape[1]):
        x4mon = mon_basis[:,i] + np.array([0,0,1])
        deg = x4mon.sum()
        if deg >= 3:
            # find row to reduce against columns 10:
            for row in range(10):
                lhs = mons_small[:,row]
                if np.all(lhs == x4mon):
                    Mx4[:,i] = -red[row,10:]
                    break
        else:
            # <=2 => it is in basis
            for j in range(mon_basis.shape[1]):
                if np.all(mon_basis[:,j] == x4mon):
                    Mx4[j,i] = 1.0
                    break
    w, V = np.linalg.eig(Mx4.T)
    V = V / V[-1]
    Esol = []
    for ii in range(10):
        if np.isreal(w[ii]):
            x2, x3, x4 = V[-2,ii].real, V[-3,ii].real, V[-4,ii].real
            Esol.append( E[0] + E[1]*x2 + E[2]*x3 + E[3]*x4 )
    return Esol

