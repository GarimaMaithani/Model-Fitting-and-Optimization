# -*- coding: utf-8 -*-
import numpy as np

def rq(A):
    Q, R = np.linalg.qr(np.flipud(A).T)
    R = np.flipud(R.T)
    Q = Q.T[:, ::-1]
    return R, Q

