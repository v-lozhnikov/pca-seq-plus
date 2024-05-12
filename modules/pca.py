import numpy as np


def double_centering(mat: np.array):
    n = mat.shape[0]
    centered = np.copy(mat)
    for i in range(n):
        centered[i] -= np.mean(mat[i])
    for j in range(n):
        centered[:, j] -= np.mean(mat[:, j])
    centered += np.mean(mat)
    return centered


def pca(mat: np.array):
    xxT = double_centering(mat ** 2) * -0.5
    p, lam, pT = np.linalg.svd(xxT)
    u = p * (lam ** 0.5)
    return lam, u
