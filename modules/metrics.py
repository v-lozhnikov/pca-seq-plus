import numpy as np
import similaritymeasures
from Levenshtein import distance as lev
from scipy.spatial import distance


def pythagoras(seq_coords: np.array) -> np.array:
    return distance.squareform(distance.pdist(seq_coords))


def cavalli_sforza(seq_coords: np.array) -> np.array:
    (seq_count, coord_count) = seq_coords.shape
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            mat[i][j] = np.sum((seq_coords[i] ** 0.5 - seq_coords[j] ** 0.5) ** 2)
    return mat ** 0.5


def jaccard(seq_coords: np.array) -> np.array:
    (seq_count, coord_count) = seq_coords.shape
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            min_sum = np.sum(np.amin([seq_coords[i, :], seq_coords[j, :]], axis=0))
            max_sum = np.sum(np.amax([seq_coords[i, :], seq_coords[j, :]], axis=0))
            rho = min_sum / max_sum
            mat[i][j] = 1 - rho
    return mat ** 0.5


def jaccard_with_sqrt(seq_coords: np.array) -> np.array:
    (seq_count, coord_count) = seq_coords.shape
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            min_sum = np.sum(np.amin([seq_coords[i, :] ** 0.5, seq_coords[j, :] ** 0.5], axis=0))
            max_sum = np.sum(np.amax([seq_coords[i, :] ** 0.5, seq_coords[j, :] ** 0.5], axis=0))
            rho = min_sum / max_sum
            mat[i][j] = 1 - rho
    return mat ** 0.5


def entropy(seq_coords: np.array) -> np.array:
    (seq_count, coord_count) = seq_coords.shape
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            inf_i = (seq_coords[i] * np.log(seq_coords[i], where=seq_coords[i] > 0))
            inf_j = (seq_coords[j] * np.log(seq_coords[j], where=seq_coords[j] > 0))
            inf_i = np.nan_to_num(inf_i)
            inf_j = np.nan_to_num(inf_j)
            mat[i][j] = np.sum((inf_i - inf_j) ** 2)
    return mat ** 0.5


def arccosine(seq_coords: np.array) -> np.array:
    (seq_count, coord_count) = seq_coords.shape
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            mat[i][j] = np.abs(np.arccos(np.sum(seq_coords[i] ** 0.5 * seq_coords[j] ** 0.5)))
    mat = np.nan_to_num(mat)
    return mat


def hamming(seqs: np.array) -> np.array:
    seq_count = seqs.shape[0]
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            mat[i][j] = np.sqrt(distance.hamming([*seqs[i]], [*seqs[j]]) / len(seqs[i]))
    return mat


def levenshtein(seqs: np.array) -> np.array:
    seq_count = seqs.shape[0]
    mat = np.zeros((seq_count, seq_count))
    for i in range(seq_count):
        for j in range(seq_count):
            mat[i][j] = lev(seqs[i], seqs[j])
    return mat


def hausdorff(tr: np.array) -> np.array:
    tr_count = tr.shape[0]
    # print(tr_count)
    mat = np.zeros((tr_count, tr_count))
    for i in range(tr_count):
        for j in range(i, tr_count):
            max_min_dist = 0
            for p1 in range(tr[i].shape[0]):
                min_dist = np.inf
                for p2 in range(tr[j].shape[0]):
                    cur_dist = np.sum((tr[i][p1] - tr[j][p2]) ** 2)
                    # print(tr[i][p1], tr[j][p2], cur_dist)
                    min_dist = min(cur_dist, min_dist)
                max_min_dist = max(min_dist, max_min_dist)
            mat[i][j] = max_min_dist
            mat[j][i] = max_min_dist
    return mat


def frechet(tr: np.array) -> np.array:
    tr_count = tr.shape[0]
    mat = np.zeros((tr_count, tr_count))
    for i in range(tr_count):
        for j in range(i, tr_count):
            res = similaritymeasures.frechet_dist(tr[i], tr[j])
            mat[i, j] = res
            mat[j, i] = res
    return mat


def mean(tr: np.array) -> np.array:
    tr_count = tr.shape[0]
    mat = np.zeros((tr_count, tr_count))
    for i in range(tr_count):
        for j in range(tr_count):
            mat[i][j] = np.sum((np.mean(tr[i], axis=0) - np.mean(tr[j], axis=0)) ** 2)
    return mat ** 0.5


def jaccard_tr(tr: np.array) -> np.array:
    tr_count = tr.shape[0]
    mat = np.zeros((tr_count, tr_count))
    for i in range(tr_count):
        for j in range(i, tr_count):
            if i == j:
                res = 0
            else:
                res = cnt_jaccard_tr(tr[i], tr[j])
            mat[i][j] = res
            mat[j][i] = res
    return mat

def cnt_jaccard_tr(tr1, tr2):
    len1 = tr1.shape[0]
    len2 = tr2.shape[0]
    intersection = 0.0
    union = 0.0
    for i in range(len1):
        min_dist = 1e9
        is_other_tr = False
        for j in range(len1):
            if i == j:
                continue
            dist = np.sum((tr1[i] - tr1[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
        for j in range(len2):
            dist = np.sum((tr1[i] - tr2[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
                is_other_tr = True
        union += min_dist
        if is_other_tr:
            intersection += min_dist

    for i in range(len2):
        min_dist = 1e9
        is_other_tr = False
        for j in range(len2):
            if i == j:
                continue
            dist = np.sum((tr2[i] - tr2[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
        for j in range(len1):
            dist = np.sum((tr2[i] - tr1[j]) ** 2)
            if dist < min_dist:
                min_dist = dist
                is_other_tr = True
        union += min_dist
        if is_other_tr:
            intersection += min_dist

    return np.sqrt(1 - (intersection / union))
