import numpy as np


def ranks(a: np.ndarray, sorted=True):
    if not sorted:
        a = np.sort(a)

    unique, unique_indices, unique_inverse = np.unique(a, return_index=True, return_inverse=True)

    return unique_indices[unique_inverse]


def sample(a: np.ndarray, a_ranks: np.ndarray, p: float):
    assert len(a) == len(a_ranks)

    ind = np.random.random_sample(a.size) <= p
    return a[ind], a_ranks[ind]
