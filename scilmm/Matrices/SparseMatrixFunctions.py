import numpy as np
from scipy.sparse import save_npz, load_npz


def save_sparse_csr(filename, array):
    save_npz(filename, array.tocsr())


def load_sparse_csr(filename):
    return load_npz(filename).tocsr()
