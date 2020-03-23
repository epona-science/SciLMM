
import pdb

import numpy as np
from scipy.sparse import eye
from sklearn import linear_model


def regress_beta_out(y, cov):
    """

    :param y: a vector of size n (per individual)
    :param cov: nxc matrix - n number of individuals, c number of covariates
    :return: y, with the covariates regressed out
    """
    # regress all covariates out of y
    CTC = cov.T.dot(cov)
    cov_coef = np.linalg.solve(CTC, cov.T.dot(y))

    y = y - cov.dot(cov_coef)

    # standardize y
    return y / y.std()

def compute_sampling_variance(y, mat_list, he_est):
    # compute H - the covariance matrix of y
    H = mat_list[0] * he_est[0]
    for mat_i, sigma2_i in zip(mat_list[1:], he_est[1:]):
        H += mat_i * he_est[i]

    H += sparse.eye(y.shape[0], format="csr") * (1.0 - he_est.sum())

    # compute HE sampling variance
    V_q = np.empty((K, K))
    for i, mat_i in enumerate(mat_list):

        if sim_num is None:
            HAi_min_I = H.dot(mat_i) - H

        for j, mat_i in enumerate(mat_list[: i + 1]):
            if sim_num is None:
                if j == i:
                    HAj_min_I = HAi_min_I
                else:
                    HAj_min_I = H.dot(mat_j) - H
                V_q[i, j] = (
                    2 * (HAi_min_I.multiply(HAj_min_I)).sum()
                )  # / float(n-1)**4
            else:
                # simulate vectors
                sim_y = np.random.randn(n, sim_num)
                Aj_minI_y = mat_j.dot(sim_y) - sim_y
                H_Aj_minI_y = H.dot(Aj_minI_y)
                Ai_min_I_H_Aj_minI_y = mat_i.dot(H_Aj_minI_y) - H_Aj_minI_y
                H_Ai_min_I_H_Aj_minI_y = H.dot(Ai_min_I_H_Aj_minI_y)
                V_q[i, j] = 2 * np.mean(
                    np.einsum("ij,ij->j", sim_y, H_Ai_min_I_H_Aj_minI_y)
                )

            V_q[j, i] = V_q[i, j]

    var_he_est = np.linalg.solve(S, np.linalg.solve(S, V_q).T).T


def HE(
    y,
    covariates,
    mat_list,
    compute_stderr=False,
):
    # standardize y
    y = regress_beta_out(y, covariates)

    # construct S and q
    K = len(mat_list)
    n = y.shape[0]
    q = np.zeros(K)
    S = np.zeros((K, K))

    for i, mat_i in enumerate(mat_list):
        q[i] = y.dot(mat_i.dot(y)) - mat_i.diagonal().dot(y ** 2)

        for j, mat_j in enumerate(mat_list):
            if j > i: continue

            S[j, i] = S[i, j] = (
                mat_i.multiply(mat_j)
            ).sum() - mat_i.diagonal().dot(mat_j.diagonal())

    # compute HE
    he_est = np.linalg.solve(S, q)

    sampling_variance = ()
    if compute_stderr:
        sampling_variance = compute_sampling_variance(y, mat_list, he_est)

    return he_est, q, S, sampling_variance


if __name__ == "__main__":
    from Simulation.Pedigree import simulate_tree
    from Simulation.Phenotype import simulate_phenotype
    from Matrices.Numerator import simple_numerator
    from Matrices.Epistasis import pairwise_epistasis
    from Matrices.Dominance import dominance

    rel, _, _ = simulate_tree(50000, 0.001, 1.4, 0.9)
    ibd, _, _ = simple_numerator(rel)
    epis = pairwise_epistasis(ibd)
    dom = dominance(rel, ibd)
    cov = np.random.randn(50000, 2)
    y = simulate_phenotype(
        [ibd, epis, dom],
        cov,
        np.array([0.3, 0.2, 0.1, 0.4]),
        np.array([0.01, 0.02, 0.03]),
        True,
    )
    print(compute_HE(y, cov, [ibd, epis, dom], True))
