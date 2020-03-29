
import logging
import pdb

import numpy as np
from scipy.sparse import eye

class HE:
    def __init__(
        self,
        phenotype,
        covariates,
        covariance_matrices,
        train_indices = None
    ):
        self.logger = logging.getLogger('scilmm.Estimation.HE')
        self.logger.info('creating instance of HE')

        self.__phenotype = phenotype
        self.__covariates = covariates
        self.__covariance_matrices = covariance_matrices
        self.__train_indices = train_indices

        self.__set_training_data()

        self.covariate_coef = None
        self.he_estimates = None
        self.phenotype_std = None
        self.q = None
        self.S = None
        self.sampling_variance = None

    @property
    def phenotype(self):
        return self.__phenotype_t

    @property
    def covariates(self):
        return self.__covariates_t

    @property
    def covariance_matrices(self):
        return self.__covariance_matrices_t

    @property
    def train_indices(self):
        return self.__train_indices

    @train_indices.setter
    def train_indices(self, new_indices):
        self.__train_indices = new_indices
        self.__set_training_data()

        self.covariate_coef = None
        self.he_estimates = None
        self.q = None
        self.S = None
        self.sampling_variance = None

    def regress_beta_out(self):
        """

        :return: phenotype, with the covariates regressed out
        """
        self.logger.debug("regressing covariates from phenotype")

        # regress all covariates out of y
        CTC = self.covariates.T.dot(self.covariates)
        self.covariate_coef = np.linalg.solve(
            CTC, self.covariates.T.dot(self.phenotype)
        )

        y = self.phenotype - self.covariates.dot(self.covariate_coef)
        self.phenotype_std = y.std()

        # standardize y
        return y / self.phenotype_std

    def compute_sampling_variance(self, y, he_est):
        self.logger.info("starting sampling variance calculation")

        # compute H - the covariance matrix of y
        H = self.covariance_matrices[0] * he_est[0]
        for mat_i, sigma2_i in zip(self.covariance_matrices[1:], he_est[1:]):
            H += mat_i * he_est[i]

        H += sparse.eye(y.shape[0], format="csr") * (1.0 - he_est.sum())

        # compute HE sampling variance
        V_q = np.empty((K, K))
        for i, mat_i in enumerate(self.covariance_matrices):

            if sim_num is None:
                HAi_min_I = H.dot(mat_i) - H

            for j, mat_i in enumerate(self.covariance_matrices[: i + 1]):
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

    def estimate(self, compute_stderr=False):
        self.logger.debug("starting HE estimation")

        # standardize y
        y = self.regress_beta_out()

        # construct S and q
        K = len(self.covariance_matrices)
        n = y.shape[0]
        q = np.zeros(K)
        S = np.zeros((K, K))

        self.logger.info("calculating using %d individuals and %d matrices", n, K)

        for i, mat_i in enumerate(self.covariance_matrices):
            q[i] = y.dot(mat_i.dot(y)) - mat_i.diagonal().dot(y ** 2)
            self.logger.debug("q value for matrix %d is %.4f", i, q[i])

            for j, mat_j in enumerate(self.covariance_matrices):
                if j > i: continue

                S[j, i] = S[i, j] = (
                    mat_i.multiply(mat_j)
                ).sum() - mat_i.diagonal().dot(mat_j.diagonal())

                self.logger.debug("covariance for matrices %d and %d is %.4f", i, j, S[i, j])

        # compute HE
        he_est = np.linalg.solve(S, q)

        self.logger.info("HE estimates are %s", he_est)

        sampling_variance = ()
        if compute_stderr:
            sampling_variance = compute_sampling_variance(y, he_est)

        self.he_estimates = he_est
        self.q = q
        self.S = S

    def __set_training_data(self):
        """Subsets data for trainging
        """
        if self.__train_indices is None:
            self.__phenotype_t = self.__phenotype
            self.__covariates_t = self.__covariates
            self.__covariance_matrices_t = self.__covariance_matrices
            return

        self.__phenotype_t = self.__phenotype[self.__train_indices]
        self.__covariates_t = self.__covariates[self.__train_indices]
        self.__covariance_matrices_t = [
            cov[self.__train_indices][:, self.__train_indices]
            for cov in self.__covariance_matrices
        ]

    def normalized_phenotypes(self):
        """Calculates the normalized phenotypes on the original
        given set of data

        :return array of results:
        """
        y = self.__phenotype - self.__covariates.dot(self.covariate_coef)
        return y / self.phenotype_std

    def estimated_breeding_values(self):
        """Calculates estimated breeding values for original dataset

        :return array of EBV:
        """
        return self.normalized_phenotypes() * np.sum(self.he_estimates)
