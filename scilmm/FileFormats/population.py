
import pdb

import numpy as np
import pandas as pd

from scilmm.Matrices.SparseMatrixFunctions import load_sparse_csr

class Population:
    def __init__(
        self,
        entries_path,
        phenotype_path = None,
        covariate_path = None,
        ibd_path = None
    ):
        """
        Creates a Population class instance.

        :param pedigree: A Pedigree object
        :param pheotypes: Series of phenotypes indexed by IID
        :param covariates: Dataframe of covariates indexed by IID
        :param covariances: List of covariance matrices aligned to Pedigree object
            entries list
        """
        self.entries = self._load_entries(entries_path)
        self.informative_indices = self.entries.to_numpy()
        self.phenotype = self._load_phenotype(phenotype_path)
        self.covariates = self._load_covariates(covariate_path)
        self.ibd = load_sparse_csr(ibd_path) if ibd_path else None

        self._prune_uninformative()

    def _load_entries(self, entries_path):
        """
        Loads in a entries file

        :param entries_path: Path to entries file
        """
        entries = pd.Series(np.load(entries_path))
        entries = entries.str.split('_').str[1].astype(int)
        return pd.Series(entries.index.values, index=entries, name='idx')

    def _load_phenotype(self, phenotype_path):
        """
        Loads in a phenotype file

        File must be space delimited without headers and first column must be
        IID

        :param phenotype_path: Path to phenotype file
        """
        if phenotype_path is None:
            return None

        phenotype = pd.read_csv(
            phenotype_path,
            sep=' ',
            header=None,
            index_col=0,
            low_memory=False
        )

        return phenotype.join(self.entries, how='inner').set_index('idx')

    def _load_covariates(self, covariate_path):
        """
        Loads in a covariate file and standardizes the attributes

        File must be space delimited with headers and first column must be
        IID

        :param covariate_path: Path to covariate file
        """
        if covariate_path is None:
            return None

        covariates = pd.read_csv(
            covariate_path,
            sep=' ',
            index_col=0,
            low_memory=False
        )

        covariates = covariates.select_dtypes(include=['number']).astype(float)
        covariates = (covariates - covariates.mean()) / covariates.std()
        covariates['intercept'] = 1

        return covariates.join(self.entries, how='inner').set_index('idx')

    def _load_ibd(self, ibd_path):
        """
        Loads in IBD matrix

        :param ibd_path: Path to IBD file
        """
        if ibd_path is None:
            return None

        ibd = load_sparse_csr(ibd_path)
        return ibd[self.entries.to_numpy()][:, self.entries.to_numpy()]

    def _prune_uninformative(self):
        """
        Prunes uninformative individuals
        """
        indices = set(self.entries.to_numpy())

        if self.ibd is not None:
            indices = indices & set(self.ibd.indices)

        if self.phenotype is not None:
            indices = indices & set(self.phenotype.index.to_numpy())

        if self.covariates is not None:
            indices = indices & set(self.covariates.index.to_numpy())

        indices = list(indices)

        if self.ibd is not None:
            self.ibd = self.ibd[indices][:,indices]
            has_rel = np.asarray(self.ibd.sum(axis=1))[:, 0] > 1
            self.ibd = self.ibd[has_rel][:, has_rel]
            indices = np.asarray(indices)[has_rel]

        if self.phenotype is not None:
            self.phenotype = self.phenotype.loc[indices].to_numpy().flatten()

        if self.covariates is not None:
            self.covariates = self.covariates.loc[indices].to_numpy()

        self.informative_indices = indices
