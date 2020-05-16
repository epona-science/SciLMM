
import os

import numpy as np
import pandas as pd

from scilmm.Estimation.HE import HE
from scilmm.Matrices.SparseMatrixFunctions import load_sparse_csr

class Population:
    def __init__(
        self,
        entries_path,
        phenotype_path = None,
        covariate_path = None,
        ibd_path = None,
        one_hot_covariates = [],
        bool_covariates = [],
        drop_covariates = [],
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
        self.entry_map = pd.Series(self.entries.index, index=self.entries.values, name='id')
        self.phenotype = self._load_phenotype(phenotype_path)
        self.covariate_info = None
        self.one_hot_covariates = one_hot_covariates
        self.drop_covariates = drop_covariates
        self.bool_covariates = bool_covariates
        self.covariates = self._load_covariates(covariate_path)
        self.ibd = load_sparse_csr(ibd_path) if ibd_path else None
        self.he = None
        self.results = None

        self.informative_indices = self._informative_indices()

    def __repr__(self):
        return "Population()"

    def __str__(self):
        template = '\n'.join([
            "Population()",
            "Individuals: {:>10} ({} Informative)",
            "",
            "Covariates:",
            "  {}",
            "",
            "Estimates:",
            "  {}"
        ])

        return template.format(
            self.entries.shape[0],
            self.informative_indices.shape[0],
            self.print_covariates(),
            np.array2string(self.he.he_estimates)
        )

    def write_data(self, output_dir):
        def path(name):
            return os.path.join(output_dir, name)

        np.savez(path('informative_indices.npz'), self.informative_indices)

        if self.covariate_info is not None:
            self.covariate_info.to_csv(path('covariate_info.csv'))

        if self.results is not None:
            self.results.to_csv(path('results.csv'))

        if self.he is not None:
            if self.he.covariate_coef is not None:
                np.savez(path('cov_coef.npz'), self.he.covariate_coef)

            if self.he.he_estimates is not None:
                np.savez(path('he_estimates.npz'), self.he.he_estimates)

            if self.he.sampling_variance is not None:
                np.savez(path('sampling_variance.npz'), self.he.sampling_variance)

    def print_covariates(self):
        if self.covariate_info is None: return "<No Covariates>"
        return self.covariate_info.to_string()

    def estimate_heritability(self, test_set_size = 0.05):
        train_set, test_set = self._train_test_split(test_set_size)

        ibd, phenotype, covariates = self._prepare_dataset(self.informative_indices)

        self.he = HE(phenotype, covariates, [ibd], train_set)
        self.he.estimate(compute_stderr=False)

        if self.covariate_info is not None:
            self.covariate_info['coef'] = self.he.covariate_coef

        predictions = [
            self.informative_indices,
            self.he.normalized_phenotypes(),
            self.he.estimated_breeding_values()
        ]

        self.results = pd.DataFrame(predictions).T
        self.results.columns = ['idx', 'norm_phenotype', 'ebv']
        self.results = self.results.set_index('idx').join(self.entry_map, how='inner')
        self.results['train'] = True
        self.results.loc[self.informative_indices[test_set], 'train'] = False

    def _train_test_split(self, test_set_size = 0.05):
        """
        Splits the data into a training and test set based upon
        given proportion

        :param test_set_size: Proportion of individuals to use in test set

        :return train_set, test_set:
        """
        indices = np.indices(self.informative_indices.shape).flatten()

        keep = int(round((1 - test_set_size) * indices.shape[0]))

        np.random.shuffle(indices)
        return indices[:keep], indices[keep:]

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

        covariates = covariates.drop(self.drop_covariates, axis=1)

        covariates['intercept'] = 1
        self.bool_covariates.append('intercept')

        float_covariates = list(
            set(covariates.columns) - set(self.bool_covariates + self.one_hot_covariates)
        )

        covariates = pd.get_dummies(covariates, columns=self.one_hot_covariates, drop_first=True)

        self.covariate_info = pd.DataFrame(
            index=covariates.columns, columns=['mean', 'stddev', 'coef']
        )
        
        self.covariate_info['mean'] = covariates[float_covariates].mean()
        self.covariate_info['stddev'] = covariates[float_covariates].std()

        for col in float_covariates:
            covariates[col] = (covariates[col] - covariates[col].mean()) / covariates[col].std()

        covariates = covariates.astype(float)

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

    def _get_ibd_informative(self):
        """
        Returns individuals from IBD array that will be informative
        """
        if self.ibd is None: pd.Series()

        has_rel = np.asarray(self.ibd.sum(axis=1))[:, 0] > 1
        return np.flatnonzero(has_rel)

    def _informative_indices(self):
        informative = self._get_ibd_informative()
        
        if self.phenotype is not None:
            informative = self.phenotype.index.intersection(informative).to_numpy()
        
        if self.covariates is not None:
            informative = self.covariates.index.intersection(informative).to_numpy()

        return informative

    def _prepare_dataset(self, indices):
        """
        Transforms datasets for use in estimation

        :param indices: The indices to keep for the dataset. You should
        only remove individuals that cannot be used for training for testing.
        """
        ibd = None
        phenotype = None
        covariates = None

        if self.ibd is not None:
            ibd = self.ibd[indices][:, indices]

        if self.phenotype is not None:
            phenotype = self.phenotype.loc[indices].to_numpy().flatten()

        if self.covariates is not None:
            covariates = self.covariates.loc[indices].to_numpy()

        return ibd, phenotype, covariates
