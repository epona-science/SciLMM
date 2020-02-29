#!/usr/bin/env python

import os
import pdb

import click
import numpy as np
import pandas as pd
import scipy
from scilmm.Matrices.SparseMatrixFunctions import load_sparse_csr
from scilmm.SparseCholesky import run_estimates


@click.command()
@click.argument("ibd_path", type=click.Path(exists=True))
@click.argument("entries_path", type=click.Path(exists=True))
@click.argument("pheno_path", type=click.Path(exists=True))
@click.argument("cov_path", type=click.Path(exists=True))
@click.argument("output", type=click.Path(exists=True))
def fitModel(ibd_path, entries_path, pheno_path, cov_path, output):
    """ Fits the GLM

    Parameters
    ---------
        ibd_path : str
            Path to IBD file

        entries_path : str
            Path to entries file

        pheno_path : str
            Path phenotype file with IID column and phenotype column
            without headers

        cov_path : str
            Path to covariate file with headers. First column should be IID
            value with one column per covariate following

        output : str
            Output directory to write results to
    """
    cov_dtypes = {
        "iid": int,
        "dob_offset": int,
        "family": int,
        "dam_age": int,
        "sire_age": int,
        "surface": str,
        "distance": float,
        "yob": int
    }

    ibd = load_sparse_csr(ibd_path)
    pheno = pd.read_csv(
        pheno_path,
        sep=' ',
        header=None,
        index_col=0,
        low_memory=False,
        dtype=int
    )
    cov = pd.read_csv(cov_path,
        sep=' ',
        index_col=0,
        low_memory=False,
        dtype=cov_dtypes
    )

    entries = pd.Series(np.load(entries_path))
    entries = entries.str.split('_').str[1].astype(int)
    entries = pd.Series(entries.index.values, index=entries)

    pheno.set_index(entries[entries.index.isin(pheno.index)], inplace=True)
    cov.set_index(entries[entries.index.isin(cov.index)], inplace=True)

    # Drop non-numeric dtypes for now
    cov = cov.select_dtypes(include=['number'])

    he_est = run_estimates(ibd, pheno, cov)
    print(he_est)
    np.savez(
        os.path.join(output, 'he.npz'),
        he=he_est[0],
        he_var=he_est[1],
        fixed=he_est[2],
        var=he_est[3],
        covar=he_est[4],
        var_he_est=he_est[5],
        vq=he_est[6]
    )

if __name__ == "__main__":
    fitModel()
