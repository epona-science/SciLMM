import argparse
import os

import numpy as np

from scilmm.Estimation.HE import compute_HE
from scilmm.Estimation.LMM import LMM, SparseCholesky
from scilmm.FileFormats.FAM import read_fam, write_fam
from scilmm.Matrices.Dominance import dominance
from scilmm.Matrices.Epistasis import pairwise_epistasis
from scilmm.Matrices.Numerator import simple_numerator
from scilmm.Matrices.Relationship import organize_rel
from scilmm.Matrices.SparseMatrixFunctions import (
    load_sparse_csr,
    save_sparse_csr,
)

def parse_arguments():
    parser = argparse.ArgumentParser(description="scilmm")

    parser.add_argument(
        "--fam",
        dest="fam_path",
        type=str,
        required=True,
        help=".fam file representing the pedigree. "
        + "the phenotype column contains all 0 if everyone is of interest, "
        + "or if only a subset is of interest their phenotype will contain 1",
    )

    parser.add_argument(
        "--pheno",
        dest="pheno_path",
        type=str,
        default=None,
        help="Path to phenotype file (space-delimited, no header, IID first column and phenotype second)",
    )

    parser.add_argument(
        "--IBD",
        dest="ibd_path",
        type=str,
        required=True,
        help="Path to IBD matrix to use",
    )

    parser.add_argument(
        "--entries",
        dest="entries_path",
        type=str,
        required=True,
        help="Path to entries list from IBD calculation",
    )

    parser.add_argument(
        "--Epistasis",
        dest="epistasis",
        action="store_true",
        default=False,
        help="Whether to calculate pairwise-epistasis matrix",
    )

    parser.add_argument(
        "--Dominance",
        dest="dominance",
        action="store_true",
        default=False,
        help="Whether to calculate dominance matrix",
    )

    parser.add_argument(
        "--covariates",
        dest="cov_path",
        type=str,
        default=None,
        help="the covaraites file (space-delimited file with header and IID as first column)",
    )

    parser.add_argument(
        "--HE",
        dest="he",
        action="store_true",
        default=False,
        help="Estimate fixed effects and covariance coefficients via Haseman-Elston",
    )

    parser.add_argument(
        "--LMM",
        dest="lmm",
        action="store_true",
        default=False,
        help="Estimate fixed effects and covariance coefficients via Linear mixed models",
    )

    parser.add_argument(
        "--REML",
        dest="reml",
        action="store_true",
        default=False,
        help="Use REML instead of simple maximum likelihood",
    )

    parser.add_argument(
        "--intercept",
        dest="intercept",
        action="store_true",
        default=False,
        help="Use an intercept as a covariate",
    )

    parser.add_argument(
        "--sim_num",
        dest="sim_num",
        type=int,
        default=100,
        help="Number of simulated vectors",
    )

    parser.add_argument(
        "--output_folder",
        dest="output_folder",
        type=str,
        default=".",
        help="Which folder it should save the output to.",
    )

    args = parser.parse_args()

    return args


def SciLMM(
    fam_path,
    pheno_path,
    ibd_path,
    entries_path,
    cov_path=None,
    output_folder=".",
    epistasis=False,
    dominance=False,
    he=False,
    lmm=False,
    reml=False,
    intercept=False,
    sim_num=100,
):
    # Check output destination
    if not os.path.exists(output_folder):
        raise Exception("The output folder does not exists")

    # Load required files
    ibd = load_sparse_csr(ibd_path)
    entries = np.load(entries_path)
    y = pd.read_csv(pheno_path, sep=' ', header=None, index_col=0)

    rel_org, sex, interest, _ = read_fam(fam_file_path=fam_path)
    rel, of_interest = organize_rel(rel_org, interest)

    if cov_path:
        cov = np.hstack((cov, np.load(cov_path)))
    else:
        cov = sex[:, np.newaxis]

    if epistasis:
        epis = pairwise_epistasis(ibd)
        save_sparse_csr(os.path.join(output_folder, "Epistasis.npz"), epis)

    if dominance:
        dom = dominance(rel, ibd)
        save_sparse_csr(os.path.join(output_folder, "Dominance.npz"), dom)

    covariance_matrices = list(filter(None, [ibd, epis, dom]))

    if he:
        print(compute_HE(y, cov, covariance_matrices, intercept))

    if lmm:
        print(
            LMM(
                SparseCholesky(),
                covariance_matrices,
                cov,
                y,
                with_intercept=intercept,
                reml=reml,
                sim_num=sim_num,
            )
        )


if __name__ == "__main__":
    args = parse_arguments()
    SciLMM(**args.__dict__)
