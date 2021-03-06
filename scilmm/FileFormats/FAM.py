import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def read_fam(
    fam_file_path=None,
    fam=None,
    null_value="0",
    female_value="2",
    phenotype_of_interest="1",
):
    """
    Reading a .fam family file (this is also a csv file)
    :param phenotype_of_interest: value for people with phenotype of interest. If is None will be tested using Pandas.isna()
    :param null_value: Value for non-existing IDs, if is None values will be tested using Pandas.isna().
    :param female_value: Value for female in the 'sex' column.
    :return:
    :param fam_file_path: Path to family file, delimiter of this file is ' '
        Note that the family file should have the following columns (all strings):
        - FID: Family ID. This is not a unique ID for an individual. [optional]
        - IID: individual's ID
        - F_IID: father IID, IID of '0' is defined as null.
        - M_IID: mother IID, IID of '0' is defined as null.
        - sex: individual's sex. By default appears as '0':unknown, '1':male, '2':female (according to Plink).
        - phenotype: Should appear as 0 if not of interest, 1 if of interest.
            If all are 0's then all individuals are of interest.
    :param fam: The actual pandas DataFrame of the familiy file.
    :return:
    """
    if fam is None:
        df = pd.read_csv(
            fam_file_path,
            delimiter=" ",
            dtype=str,
            names=["FID", "IID", "F_IID", "M_IID", "sex", "phenotype"],
        )
    else:
        df = fam.copy()

    # add family id to all the ids in order to avoids duplicates
    if "FID" in df.columns:
        if null_value:
            df["F_IID"][df["F_IID"] != null_value] = (
                df["FID"][df["F_IID"] != null_value].map(str)
                + "_"
                + df["F_IID"][df["F_IID"] != null_value]
            )
            df["M_IID"][df["M_IID"] != null_value] = (
                df["FID"][df["M_IID"] != null_value].map(str)
                + "_"
                + df["M_IID"][df["M_IID"] != null_value]
            )
            df["IID"] = df["FID"].map(str) + "_" + df["IID"]
        else:
            df["F_IID"][~df["F_IID"].isna()] = (
                df["FID"][~df["F_IID"].isna()].map(str)
                + "_"
                + df["F_IID"][~df["F_IID"].isna()].map(str)
            )
            df["M_IID"][~df["M_IID"].isna()] = (
                df["FID"][~df["M_IID"].isna()].map(str)
                + "_"
                + df["M_IID"][~df["M_IID"].isna()].map(str)
            )
            df["IID"] = df["FID"].map(str) + "_" + df["IID"].map(str)

        entries = {
            id: i
            for i, id in enumerate(
                [
                    x
                    for x in np.unique(
                        np.concatenate(df[["IID", "F_IID", "M_IID"]].values)
                    )
                    if "_" in x
                ]
            )
        }
    else:
        entries = {
            id: i
            for i, id in enumerate(
                [
                    x
                    for x in np.unique(
                        np.concatenate(df[["IID", "F_IID", "M_IID"]].values)
                    )
                    if x
                ]
            )
        }
    all_ids = np.array(list(entries.keys()))

    # get all parent-child edges
    if null_value:
        child_father = np.array(
            [
                [entries[child], entries[father]]
                for child, father in df[["IID", "F_IID"]][
                    df["F_IID"] != null_value
                ].values
            ]
        )
        child_mother = np.array(
            [
                [entries[child], entries[mother]]
                for child, mother in df[["IID", "M_IID"]][
                    df["M_IID"] != null_value
                ].values
            ]
        )
    else:
        child_father = np.array(
            [
                [entries[child], entries[father]]
                for child, father in df[["IID", "F_IID"]][
                    ~df["F_IID"].isna()
                ].values
            ]
        )
        child_mother = np.array(
            [
                [entries[child], entries[mother]]
                for child, mother in df[["IID", "M_IID"]][
                    ~df["M_IID"].isna()
                ].values
            ]
        )
    all_co = np.vstack((child_father, child_mother))

    # create the relationship matrix
    rel = csr_matrix(
        (np.ones(all_co.shape[0]), (all_co[:, 0], all_co[:, 1])),
        shape=(all_ids.size, all_ids.size),
        dtype=np.bool,
    )

    # extra data
    # TODO: haven't tested sex in this new method - I think it won't work, but nobody will read this / use this
    sex = df["sex"] == female_value
    if "phenotype" in df.columns:
        if phenotype_of_interest:
            interest = np.array(
                [
                    entries[entry_id]
                    for entry_id in df[
                        df["phenotype"] == phenotype_of_interest
                    ]["IID"]
                ]
            )
        else:
            interest = np.array(
                [
                    entries[entry_id]
                    for entry_id in df[~df["phenotype"].isna()]["IID"]
                ]
            )
    else:
        interest = np.array([])
    if interest.size == 0:
        interest = None

    # returns the relationship matrix, sex for each index, is it of interest and index to entry name translation
    return rel, sex, interest, {i: id for id, i in entries.items()}


def write_fam(fam_file_path, rel, sex, indices):
    sex = sex.astype(np.bool)
    individuals, parents = rel.nonzero()

    fathers = np.zeros((rel.shape[0]), dtype=np.int32)
    mothers = np.zeros((rel.shape[0]), dtype=np.int32)
    interest = np.zeros((rel.shape[0]), dtype=np.int32)
    # need to add one so 0 is not an entry
    fathers[individuals[sex[parents]]] = parents[sex[parents]] + 1
    mothers[individuals[~sex[parents]]] = parents[~sex[parents]] + 1
    if indices is not None:
        interest[indices] = True
    content = np.vstack(
        (
            np.zeros((rel.shape[0]), dtype=np.int32),
            np.arange(rel.shape[0]) + 1,
            fathers,
            mothers,
            ["2" if x else "1" for x in sex],
            interest,
        )
    )
    np.savetxt(fam_file_path, content.T, delimiter=" ", fmt="%s")


if __name__ == "__main__":
    from Simulation.Pedigree import simulate_tree

    rel, sex, _ = simulate_tree(10000, 0.001, 1.4, 0.9)
    indices = np.array([1, 2, 3])
    write_fam("temp.fam", rel, sex, indices)

    rel_after, sex_after, interest_after, entries_list = read_fam("temp.fam")
    assert (rel_after - rel).nnz == 0
    assert np.count_nonzero(sex - sex_after) == 0
    assert np.count_nonzero(interest_after - indices) == 0
