from __future__ import annotations

from collections.abc import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from natsort import natsorted


def binarize(adata: ad.AnnData, verbose: bool = True) -> None:
    """
    Binarize the values in adata.X

    Args:
        adata: adata object
        verbose: provide written feedback

    Returns:
        Nothing, updates adata.layers and adata.X
    """
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    elif verbose:
        print("counts layer already set")

    if "binary" not in adata.layers:
        X = adata.layers["counts"].copy()
        X.data = np.ones_like(X.data, dtype=np.float32)  # set all nonzero entries to 1
        X.eliminate_zeros()  # just to be sure
        adata.layers["binary"] = X.copy()
    elif verbose:
        print("binary layer already set")

    adata.X = adata.layers["binary"].copy()
    if verbose:
        print("binary layer set as adata.X")


def knn_count_smoothing(
    adata: ad.AnnData,
    layer: str = "binary",
    layer_added: str = None,
    neighbors_key: str = "neighbors",
    verbose: bool = True,
) -> None:
    """
    For each cell, replace its gene vector with the average of its KNN neighborhood.

    Runs sc.pp.neighbors if it has not run.
    See https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pp.neighbors.html

    Args:
        adata: adata object
        layer: name of the adata layer to use for smoothing
        layer_added: key in adata.layers for function output (default: "KNN_binary_mean")
        neighbors_key: See sc.pp.neighbors for details
        verbose: provide written feedback

    Returns:
        Nothing, updates adata.layers and adata.X
    """
    if layer not in adata.layers:
        raise KeyError(f"{layer=} not found in adata.layers.")

    if layer_added is None:
        layer_added = f"KNN_{layer}_mean"

    if neighbors_key not in adata.uns:
        raise ValueError(
            f"{neighbors_key=} not found. "
            "Please run sc.pp.neighbors with the correct `use_rep`"
        )

    # KNN neighborhood connectivity map
    connectivities = f"{neighbors_key}_connectivities"
    knn = adata.obsp[connectivities].copy()
    knn.data = np.ones_like(knn.data)
    knn.setdiag(1)  # include self

    # number of neighbors per cell + itself
    deg = np.asarray(knn.sum(axis=1))

    # row-normalized connectivity map
    knn = knn.multiply(1 / deg)  # coo_matrix

    # average gene presence across its neighborhood
    X = adata.layers[layer]
    data = knn.dot(X)

    # sanity checks
    assert data.shape == X.shape
    assert data.dtype == np.float32
    assert isinstance(data, sp.csr_matrix)

    adata.layers[layer_added] = data

    adata.X = adata.layers[layer_added].copy()
    if verbose:
        print(f"{layer_added} layer set as adata.X")


def combine_obs_columns(
    adata: ad.AnnData, columns: list, column_name: str, delim: str = "_"
):
    """
    Create a new column in adata.obs by combining all columns with the delimiter.

    Args:
        adata: an adata object
        columns: a list of columns in adata.obs to combine
        column_name: the name for the new column
        delim: the delimiter to use while joining the columns

    Returns:
        Nothing, updates adata.obs
    """
    if not isinstance(columns, (list, set, tuple)):
        raise TypeError(f"columns must be a list, set or tuple, not {type(columns)}")
    if len(columns) < 2:
        raise ValueError(f"columns must contain 2 or more columns")
    adata.obs[column_name] = (
        adata.obs[columns].astype(str).apply(lambda x: delim.join(x), axis=1)
    )


def pseudobulk(
    adata: ad.AnnData, column: str, layer: str = "binary",
) -> pd.DataFrame:
    """
    Generate a pseudobulk table (genes x samples) for all samples in the sample_column
    and the cluster in the cluster_column, if specified.

    Args:
        adata: adata object
        column: column in adata.obs with groups to compare
        layer: name of the adata layer to aggregate

    Returns:
        a dataframe with summed layer values per sample
    """
    sample2counts = {}
    for sample in adata.obs[column].unique():
        X = adata[adata.obs[column] == sample].layers[layer]
        sample2counts[sample] = X.sum(axis=0).A1

    pseudobulk_df = pd.DataFrame(data=sample2counts, index=adata.var_names)
    # convert to integers if there are no decimal values
    if (pseudobulk_df == np.floor(pseudobulk_df)).any().any():
        pseudobulk_df = pseudobulk_df.astype(int)
    return pseudobulk_df


def detection_rates(
    adata: ad.AnnData, column: str, normalize: bool = True
) -> pd.DataFrame:
    """
    Calculate gene detection rates per group in the specified column of adata.obs.

    Args:
        adata: adata object
        column: column in adata.obs with groups to compare
        normalize: normalize detection rates for sample quality

    Returns:
        a dataframe with normalized gene detection rates
    """
    # gene detection rate per sample
    columns = []
    det_rate_cols = []
    for sample, ncells in adata.obs[column].value_counts().items():
        columns.append(sample)
        det_rates = (
            adata[adata.obs[column] == sample].layers["binary"].sum(axis=0).A / ncells
        )
        det_rate_cols.append(det_rates[0, :])
    det_rate_df = pd.DataFrame(det_rate_cols, index=columns, columns=adata.var_names).T

    # normalize detection rates for sample quality
    if normalize:
        dm = det_rate_df.values
        eps = 1e-9
        dm_clipped = np.clip(dm.astype(np.float64), eps, 1 - eps)
        logit_dm = np.log(dm_clipped / (1 - dm_clipped))

        zero_mask = dm == 0
        logit_dm_masked = logit_dm.copy()
        logit_dm_masked[zero_mask] = np.nan

        sample_medians = np.nanmedian(logit_dm_masked, axis=0)
        worst = sample_medians.min()
        shifts = sample_medians - worst

        logit_corrected = logit_dm.copy()
        for i, s in enumerate(shifts):
            col_mask = ~zero_mask[:, i]  # noqa
            logit_corrected[col_mask, i] -= s

        normalized = 1 / (1 + np.exp(-logit_corrected))
        normalized[zero_mask] = 0
        # shouldn't be necessary, but doesn't hurt to make sure
        normalized = np.clip(normalized, 0, 1)

        det_rate_df = pd.DataFrame(
            normalized.astype(np.float32),
            index=det_rate_df.index,
            columns=det_rate_df.columns,
        )
    return det_rate_df
