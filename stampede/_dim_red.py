from __future__ import annotations

from collections.abc import Iterable

import anndata as ad
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.typing import ColorType
from sklearn.decomposition import TruncatedSVD

from ._qc import cell_qc_postfilter, gene_qc_postfilter


def dim_red(
    adata: ad.AnnData,
    n_dims: int = 50,
    use_genes: str = None,
    key_added: str = "X_svd",
    random_state: int = 42,
) -> None:
    """
    Dimensionality reduction using Term Frequency Latent Semantic Indexing.

    Args:
        adata: adata object
        n_dims: number of dimensions to produce
        use_genes: Boolean column in adata.var with True for genes to be used in dimensionality reduction.
        key_added: key in adata.obsm for function output
        random_state: random seed value

    Returns:
        Nothing, updates adata.obsm and adata.uns
    """
    if use_genes is None:
        adata_sub = adata
    else:
        if not use_genes in adata.var.columns:
            raise KeyError(
                f"{use_genes} not found in adata.var.columns. "
                "Use an existing column name for use_genes"
            )
        if not pd.api.types.is_bool_dtype(adata.var[use_genes]):
            raise ValueError(
                f"adata.var[{use_genes}] is not boolean. "
                "Use a column with bools to subset genes taken "
                "along for dimensionality reduction"
            )

        adata_sub = adata[:, adata.var[use_genes]].copy()
        cell_qc_postfilter(adata_sub)
        gene_qc_postfilter(adata_sub)

    prefix, uns_key = key_added.split("_", 1)
    if prefix != "X" or len(uns_key) == 0:
        raise ValueError(f"{key_added=} must start with 'X_', e.g. 'X_svd'")

    layer = "binary"
    if layer not in adata_sub.layers:
        raise KeyError(
            f"{layer=} not found in adata.layers. Please run st.pp.binarize first!"
        )
    X = adata_sub.layers[layer]
    if "nFeature_RNA_postfilter" not in adata_sub.obs.columns:
        raise KeyError(
            "nFeature_RNA_postfilter not in adata.obs. "
            "Please run st.pp.cell_qc_postfilter first!"
        )
    cell_sums = adata_sub.obs["nFeature_RNA_postfilter"].to_numpy(dtype=np.float32)
    if "nCell_postfilter" not in adata_sub.var.columns:
        raise KeyError(
            "nCell_postfilter not in adata.var. "
            "Please run st.pp.gene_qc_postfilter first!"
        )
    gene_counts = adata_sub.var["nCell_postfilter"].to_numpy(dtype=np.float32)
    n_cells = X.shape[0]

    # Latent Semantic Indexing:
    # Term Frequency:
    tf = X.multiply(1.0 / cell_sums[:, None])

    # Inverse Document Frequency
    idf = np.log1p(n_cells / (1.0 + gene_counts))
    X_tfidf = tf.multiply(idf)

    # Truncated Singular Value Decomposition
    svd = TruncatedSVD(n_components=n_dims + 1, random_state=random_state)

    # Drop 1st dimension, is not informative (like scATAC; this case expl var 200-fold lower than 2nd dim)
    adata.obsm[key_added] = svd.fit_transform(X_tfidf)[:, 1:]
    adata.uns[uns_key] = {
        "idf": idf,
        "explained_variance_ratio": svd.explained_variance_ratio_[1:],
    }


def plot_scree(adata: ad.AnnData, obsm_key: str = None) -> tuple[Figure, Axes]:
    """
    Scree plot

    Args:
        adata: adata object
        obsm_key: key in adata.obsm with dim_red output (default: "X_svd")

    Returns:
        matplotlib figure and array of axes
    """
    if obsm_key is None:
        obsm_key = "X_svd"
    uns_key = obsm_key.split("_", 1)[1]

    evr = adata.uns[uns_key]["explained_variance_ratio"]
    xs = [i + 1 for i in range(len(evr))]

    fig, ax = plt.subplots(1, 1, figsize=(6, 3))
    ax.scatter(x=xs, y=evr, s=5, c="black")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Explained variance ratio")
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Explained variance ratio")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlim(0, ax.get_xlim()[1])
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.tick_params(which="minor", length=2)

    return fig, ax


def plot_dim_red(
    adata: ad.AnnData,
    columns: str | Iterable,
    obsm_key: str = "X_svd",
    cmap: ColorType = "tab10",
    n_dims: int | tuple = 6,
    subset_size: int = 2_000,
    random_state: int = 42,
) -> list[tuple[Figure, Axes]]:
    """
    Grid plot visualizing a range of reduced dimensions.

    Args:
        adata: adata object
        columns: one or more columns in adata.obs to plot. One multiplot per column
        obsm_key: key in adata.obsm with dim_red output
        cmap: colormap
        n_dims: number of dimensions to plot, or a tuple with dimensions
        subset_size: subsample the data to this number (per column)
        random_state: random seed value

    Returns:
        a list of tuples with matplotlib figure and axis
    """
    cmap = plt.get_cmap(cmap)
    if isinstance(columns, str):
        columns = [columns]

    data = adata.obs[columns].copy()
    if isinstance(n_dims, int):
        n_dims = (1, n_dims)
    dims = list(range(n_dims[0], n_dims[1] + 1))
    data[[str(dim) for dim in dims]] = adata.obsm[obsm_key][:, dims]
    n_plots = n_dims[1] - n_dims[0]

    uns_key = obsm_key.split("_", 1)[1]
    evr_array = adata.uns[uns_key]["explained_variance_ratio"]

    ret = []
    for c in columns:
        df = data
        if subset_size and len(data[c]) > subset_size * len(data[c].unique()):
            try:
                df = data.groupby(c, observed=False).sample(
                    n=subset_size, random_state=random_state
                )
            except ValueError as e:
                size_error = (
                    "Cannot take a larger sample than population when 'replace=False'"
                )
                if str(e) == size_error:
                    raise ValueError(
                        f"{size_error}. Set '{subset_size=}' to False or a lower value"
                    )
                raise e

        levels, categories = pd.factorize(df[c])
        colors = [cmap.colors[i] for i in levels]  # noqa

        # plot the upper triangle of a cross correlation matrix of PCs
        # excluding the diagonal
        fig, axs = plt.subplots(
            nrows=n_plots,
            ncols=n_plots,
            figsize=(n_plots * 3, n_plots * 3),
            # gridspec_kw={'wspace': 0, 'hspace': 0},
            tight_layout=True,
        )
        fig.suptitle(c, fontweight="bold")
        n = fig._suptitle.get_fontsize()  # noqa
        fig._suptitle.set_fontsize(n + 10)  # noqa
        for row in range(0, n_plots):
            row_label = str(dims[row] + 0)
            for col in range(0, n_plots):
                col_label = str(dims[col] + 1)
                ax = axs[row, col]

                if row <= col:
                    # add plots only in the upper triangle
                    ax.scatter(
                        data=df,
                        x=col_label,
                        y=row_label,
                        c=colors,
                        alpha=0.1,
                        s=1,
                    )
                    ax.spines[["top", "right"]].set_visible(False)
                    ax.set_xticks([])
                    ax.set_yticks([])

                    # add labels and edges on the straight sides of the plot
                    if row == 0:
                        evr = np.format_float_scientific(
                            evr_array[int(col_label)],
                            precision=2,
                        )
                        if col == 0:
                            # explain the label
                            ax.set_xlabel(f"Dim {col_label} (explained var: {evr})")
                        else:
                            ax.set_xlabel(f"Dim {col_label} ({evr})")
                        ax.xaxis.set_label_position("top")
                        ax.spines["top"].set_visible(True)
                    if col == n_plots - 1:
                        evr = np.format_float_scientific(
                            evr_array[int(row_label)],
                            precision=2,
                        )
                        ax.set_ylabel(f"Dim {row_label} ({evr})")
                        ax.yaxis.set_label_position("right")
                        ax.spines["right"].set_visible(True)
                else:
                    # hide the lower triangle
                    ax.axis("off")

        # legend
        ax = axs[1, 0]
        handles = []
        for i, cat in enumerate(categories):
            patch = mpatches.Patch(
                color=cmap.colors[i],  # noqa
                label=cat,
            )
            handles.append(patch)
        ax.legend(handles=handles, loc="center", title=c)
        ax.axis("off")

        ret.append((fig, ax))

    return ret
