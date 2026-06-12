from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from sklearn.preprocessing import MinMaxScaler

if TYPE_CHECKING:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.inference import Inference


def pydeseq2(
    adata: ad.AnnData,
    design: str,
    contrast: list,
    layer: str = "binary",
    inference: Inference = None,
    n_cpus: int = 16,
    return_objects: bool = False,
    dds_kwargs: dict = None,
    ds_kwargs: dict = None,
) -> tuple[DeseqDataSet, DeseqStats, pd.DataFrame] | pd.DataFrame:
    """
    Wrapper around pyDEseq2 for adata objects.

    See https://pydeseq2.readthedocs.io/en/latest/auto_examples/plot_minimal_pydeseq2_pipeline.html

    Args:
        adata: adata object
        design: a formula in the format 'x + z' or '~x+z'.
         Each factor must be a column in adata.obs
        contrast:  a list of three strings in the following format:
         ['variable_of_interest', 'tested_level', 'ref_level']
        layer: name of the adata layer where values are drawn from
        inference: pyDESeq2 inference class instance
        n_cpus: number of threads to use
        return_objects: return the DeseqDataSet, DeseqStats and the results_df.
         If False, only return the results_df
        dds_kwargs: kwargs passed to DeseqDataSet
        ds_kwargs: kwargs passed to DeseqStats

    Returns:
        pydeseq2 output
    """
    # optional dependency
    from pydeseq2.dds import DeseqDataSet  # noqa
    from pydeseq2.default_inference import DefaultInference  # noqa
    from pydeseq2.ds import DeseqStats  # noqa

    if inference is None:
        inference = DefaultInference(n_cpus=n_cpus)
    if dds_kwargs is None:
        dds_kwargs = {}
    if ds_kwargs is None:
        ds_kwargs = {}

    # pyDESeq2 does not work with sparse matrices
    counts = adata.layers[layer]
    if isinstance(counts, pd.DataFrame):
        # dataframes
        if len(counts.index) == len(adata.var.index) and len(counts.columns) == len(
            adata.obs.index
        ):
            # must have cells for rows and genes for columns
            counts = counts.T
    elif "scipy.sparse." in str(type(counts)):
        # any kind of sparse matrics
        counts = pd.DataFrame.sparse.from_spmatrix(
            data=counts,
            index=adata.obs.index,
            columns=adata.var.index,
        )
    else:
        # numpy arrays/matrices
        counts = pd.DataFrame(
            data=counts,
            index=adata.obs.index,
            columns=adata.var.index,
        )

    # cannot capture the warnings due to multiprocessing
    with warnings.catch_warnings():
        # invalid value encountered in slogdet
        warnings.simplefilter("ignore", category=RuntimeWarning)

        dds = DeseqDataSet(
            counts=counts,
            metadata=adata.obs,
            design=design,
            inference=inference,
            **dds_kwargs,
        )
        dds.deseq2()

        ds = DeseqStats(
            dds,
            contrast=contrast,
            n_cpus=n_cpus,
            **ds_kwargs,
        )
        ds.summary()

        df = ds.results_df

    if return_objects:
        return dds, ds, df
    else:
        return df


def plot_pydeseq2_volcano(
    df: pd.DataFrame,
    symbol_column: str = "index",
    log2fc_column: str = "log2FoldChange",
    pvalue_column: str = "padj",
    baseMean_column: str = "baseMean",
    pval_thresh: float = 0.05,
    log2fc_thresh: float = 0.75,
    to_label: int | list | None = 5,
    scale_size: bool = False,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[Figure, Axes]:
    """
    Generate a volcano plot from a pyDESeq2 results dataframe.

    Adapted from https://github.com/mousepixels/sanbomics/blob/master/sanbomics/plots.py

    Args:
        df: a pyDESeq2 results dataframe
        symbol_column: column name of gene IDs to use
        log2fc_column: column name of log2 Fold-Change values
        pvalue_column: column name of the adjusted p values to be converted to -log10 p-values
        baseMean_column: column name of base mean values for each gene
        pval_thresh: threshold pvalue_column for points to be significant
        log2fc_thresh: threshold for the absolute value of the log2 fold change to be
         considered significant
        to_label: If an int is passed, that number of top down and up genes will be labeled.
            If a list of gene Ids is passed, only those will be labeled
        scale_size: scale the marker size in the plot (based on the baseMean values)
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.text

    Returns:
        matplotlib figure and axis object
    """
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    alpha = 0.33

    df = df.copy().reset_index(drop=False).dropna()
    pval_thresh = -np.log10(pval_thresh)
    min_value = min(1e-9, df[df[pvalue_column] > 0][pvalue_column].min() / 10)
    df["-log10(padj)"] = -np.log10(np.clip(df[pvalue_column], min_value, None))
    df["size"] = 100 * df[baseMean_column]
    if scale_size:
        scaler = MinMaxScaler(feature_range=(10, 100))
        df["size"] = scaler.fit_transform(df["size"].values.reshape(-1, 1))

    def map_genes(row):
        l2fc, log10p = row
        if log10p <= pval_thresh:
            if abs(l2fc) <= log2fc_thresh:
                return "NS"
            else:
                return "Log2FC"
        else:
            if abs(l2fc) <= log2fc_thresh:
                return "p-value"
            else:
                return "p-value & Log2FC"

    df["cat"] = df[[log2fc_column, "-log10(padj)"]].apply(map_genes, axis=1)
    cat2color = {
        "NS": "lightgrey",
        "p-value": "tab:blue",
        "Log2FC": "tab:green",
        "p-value & Log2FC": "tab:red",
    }
    df["color"] = df["cat"].map(cat2color)

    fig, ax = plt.subplots(**subplot_kwargs)
    df1 = df[df[pvalue_column] >= min_value]
    df2 = df[df[pvalue_column] < min_value]
    ax.scatter(
        x=df1[log2fc_column],
        y=df1["-log10(padj)"],
        c=df1["color"],
        s=df1["size"],
        marker="o",
        alpha=alpha,
        **plot_kwargs,
    )
    ax.scatter(
        x=df2[log2fc_column],
        y=df2["-log10(padj)"],
        c=df2["color"],
        s=df2["size"],
        marker="^",
        alpha=alpha,
        **plot_kwargs,
    )
    ax.axhline(pval_thresh, zorder=-1, c="k", lw=1.5, ls="--", alpha=alpha)
    ax.axvline(log2fc_thresh, zorder=-1, c="k", lw=1.5, ls="--", alpha=alpha)
    ax.axvline(-log2fc_thresh, zorder=-1, c="k", lw=1.5, ls="--", alpha=alpha)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("$log_{2}$ fold change")
    ax.set_ylabel(f"-$log_{{10}}$ adjusted p-value")

    # legend with color descriptions + dot size scale
    legend_elements = [
        # Line2D([0], [0], marker='o', color='w', label="color: ", markersize=0),
        # Line2D([0], [0], marker='o', color='w', label="baseMean: ", markersize=0)
    ]
    #     n = 0
    for cat, color in cat2color.items():
        legend_elements.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=cat,
                markerfacecolor=color,
                markersize=10,
                lw=0,
                alpha=alpha,
            )
        )

        # n += 1 / (len(cat2color) + 1)
        # s = df["size"].quantile(n)
        # bm = df[baseMean_column].quantile(n).round(2)
        # legend_elements.append(
        #     Line2D(
        #         [0],
        #         [0],
        #         marker="o",
        #         color="w",
        #         label=bm,
        #         markerfacecolor=cat2color["NS"],
        #         markersize=s,
        #         lw=0,
        #         alpha=0.33,
        #     )
        # )

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(0.0, 1.02, 1.0, 0.102),
        loc="lower left",
        ncols=len(cat2color),
        mode="expand",
        borderaxespad=0.0,
        frameon=False,
    )

    if to_label:
        # label the top and bottom n genes
        df["sorter"] = df["-log10(padj)"] * df[log2fc_column]
        if isinstance(to_label, int):
            label_df = pd.concat(
                (
                    df.sort_values("sorter")[-to_label:],
                    df.sort_values("sorter")[0:to_label],
                )
            )
        else:
            label_df = df[df[symbol_column].isin(to_label)]
        texts = []
        for i in range(len(label_df)):
            txt = ax.text(
                x=label_df.iloc[i][log2fc_column],
                y=label_df.iloc[i]["-log10(padj)"],
                s=label_df.iloc[i][symbol_column],
                **text_kwargs,
            )
            txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
            texts.append(txt)
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", zorder=5), ax=ax)

    return fig, ax
