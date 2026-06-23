from __future__ import annotations

import itertools
import os
from collections.abc import Iterable, Sequence

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from matplotlib.typing import ColorType
from natsort import natsort_keygen, natsorted


def slide_qc(
    adata: ad.AnnData,
    slides: dict,
    add_cols: Iterable | str = None,
    data_dir: str = None,
) -> None:
    """
    Use the fov_positions file to create a dataframe with metadata columns per slide
    and fov, and store this in adata.uns["fov_metadata"].
    Additional adds columns to adata.obs reflecting the distance from the cell to the
    camera's FOV edge.

    Args:
        adata: adata object generated using the slides dict
        slides: a dictionary with the slide number as keys, and a dictionary as values.
          The value dict must contain keys "exprmat" and "metadata", with should map to
          matching respective files
        add_cols: additional columns to visualize (e.g. conditions)
        data_dir: optional filepath prefix

    Returns:
        Nothing, updates adata.uns and adata.obs
    """
    if data_dir is None:
        data_dir = ""
    if add_cols is None:
        add_cols = []
    elif isinstance(add_cols, str):
        add_cols = [add_cols]
    # add a composite column to adata.obs if it does not exist
    if "slide-fov" not in adata.obs.columns:
        adata.obs["slide-fov"] = (
            adata.obs["slide"].astype(str) + "-" + adata.obs["fov"].astype(str)
        )

    # get the coordinates of each fov from the fov_positions file
    fovsxy = []
    for slide, files in slides.items():
        fovdata = pd.read_csv(
            os.path.join(data_dir, files["fov_positions"]),
            usecols=["FOV", "x_global_px", "y_global_px"],
            dtype=int,
        )
        for _, row in fovdata.iterrows():
            fov = row["FOV"]
            idx = f"{slide}-{fov}"
            x = row["x_global_px"]
            y = row["y_global_px"]
            fovsxy.append([idx, slide, fov, x, y])
    fov_df = pd.DataFrame(
        data=fovsxy, columns=["slide-fov", "slide", "fov", "x", "y"]
    ).set_index("slide-fov")

    # Add additional metadata columns
    fov_df["nCounts"] = adata.obs.groupby("slide-fov", observed=True)[
        "nCount_RNA"
    ].sum()
    fov_df = pd.merge(
        left=fov_df,
        right=adata.obs["slide-fov"].value_counts().rename("nCell"),
        on="slide-fov",
    )
    fov_df["mean_CountsPerCell"] = fov_df["nCounts"] / fov_df["nCell"]
    fov_df["nCount_negprobes"] = adata.obs.groupby("slide-fov", observed=True)[
        "nCount_negprobes"
    ].sum()
    fov_df["mean_NegProbe-CountsPerCell"] = fov_df["nCount_negprobes"] / fov_df["nCell"]
    fov_df["nCount_falsecode"] = adata.obs.groupby("slide-fov", observed=False)[
        "nCount_falsecode"
    ].sum()
    fov_df["mean_FalseCode-CountsPerCell"] = (
        fov_df["nCount_falsecode"] / fov_df["nCell"]
    )
    fov_df["mean_CellSize"] = (
        adata.obs.groupby("slide-fov", observed=False)["Area.um2"].sum()
        / fov_df["nCell"]
    )
    if "Failed_AtoMX_QC" in adata.obs:
        slidefov2passfail = (
            adata.obs.groupby("slide-fov", observed=False)["qcFlagsFOV"]
            .first()
            .to_dict()
        )
        fov_df = fov_df.reset_index()
        fov_df["Failed_AtoMX_QC"] = (
            fov_df["slide-fov"]
            .replace(slidefov2passfail)
            .replace({"Pass": "0", "Fail": "1"})
            .astype(int)
        )
    for col in add_cols:
        # Take most-occurring value in col per slide-fov:
        fov_df[col] = adata.obs.groupby("slide-fov", observed=False)[col].agg(
            lambda val: val.mode()[0]
        )

    adata.uns["fov_metadata"] = fov_df

    # determine the dimensions of the camera's FOV
    dim_x_px, dim_y_px = _fov_dimensions(fov_df)
    adata.uns["fov_dims_px"] = {"x": dim_x_px, "y": dim_y_px}

    # compute the distance of each cell to the nearest edge along the x- and y-axis
    if "dist2edge_px" not in adata.obs.columns:
        adata.obs["x2edge_px"] = np.minimum(
            adata.obs["CenterX_local_px"], dim_x_px - adata.obs["CenterX_local_px"]
        )
        adata.obs["y2edge_px"] = np.minimum(
            adata.obs["CenterY_local_px"], dim_y_px - adata.obs["CenterY_local_px"]
        )
        adata.obs["dist2edge_px"] = np.minimum(
            adata.obs["x2edge_px"], adata.obs["y2edge_px"]
        )


def gene_qc(
    adata: ad.AnnData,
    mult: int | float = 1,
    noise_threshold: float | Iterable = None,
    overwrite: bool = True,
) -> None:
    r"""
    Add QC parameters to adata.var.

    About the Signal-to-noise filter:
        Approach from https://doi.org/10.1038/s41467-025-64990-y
        Wang et al. "Systematic benchmarking of imaging spatial
        transcriptomics platforms in FFPE tissues" Nat Com, 2025.

        Calculate the mean expression and standard deviation of the negative control
        probes.
        Flag genes with average expression < mean + mult\* x STD of ctrl probes.

        \*the paper used mult=2

    Args:
        adata: an adata object
        noise_threshold: manually specify the mimimum mean_Transcript threshold.
         If None, use the filter specified above.
        mult: if noise_threshold is None, mult is used in the noise
         threshold computation specified above.
        overwrite: overwrite existing qc columns

    Returns:
        Nothing, updates adata.var
    """
    if "is_negctrl" not in adata.var.columns or overwrite:
        adata.var["is_negctrl"] = adata.var_names.str.startswith("Negative")
    if "is_sysctrl" not in adata.var.columns or overwrite:
        adata.var["is_sysctrl"] = adata.var_names.str.startswith("System")
    if "nCell" not in adata.var.columns or overwrite:
        # number of nonzero cells per gene
        adata.var["nCell"] = (adata.X > 0).sum(axis=0).A1
        adata.var["pctCell"] = 100 * adata.var["nCell"] / adata.n_obs
    if "nTranscript" not in adata.var.columns or overwrite:
        adata.var["nTranscript"] = np.array(adata.X.sum(axis=0)).ravel()
    if "mean_Transcript" not in adata.var.columns or overwrite:
        adata.var["mean_Transcript"] = adata.var["nTranscript"] / adata.n_obs
    if "above_noise" not in adata.var.columns or overwrite:
        if noise_threshold is None:
            negctrls = adata.var.loc[adata.var["is_negctrl"], "mean_Transcript"]
            mean = negctrls.mean()
            std = negctrls.std()
            noise_threshold = mean + mult * std
            adata.uns["noise_values"] = {
                "mean": mean,
                "std": std,
                "mult": mult,
                "threshold": noise_threshold,
            }
        adata.var["above_noise"] = adata.var["mean_Transcript"] > noise_threshold


def plot_noise_threshold(adata: ad.AnnData, bins: int = 50, **kwargs):
    fig, ax = plot_column_distribution(
        adata,
        "mean_Transcript",
        plot_kwargs={"bins": bins, "label": "mean transcripts"},
        **kwargs,
    )
    ax.axvline(
        adata.uns["noise_values"]["mean"], color="red", ls="--", label="mean noise"
    )
    ax.axvline(
        adata.uns["noise_values"]["threshold"], color="red", label="noise threshold"
    )
    ax.legend()

    title = ax.get_title()
    t = int(adata.var["above_noise"].value_counts()[True])
    n = len(adata.var)
    ax.set_title(title + f"\n {t:_}/{n:_} genes passed threshold")

    return fig, ax


def gene_qc_postfilter(adata: ad.AnnData) -> None:
    """
    Compute metadata after filtering

    Args:
        adata: an adata object

    Returns:
        Nothing, updates adata.var
    """

    adata.var["nCell_postfilter"] = adata.X.count_nonzero(axis=0)
    adata.var["pctCell_postfilter"] = (
        100 * adata.var["nCell_postfilter"] / adata.n_obs
    ).round(2)

    if adata.var["nCell"].eq(adata.var["nCell_postfilter"]).all():
        raise ValueError("adata was not filtered")


def cell_qc_postfilter(adata: ad.AnnData) -> None:
    """
    Compute metadata after filtering

    Args:
        adata: an adata object

    Returns:
        Nothing, updates adata.obs
    """

    adata.obs["nFeature_RNA_postfilter"] = adata.X.count_nonzero(axis=1)
    adata.obs["nCount_RNA_postfilter"] = adata.X.sum(axis=1)

    if adata.obs["nCount_RNA"].eq(adata.obs["nCount_RNA_postfilter"]).all():
        raise ValueError("adata was not filtered")


def _fov_dimensions(fov_df):
    dx = []
    dy = []
    for slide in sorted(fov_df["slide"].unique()):
        last_x = None
        last_y = None
        df = fov_df[fov_df["slide"] == slide].sort_values("fov")
        for fov, row in df.iterrows():
            x, y = row["x"], row["y"]
            if last_x is None:
                pass
            else:
                # compute dx if the fov only changed along the x-axis
                # compute dy if the fov changed along the y-axis
                if y == last_y:
                    _dx = abs(x - last_x)
                    dx.append(_dx)
                else:
                    _dy = abs(y - last_y)
                    dy.append(_dy)
            last_x = x
            last_y = y
    # median distance between 2 FOVs in pixels
    x_px = round(sorted(dx)[len(dx) // 2])
    y_px = round(sorted(dy)[len(dy) // 2])
    return x_px, y_px


def plot_slide_qc(
    adata: ad.AnnData,
    columns: str | Iterable = None,
    figsize: tuple = None,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    legend_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the values from one or more QC columns in adata.uns["fov_metadata"]
    (added by `slide_qc_data()`).
    Specify columns to limit the number of plots.

    Args:
        adata: an adata object
        columns: columns in adata.uns["fov_metadata"] to plot (default: all)
        figsize: tuple of figure, will be multiplied by the number of plots
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        legend_kwargs: kwargs passed to the legends

    Returns:
        matplotlib figure and array of axes
    """
    if figsize is None:
        figsize = (2.5, 4)
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if legend_kwargs is None:
        legend_kwargs = {}
    fov_df = adata.uns["fov_metadata"]
    required_cols = ["slide", "x", "y"]
    for col in required_cols:
        if col not in fov_df.columns:
            raise ValueError(f"column={col} is required in adata.obs")
    if columns is None:
        columns = fov_df.columns.to_list()
    elif isinstance(columns, str):
        columns = [columns]
    fov_df = fov_df[[c for c in columns if c not in required_cols] + required_cols]

    slides = natsorted(fov_df["slide"].unique())
    blacklist = {"slide-fov", "slide", "fov", "x", "y"}
    qc_columns = [c for c in fov_df.columns if c not in blacklist]
    nrows = len(slides) + 1
    ncols = len(qc_columns)
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        squeeze=False,
        figsize=(figsize[0] * ncols, figsize[1] * nrows + 1),
        sharex=True,
        sharey=True,
        gridspec_kw={
            "wspace": 0,
            "hspace": 0,
            "height_ratios": [1.5] + [4] * len(slides),
        },
        layout="tight",
        **subplot_kwargs,
    )
    fig.supylabel("y", x=1.0, ha="right", rotation=0)
    fig.supxlabel("x")

    for i_col, column in enumerate(qc_columns):
        if pd.api.types.is_numeric_dtype(fov_df[column]):
            # normalize the colormap for all slides at once
            norm = Normalize(vmin=fov_df[column].min(), vmax=fov_df[column].max())
            palette = "coolwarm"
            show_legend = False
        else:
            norm = None
            palette = "tab10"
            show_legend = True

        axs[0, i_col].axis("off")

        handles, labels = [], []
        for i_row, slide in enumerate(slides):
            ax = axs[i_row + 1, i_col]

            # main plot
            sns.scatterplot(
                data=fov_df[fov_df["slide"] == slide],
                x="x",
                y="y",
                hue=column,
                palette=palette,
                hue_norm=norm,
                legend=show_legend,
                ax=ax,
                **plot_kwargs,
            )
            if not pd.api.types.is_numeric_dtype(fov_df[column]):
                handles, labels = ax.get_legend_handles_labels()
                ax.get_legend().remove()

            # tweak all ticks, borders & labels
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            if i_col == 0:
                ax.set_ylabel(f"slide {slide}")

            # only plot the colorbar once per column
            if i_row == 0:
                if pd.api.types.is_numeric_dtype(fov_df[column]):
                    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(
                        sm,
                        location="bottom",
                        orientation="horizontal",
                        shrink=0.4,
                        pad=0.1,
                        aspect=30,
                        ax=axs[0, i_col],
                    )
                    # colorbar label doubles as the figure title
                    cbar.set_label(column)
                    cbar.ax.xaxis.set_label_position("top")
                else:
                    axs[0, i_col].legend(
                        handles=handles,
                        labels=labels,
                        loc="lower center",
                        bbox_to_anchor=(0.5, 0.0),
                        frameon=False,
                        title=column,
                        **legend_kwargs,
                    )

    # return the plot elements for manual post-processing
    return fig, axs


# def plot_fov_edge_distances(
#     adata,
#     figsize=(12, 6),
#     subplot_kwargs=None,
#     # plot_kwargs=None,
# ):
#     bins = 50
#     cmap = "Blues"
#     color1 = "dodgerblue"
#     color2 = "limegreen"
#     if subplot_kwargs is None:
#         subplot_kwargs = {}
#     # if plot_kwargs is None:
#     #     plot_kwargs = {}
#
#     fig, axs = plt.subplots(
#         ncols=3,
#         gridspec_kw={"width_ratios": [1, 1, 0.05]},
#         constrained_layout=True,
#         figsize=figsize,
#         **subplot_kwargs,
#     )
#     x, y = adata.uns["fov_dims_px"].values()
#     fig.suptitle(
#         "Distributions of cell distances to the edge of the camera's FOV "
#         + f"({x}x{y} pixels)"
#     )
#
#     # 1D histplot
#     sns.histplot(
#         np.log1p(adata.obs["dist2edge_px"]),
#         bins=bins,
#         stat="percent",
#         color=color1,
#         ax=axs[0],
#         alpha=0.5,
#     )
#     axs[0].set_title("Minimum distance to the FOV edge")
#     axs[0].set_xlabel("log1p(pixels)", color=color1)
#     axs[0].set_ylabel("% of cells")
#     axs01 = axs[0].twiny()
#     sns.histplot(
#         adata.obs["dist2edge_px"],
#         bins=bins,
#         stat="percent",
#         color=color2,
#         ax=axs01,
#         alpha=0.5,
#     )
#     axs01.set_xlabel("pixels", color=color2)
#     axs[0].set_zorder(1)
#     axs01.set_zorder(0)
#     axs[0].patch.set_alpha(0)  # background transparency
#
#     # 2D histplot
#     sns.histplot(
#         x=adata.obs["x2edge_px"],
#         y=adata.obs["y2edge_px"],
#         cmap=cmap,
#         cbar=False,
#         ax=axs[1],
#     )
#     axs[1].set_title("2D distance to FOV edge")
#     axs[1].set_xlabel("Pixels along the x-axis")
#     axs[1].set_ylabel("Pixels along the y-axis")
#
#     # colormap
#     norm = Normalize(
#         vmin=0,
#         vmax=max(adata.obs["x2edge_px"].max(), adata.obs["y2edge_px"].max()),
#     )
#     ColorbarBase(axs[2], cmap=cmap, norm=norm)
#     axs[2].set_ylabel("Number of cells")
#
#     return fig, axs


def plot_correlations(
    adata: ad.AnnData,
    xcolumn: str,
    ycolumn: str,
    log1p_xcolumn: bool = False,
    log1p_ycolumn: bool = False,
    color_xcolumn: ColorType = None,
    color_ycolumn: ColorType = None,
    cmap_2d: ColorType = "Blues",
    min_quantile: float = 0.0,
    max_quantile: float = 0.99,
    bins_1d: str | int = "auto",
    bins_2d: str | int = "auto",
    stat: str = "percent",
    figsize: tuple = (8, 7),
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the distributions and 2D correlation between two columns in adata.obs.

    Args:
        adata: an adata object
        xcolumn: columns in adata.obs to plot on the x-axis
        ycolumn: columns in adata.obs to plot on the y-axis
        log1p_xcolumn: normalize the xcolumn?
        log1p_ycolumn: normalize the ycolumn?
        color_xcolumn: color of the xcolumn plot
        color_ycolumn: color of the ycolumn plot
        cmap_2d: colormap of the 2d correlation plot
        min_quantile: lowest quantile of values to plot
        max_quantile: highest quantile of values to plot
        bins_1d: number of bins on the 1-dimensional histogram plots
        bins_2d: number of bins on the 2-dimensional histogram plot
        stat: which statistic to plot, see sns.histplot for more details
        figsize: figure size
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        gridspec_kw={
            "width_ratios": [1, 1, 0.05],
            "height_ratios": [1, 1],
            # "wspace": 0,
            # "hspace": 0,
        },
        constrained_layout=True,
        figsize=figsize,
        **subplot_kwargs,
    )

    x = adata.obs[xcolumn]
    x_min = np.quantile(x, min_quantile)
    x_max = np.quantile(x, max_quantile)
    y = adata.obs[ycolumn]
    y_min = np.quantile(y, min_quantile)
    y_max = np.quantile(y, max_quantile)
    keep = adata.obs[x.between(x_min, x_max) & y.between(y_min, y_max)].index

    y = y.loc[keep]
    ylabel = ycolumn
    if log1p_ycolumn:
        y = np.log1p(y)
        ylabel = f"log1p({ycolumn})"
    sns.histplot(
        y=y,
        bins=bins_1d,
        stat=stat,
        color=color_ycolumn,
        ax=axs[0, 0],
        alpha=0.5,
        **plot_kwargs,
    )
    axs[0, 0].set_ylim(y_min, y_max)
    axs[0, 0].invert_xaxis()
    axs[0, 0].yaxis.tick_right()
    axs[0, 0].xaxis.tick_top()
    axs[0, 0].xaxis.set_label_position("top")
    # axs[0, 0].yaxis.set_label_position("right")
    axs[0, 0].set_ylabel(ylabel)

    x = x.loc[keep]
    xlabel = xcolumn
    if log1p_xcolumn:
        x = np.log1p(x)
        xlabel = f"log1p({xcolumn})"
    sns.histplot(
        x=x,
        bins=bins_1d,
        stat=stat,
        color=color_xcolumn,
        ax=axs[1, 1],
        alpha=0.5,
        **plot_kwargs,
    )
    axs[1, 1].set_xlim(x_min, x_max)
    axs[1, 1].invert_yaxis()
    axs[1, 1].xaxis.tick_top()
    axs[1, 1].yaxis.tick_right()
    axs[1, 1].yaxis.set_label_position("right")
    # axs[1, 1].set_xlabel(None)
    axs[1, 1].set_xlabel(xlabel)

    sns.histplot(
        x=x,
        y=y,
        bins=bins_2d,
        stat=stat,
        cmap=cmap_2d,
        cbar=False,
        ax=axs[0, 1],
        **plot_kwargs,
    )
    axs[0, 1].set_xlabel(None)
    axs[0, 1].set_ylabel(None)
    axs[0, 1].set_xlim(x_min, x_max)
    axs[0, 1].set_ylim(y_min, y_max)
    # axs[0, 1].set_xlabel(xlabel)
    # axs[0, 1].xaxis.set_label_position("top")
    # axs[0, 1].set_ylabel(ylabel)
    # axs[0, 1].yaxis.set_label_position("right")
    axs[0, 1].get_xaxis().set_ticklabels([])
    axs[0, 1].get_yaxis().set_ticklabels([])
    # axs[0, 1].get_xaxis().set_visible(False)
    # axs[0, 1].get_yaxis().set_visible(False)

    # colormap
    norm = Normalize(vmin=min(y.min(), x.min()), vmax=max(y.max(), x.max()))
    ColorbarBase(axs[0, 2], cmap=cmap_2d, norm=norm)
    cbar_label = "Number of cells"
    if log1p_xcolumn and log1p_ycolumn:
        cbar_label = "log1p(cells)"
    axs[0, 2].set_ylabel(cbar_label)

    # hide empty subfigs
    axs[1, 0].set_axis_off()
    axs[1, 2].set_axis_off()

    return fig, axs


def plot_avg_per_pixel(
    adata: ad.AnnData,
    column: str,
    fill_cell_area: bool = True,
    normalize_cell_area: bool = True,  # causes edge effects
    log1p: bool = False,
    cmap: ColorType = "gist_rainbow",
    background_color: ColorType = "black",
    figsize: tuple = (20, 15),
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the average values of the given column over all FOVs.
    Color's the cell's center pixel, unless fill_cell_area is set to True (slow).

    Args:
        adata: an adata object
        column: a column in adata.obs with numeric values
        fill_cell_area: distribute the column value over all pixels covered by the cell,
         assuming square cells
        normalize_cell_area: if fill_cell_area is True, normalize the column value
         over the cell area
        log1p: normalize the final values per pixel?
        cmap: colormap (default: "gist_rainbow")
        background_color: color for pixels with 0 values (default: "black")
        figsize: figure size
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    # colorbar label
    label = column
    if normalize_cell_area:
        label = f"{label}/cell area"
    label = f"mean({label})"
    if log1p:
        label = f"log1p({label})"

    # create a 2D array with the average values
    x_max = adata.uns["fov_dims_px"]["x"]
    y_max = adata.uns["fov_dims_px"]["y"]
    grid = np.full((y_max, x_max), 0.0)
    if not fill_cell_area:
        # Group and average per coordinate
        df = (
            adata.obs[[column, "CenterX_local_px", "CenterY_local_px"]]
            .groupby(["CenterX_local_px", "CenterY_local_px"], observed=True)
            .mean()
            .reset_index()
        )
        grid[df["CenterY_local_px"], df["CenterX_local_px"]] = df[column]
    else:
        grid_n = grid.copy()
        for _, row in adata.obs.iterrows():
            val = row[column]
            if normalize_cell_area:
                # divide the column value over the area of the cell
                val = val / (row["Width"] * row["Height"])  # row['Area']

            # add the normalized value to all pixels
            half_w = row["Width"] // 2
            half_h = row["Height"] // 2
            x_start = max(row["CenterX_local_px"] - half_w, 0)
            x_end = min(row["CenterX_local_px"] + half_w + 1, x_max)
            y_start = max(row["CenterY_local_px"] - half_h, 0)
            y_end = min(row["CenterY_local_px"] + half_h + 1, y_max)
            grid[y_start:y_end, x_start:x_end] += val

            # track the number of cells covering each pixel
            grid_n[y_start:y_end, x_start:x_end] += 1

        # average the value over the number of cells
        np.clip(grid_n, min=1, max=None, out=grid_n)
        grid = grid / grid_n
    if log1p:
        grid = np.log1p(grid)

    # assign the background_color
    cmap = plt.get_cmap(cmap).copy()
    cmap.set_bad(color=background_color)
    masked_grid = np.ma.masked_where(grid == 0.0, grid)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        gridspec_kw={
            "width_ratios": [2, 1],
        },
        figsize=figsize,
        **subplot_kwargs,
    )
    n = len(adata.obs["slide-fov"].unique())
    axs[0].set_title(f"Average {column} per pixel over {n} FOVs")
    vmin = np.nanmin(grid[grid > 0])  # first real, nonzero value
    vmax = np.nanmax(grid)
    im = axs[0].imshow(
        masked_grid,
        cmap=cmap,
        interpolation="none",
        # aspect='equal',
        vmin=vmin,
        vmax=vmax,
        **plot_kwargs,
    )
    axs[0].set_box_aspect(1)
    axs[0].xaxis.set_ticks(np.arange(0, x_max, (x_max // 2000) * 100))
    axs[0].yaxis.set_ticks(np.arange(0, y_max, (y_max // 2000) * 100))
    axs[0].yaxis.set_minor_locator(MultipleLocator(100))
    axs[0].xaxis.tick_top()
    axs[0].xaxis.set_minor_locator(MultipleLocator(100))
    axs[0].xaxis.set_tick_params(rotation=45, which="minor", length=2)
    axs[0].xaxis.set_label_position("top")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    # colorbar
    norm = Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(
        im,
        cmap=cmap,
        norm=norm,
        ax=axs[0],
        location="bottom",
        # orientation='horizontal',
        fraction=0.025,
        pad=0.005,
        label=label,
    )

    # Lineplot
    axs[1].set_title(f"Average values per axis")
    x_average = np.average(grid, axis=0)
    x_stdev = np.std(grid, axis=0)
    y_average = np.average(grid, axis=1)
    y_stdev = np.std(grid, axis=1)
    axs[1].set_xlim(0, max(x_max, y_max))
    axs[1].set_ylim(
        min(
            np.quantile(y_average - y_stdev, 0.02),
            np.quantile(x_average - x_stdev, 0.02),
        )
        * 0.98,
        max(
            np.quantile(y_average + y_stdev, 0.98),
            np.quantile(x_average + x_stdev, 0.98),
        )
        * 1.02,
    )
    axs[1].plot(x_average, label="mean x-axis")
    axs[1].fill_between(
        range(len(x_average)),
        x_average - x_stdev,
        x_average + x_stdev,
        alpha=0.2,
        label="std x-axis",
    )
    axs[1].plot(y_average, label="mean y-axis")
    axs[1].fill_between(
        range(len(y_average)),
        y_average - y_stdev,
        y_average + y_stdev,
        alpha=0.2,
        label="std y-axis",
    )
    axs[1].set_box_aspect(1)
    axs[1].set_ylabel(label)
    axs[1].set_xlabel("axis coordinate")
    axs[1].xaxis.set_minor_locator(MultipleLocator(100))
    axs[1].tick_params(which="minor", length=2)
    axs[1].legend()

    return fig, axs


def plot_violin(
    adata: ad.AnnData,
    columns: str | list,
    inner: str = "quart",
    fill: bool = False,
    cut: int = 0,
    log_scale: bool | Sequence[bool] = False,
    figsize: tuple = None,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Violin plots for one or more columns in adata.obs.

    Wraps seaborn's violinplot.
    See https://seaborn.pydata.org/generated/seaborn.violinplot.html

    Args:
        adata: an adata object
        columns: one or more column in adata.obs
        inner: See sns.violinplot for more details
        fill: See sns.violinplot for more details
        cut: See sns.violinplot for more details
        log_scale: See sns.violinplot for more details
        figsize: tuple of figure, will be multiplied by the number of plots
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if isinstance(columns, str):
        columns = [columns]
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    ncols = len(columns)

    if figsize is None:
        figsize = (2 * ncols, 4)
    fig, axs = plt.subplots(
        nrows=1, ncols=ncols, squeeze=False, figsize=figsize, **subplot_kwargs
    )
    color_cycle = itertools.cycle(plt.get_cmap("tab10").colors)  # noqa

    if isinstance(log_scale, bool):
        log_scale = (log_scale,) * ncols
    else:
        if len(log_scale) != len(columns):
            raise ValueError(
                f"Length of log_scale must match number of columns to plot."
            )

    for i, column in enumerate(columns):
        color = next(color_cycle)
        sns.violinplot(
            y=adata.obs[column],
            color=color,
            inner=inner,
            fill=fill,
            cut=cut,
            log_scale=(False, log_scale[i]),
            ax=axs[0, i],
            **plot_kwargs,
        )
        # sns.stripplot(
        #     y=adata.obs[key],
        #     color=color,
        #     alpha=0.005,
        #     log_scale=log_scale,
        #     ax=axs[i],
        #     **plot_kwargs,
        # )
        axs[0, i].spines["top"].set_visible(False)
        axs[0, i].spines["right"].set_visible(False)
        axs[0, i].spines["bottom"].set_visible(False)
        axs[0, i].spines["left"].set_visible(True)
        axs[0, i].set_title(column)

    fig.tight_layout()
    return fig, axs


def plot_ncell_per_condition(
    adata: ad.AnnData,
    columns: str | list,
    offset_between_conditions: int | list = 1,
    palette: ColorType | dict[str, str] = "terrain",
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the number of cells per condition in a column in adata.obs.

    Args:
        adata: an adata object
        columns: one or more columns in adata.obs to visualize, in order of significance
        offset_between_conditions: distance between different conditions
         Can be a single value, or a list of offset values for each column
         (length=len(columns)-1)
        palette: color palette (default: "terrain")
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.set_xticks and ax.set_yticks

    Returns:
        matplotlib figure and array of axes
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(offset_between_conditions, Iterable):
        if len(offset_between_conditions) != len(columns) - 1:
            raise IndexError(
                f"{offset_between_conditions=} must be a value, "
                "or a list of values of len(columns)-1"
            )
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}

    # use a separator that is unlikely to be in the data
    endash = "–"
    col_orders = []
    for col in columns:
        series = adata.obs[col]
        if hasattr(series, "cat"):
            col_orders.append(series.cat.categories.tolist())
        else:
            unique_vals = series.astype(str).unique()
            col_orders.append(sorted(unique_vals, key=natsort_keygen()))

    full_index = [endash.join(combo) for combo in itertools.product(*col_orders)]
    ncell = (
        adata.obs[columns]
        .astype(str)
        .agg(endash.join, axis=1)
        .value_counts()
        .reindex(full_index, fill_value=0)
    )

    x = []
    y = []
    labels = []
    idx2color = {}
    i = 0
    last = None
    for idx, val in ncell.items():
        y.append(val)
        labels.append(idx)
        idx_sep = idx.split(endash)
        if not isinstance(offset_between_conditions, Iterable):
            if last is None:
                last = idx_sep[0]
            # add the offset if the first column changes
            if idx_sep[0] != last:
                i += offset_between_conditions
                last = idx_sep[0]
        else:
            if last is None:
                last = idx_sep
            # add the j-th offset if the j-th column changes
            for j, offset in enumerate(offset_between_conditions):
                if idx_sep[j] != last[j]:
                    i += offset
                    last = idx_sep
                    break
        x.append(i)
        if isinstance(palette, dict):
            idx2color[i] = palette[idx]
        i += 1

    if isinstance(palette, dict):
        palette = idx2color

    fig, ax = plt.subplots(**subplot_kwargs)
    sns.barplot(
        x=x,
        y=y,
        edgecolor=".9",
        native_scale=True,  # use x & y values as-is
        palette=palette,
        hue=x,
        legend=False,
        ax=ax,
        **plot_kwargs,
    )
    ax.set_xticks(
        ticks=x,
        labels=labels,
        rotation=45,
        ha="right",
        **text_kwargs,
    )
    if text_kwargs:
        ax.set_yticks(
            ticks=ax.get_yticks(),
            labels=ax.get_yticklabels(),
            **text_kwargs,
        )
    ax.set_xlim(min(x) - 2, max(x) + 2)
    ax.set_title(f"Number of cells per {endash.join(columns)}", **text_kwargs)
    ax.set_xlabel(None)
    ax.set_ylabel("Number of cells", **text_kwargs)
    return fig, ax


def plot_value_distribution(
    adata: ad.AnnData,
    layer: str = None,
    min_quantile: float = 0.00,
    max_quantile: float = 0.95,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, list[Axes]]:
    """
    Plot the number of occurrences of values in the dataset.

    Args:
        adata: an adata object.
        layer: the layer the values are drawn from (default: X)
        min_quantile: lowest quantile of values to plot
        max_quantile: highest quantile of values to plot
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if layer is None:
        array = adata.X
    else:
        array = adata.layers[layer]
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    if isinstance(array, sp.csr_matrix):
        data = array.data
        values, counts = np.unique(data, return_counts=True)

        total_elements = array.shape[0] * array.shape[1]
        num_zeros = total_elements - len(data)

        if num_zeros > 0:
            values = np.concatenate([[0], values])
            counts = np.concatenate([[num_zeros], counts])

        # Sort by value
        sort_idx = np.argsort(values)
        values, counts = values[sort_idx], counts[sort_idx]
    else:
        values, counts = np.unique(array, return_counts=True, sorted=True)
    log_counts = np.log10(counts)
    x_min = np.quantile(values, min_quantile)
    x_max = np.quantile(values, max_quantile)
    x_pad = (x_max - x_min) * 0.02

    fig, ax = plt.subplots(**subplot_kwargs)
    x_prev = None
    for x, y in zip(values, log_counts):
        if x < x_min:
            continue
        if x > x_max:
            break
        if x_prev is None:
            width = values[1] - values[0]
        else:
            width = x - x_prev
        x_prev = x
        ax.bar(x=x, height=y, width=width, **plot_kwargs)
    ax.set_title("Distributions of raw values in the dataset")
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_xlabel("value")
    ax.set_ylabel("log10(occurrences)")

    return fig, ax


def plot_column_distribution(
    adata: ad.AnnData,
    column: str,
    axis: int = None,
    min_quantile: float = 0.00,
    max_quantile: float = 0.95,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
) -> tuple[Figure, Axes]:
    """
    Plot the distribution of values for a column present in either adata.obs or
    adata.var.

    Args:
        adata: an adata object.
        column: a column in either adata.obs or adata.var
        axis: specify if the column name is present in both obs (0) and var (1).
        min_quantile: lowest quantile of values to plot
        max_quantile: highest quantile of values to plot
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function

    Returns:
        matplotlib figure and array of axes
    """
    if axis is None:
        if column in adata.obs.columns:
            axis = 0
        elif column in adata.var.columns:
            axis = 1
        else:
            raise IndexError(f"{column=} not found in adata.obs or var")
    if axis == 0:
        series = adata.obs[column]
        unit = "cells"
    elif axis == 1:
        series = adata.var[column]
        unit = "genes"
    else:
        raise IndexError("Axis must be None, 0 or 1")
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}

    x_min = series.quantile(min_quantile)
    x_max = series.quantile(max_quantile)
    series = series[series.between(x_min, x_max)]

    fig, ax = plt.subplots(**subplot_kwargs)
    sns.histplot(series, ax=ax, **plot_kwargs)
    ax.set_xlim(x_min, x_max)
    ax.set_title(f"Distribution of {column} over all {unit}")

    return fig, ax
