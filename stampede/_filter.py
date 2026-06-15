from __future__ import annotations

import functools
import operator

import anndata as ad


def filter_edges(
    adata,
    all_edges: int = 0,
    left: int = 0,
    top: int = 0,
    right: int = 0,
    bottom: int = 0,
    slide: int = None,
    verbose: bool = True,
):
    """
    Filter cells based on their distance to one or more edges of its FOV.
    Uses the largest distance per edge.

    Args:
        adata: adata object
        all_edges: minimum distance from any edge in pixels
        left: minimum distance from the left edge in pixels (x = xmin + left)
        top: minimum distance from the top edge in pixels (y = ymin + top)
        right: minimum distance from the right edge in pixels (x = xmax - right)
        bottom: minimum distance from the bottom edge in pixels (y = ymax - bottom)
        slide: which slide to filter (default: all)
        verbose: provide written feedback

    Returns:
        the filtered adata object
    """
    filter_columns = []
    if left or right:
        left = max(left, all_edges)
        right = max(right, all_edges)
        filter_columns.append(
            adata.obs["CenterX_local_px"].between(
                0 + left, adata.uns["fov_dims_px"]["x"] - right
            )
        )
    if top or bottom:
        top = max(top, all_edges)
        bottom = max(bottom, all_edges)
        filter_columns.append(
            adata.obs["CenterY_local_px"].between(
                0 + top, adata.uns["fov_dims_px"]["y"] - bottom
            )
        )
    if all_edges:
        filter_columns.append(adata.obs["dist2edge_px"] >= all_edges)

    # combine all filters
    if len(filter_columns) == 0:
        return adata
    elif len(filter_columns) == 1:
        total_cell_filter = filter_columns[0]
    else:
        total_cell_filter = functools.reduce(operator.and_, filter_columns)

    before = len(adata.obs)
    if slide:
        # keep all cells from other slides
        if slide not in adata.obs["slide"]:
            raise ValueError(f"{slide=} not found in adata.ons['slide']!")
        adata = adata[total_cell_filter | (adata.obs["slide"] != slide), :].copy()
    else:
        adata = adata[total_cell_filter, :].copy()
    after = len(adata.obs)
    if verbose:
        print(f"{before - after:_} cells filtered out, {after:_} cell remaining.")
    return adata


def filter_genes(
    adata: ad.AnnData,
    ncell_min: int = 0,
    ncell_max: int = float("inf"),
    ntranscript_min: int = 0,
    ntranscript_max: int = float("inf"),
    filter_columns: str | list = None,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Filter adata.var by a set of qc_params.

    Args:
        adata: adata object
        ncell_min: minimum number of cells the gene is found in.
        ncell_max: maximum number of cells the gene is found in.
        ntranscript_min: minimum number of transcripts the gene must have.
        ntranscript_max: maximum number of transcripts the gene must have.
        filter_columns: a list of additional columns to filter by.
         Columns by (convertible to) boolean, where False values are removed.
        verbose: provide written feedback

    Returns:
        the filtered adata object
    """
    if filter_columns is None:
        filter_columns = []
    elif isinstance(filter_columns, str):
        filter_columns = [filter_columns]
    adata.strings_to_categoricals()
    filter_columns = [adata.obs[col] for col in filter_columns]

    required_cols = ["nCell", "nTranscript", "above_noise", "is_negctrl", "is_sysctrl"]
    missing = [col for col in required_cols if col not in adata.var.columns]
    if missing:
        raise ValueError(
            f"Not all required columns ({missing}) are present in adata.var. Run st.pp.gene_qc() first."
        )

    ncells_filter = adata.var["nCell"].between(ncell_min, ncell_max)
    filter_columns.append(ncells_filter)
    ntranscript_filter = adata.var["nTranscript"].between(
        ntranscript_min, ntranscript_max
    )
    filter_columns.append(ntranscript_filter)
    noise_filter = adata.var["above_noise"]
    filter_columns.append(noise_filter)
    negprobe_filter = ~adata.var["is_negctrl"]
    filter_columns.append(negprobe_filter)
    falsecode_filter = ~adata.var["is_sysctrl"]
    filter_columns.append(falsecode_filter)

    # combine all filters
    total_gene_filter = functools.reduce(operator.and_, filter_columns)

    before = len(adata.var)
    adata = adata[:, total_gene_filter]
    after = len(adata.var)
    if verbose:
        print(f"{before - after:_} genes filtered out, {after:_} genes remaining.")
    return adata


def filter_cells(
    adata: ad.AnnData,
    falsecode_max: int = 5,
    negprobe_max: int = 3,
    ntranscript_min: int = 0,
    ntranscript_max: int = float("inf"),
    area_min: int = 25,
    area_max: int = 100,
    filter_columns: list = None,
    filter_internalqc: bool = False,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Filter adata.obs by a set of qc_params.

    Args:
        adata: adata object
        falsecode_max: maximum number of false codes the cell may have
        negprobe_max: maximum number of negative probes the cell may have
        ntranscript_min: minimum number of transcripts the cell must have
        ntranscript_max: maximum number of transcripts the cell must have
        area_min: minimum area (in pixels) the cell must have
        area_max: maximum area (in pixels) the cell must have
        filter_columns: a list of additional columns to filter by.
         Columns by (convertible to) boolean, where False values are removed.
        filter_internalqc: filter by columns `qcCellsPassed` and `qcFlagsFOV`.
        verbose: provide written feedback

    Returns:
        the filtered adata object
    """
    if filter_columns is None:
        filter_columns = []
    elif isinstance(filter_columns, str):
        filter_columns = [filter_columns]
    # else:
    #     for col in filter_columns:
    #         if adata.obs[col].dtype != bool:
    #             raise TypeError(f"filter_column '{col}' must have a boolean dtype")
    adata.strings_to_categoricals()
    filter_columns = [adata.obs[col] for col in filter_columns]

    falsecode_filter = ~(adata.obs["nCount_falsecode"] >= falsecode_max)
    filter_columns.append(falsecode_filter)
    negprobe_filter = ~(adata.obs["nCount_negprobes"] >= negprobe_max)
    filter_columns.append(negprobe_filter)
    transcript_filter = adata.obs["nCount_RNA"].between(
        ntranscript_min, ntranscript_max
    )
    filter_columns.append(transcript_filter)
    area_filter = adata.obs["Area.um2"].between(area_min, area_max)
    filter_columns.append(area_filter)
    if filter_internalqc:
        internal_qc = adata.obs["qcCellsPassed"] & (adata.obs["qcFlagsFOV"] == "Pass")
        filter_columns.append(internal_qc)

    # combine all filters
    total_cell_filter = functools.reduce(operator.and_, filter_columns)

    before = len(adata.obs)
    adata = adata[total_cell_filter, :].copy()
    after = len(adata.obs)
    if verbose:
        print(f"{before - after:_} cells filtered out, {after:_} cells remaining.")
    return adata
