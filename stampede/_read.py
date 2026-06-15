from __future__ import annotations

import os
import warnings

import anndata as ad
import pandas as pd
import scipy.sparse as sp

from . import config


def validate_input(
    slides: dict, samples_df: pd.DataFrame, data_dir: str = None
) -> None:
    """
    Check the contents of the slides dictionary and samples_df for expected keys and
    columns, respectively.

    Args:
        slides: a dictionary with the slide number as keys, and a dictionary as values.
          The value dict must contain keys "exprmat" and "metadata", with should map to
          matching respective files
        samples_df: a dataframe with sample metadata
        data_dir: optional filepath prefix

    Returns:
        Nothing
    """
    # validate the slides dictionary
    expected_keys = ["exprmat", "metadata", "fov_positions"]
    for slide, d in slides.items():
        if not isinstance(slide, int):
            raise TypeError(
                f"Keys for dictionary `slides` must be integer values, not {type(slide)=} (found {slide=})"
            )

        for key in expected_keys:
            if key not in d.keys():
                raise KeyError(f"Missing {key=} in slides dictionary")
        for filename in d.values():
            if data_dir:
                filename = os.path.join(data_dir, filename)
            if not os.path.exists(filename):
                raise FileNotFoundError(filename)

    # a "slide" column is optional with only 1 slide
    if len(slides) == 1 and "slide" not in samples_df.columns:
        # print("Adding column 'slide' to the samples_df")
        samples_df["slide"] = list(slides)[0]

    # validate the samples_df
    for column in config["sample_md_columns"]:
        if column not in samples_df.columns:
            raise ValueError(f"{column=} not found in samples_df")

    # validate overlap between the slides dictionary and the samples_df
    for slide in slides:
        if slide not in set(samples_df["slide"]):
            raise ValueError(f"{slide=} not found in samples_df['slide']")


def read_cosmx(
    slides: dict,
    samples_df: pd.DataFrame,
    adata_file: str,
    samples_df_columns: list = None,
    metadata_df_columns: list = None,
    data_dir: str = None,
    overwrite: bool = True,
    verbose: bool = True,
    **kwargs,
) -> str:
    """
    Read exprMat_file for each slide, convert the contents to sparse anndata objects,
    and concatenate the results.

    Args:
        slides: a dictionary with the slide number as keys, and a dictionary as values.
          The value dict must contain keys "exprmat" and "metadata", with should map to
          matching respective files
        samples_df: a dataframe with sample metadata to be added to adata.obs
        adata_file: filepath to write the adata object to
        samples_df_columns: list of columns in samples_df to add to adata.obs (default: all)
        metadata_df_columns: list of columns in the metadata file to add to adata.obs (default: all)
        data_dir: optional filepath prefix
        overwrite: overwrite existing output
        verbose: provide written feedback
        **kwargs: keyword arguments passed to pd.read_csv

    Returns:
        the value of the adata_file argument
    """
    if data_dir is None:
        data_dir = ""
    for col in ["usecols", "dtype", "chunksize"]:
        if col in kwargs:
            raise NotImplementedError(f"cannot use '{col}' in kwargs")
    if overwrite is False and os.path.exists(adata_file):
        if verbose:
            print(f"adata_file already exists and {overwrite=}")
        return adata_file

    adatas = []
    for slide, files in slides.items():
        if verbose:
            print(f"Slide {slide}/{len(slides)} starting...")

        # filename to write slide-specific adata object to
        dirname = os.path.dirname(adata_file)
        basename = os.path.basename(adata_file)
        base, suffix = basename.split(".h5ad")
        adata_file_slide = os.path.join(dirname, f".{base}_slide{slide}.h5ad{suffix}")

        adatas.append(adata_file_slide)
        if overwrite is False and os.path.exists(adata_file_slide):
            if verbose:
                print(f"Slide {slide}/{len(slides)} done!\n")
            continue

        fname = os.path.join(data_dir, files["exprmat"])

        # get the gene columns from the exprMat_file
        columns = pd.read_csv(fname, dtype=int, nrows=0).columns.to_list()
        for col in config.get("exprmat_md_columns"):
            if col not in columns:
                raise ValueError(f"column={col} not found in {files['exprmat']}")
            columns.remove(col)

        # get the cell metadata from the exprMat_file, the metadata_file and the samples_df
        obs = pd.read_csv(
            fname, dtype=int, usecols=config.get("exprmat_md_columns"), **kwargs
        )
        obs["slide"] = slide
        obs["slide-fov"] = f"{slide}-" + obs["fov"].astype(str)
        index = "slide-fov-cell_ID"
        obs[index] = obs["slide-fov"] + "-" + obs["cell_ID"].astype(str)
        obs = _add_metadata(
            obs, files["metadata"], slide, metadata_df_columns, data_dir, **kwargs
        )
        obs = _add_samples_df_metadata(obs, samples_df, slide, samples_df_columns)
        obs.set_index(index, inplace=True)

        # convert to sparse adata object
        with warnings.catch_warnings():
            # warning: Transforming to str index.
            warnings.simplefilter("ignore", ad.ImplicitModificationWarning)
            adata = ad.AnnData(
                X=_add_x(fname, columns, verbose=verbose, **kwargs),
                obs=obs,
                var=pd.DataFrame(index=columns),
            )

        # remove FOVs that were not assigned to any sample
        fovs_before = set(adata.obs["fov"])
        n_cells_before = len(adata.obs)
        adata = adata[adata.obs["sample"].isin(samples_df["sample"]), :].copy()
        fovs_after = set(adata.obs["fov"])
        n_cells_after = len(adata.obs)
        n_cells_dropped = n_cells_before - n_cells_after
        if verbose and n_cells_dropped > 0:
            fovs_dropped = fovs_before - fovs_after
            n_fovs_dropped = len(fovs_dropped)
            print(
                f"Removed {n_cells_dropped:_} cells from {n_fovs_dropped:_} FOVs "
                "that are missing from samples_df['fovs']."
            )
            print(f"Missing FOVs: {sorted(fovs_dropped)}.")

        # write slide specific adata object to file
        adata.write_h5ad(adata_file_slide)

        # free memory
        del adata, obs, columns
        if verbose:
            print(f"Slide {slide}/{len(slides)} done!\n")

    # concatenate all adatas
    if len(adatas) == 0:
        raise ValueError("No data to concatenate!")
    elif len(adatas) == 1:
        os.rename(adatas[0], adata_file)
    else:
        ad.experimental.concat_on_disk(adatas, adata_file)
        for f in adatas:
            os.remove(f)

    return adata_file


def _add_x(fname, columns, chunksize=50_000, verbose=True, **kwargs):
    sparse_blocks = []
    i = 1
    for chunk in pd.read_csv(
        fname, dtype=int, usecols=columns, chunksize=chunksize, **kwargs
    ):
        # Convert to sparse matrix
        sparse_blocks.append(sp.csr_matrix(chunk.values))
        if verbose:
            print(f"{round(i * chunksize):_} rows parsed")
            i += 1

    # Stack all chunks vertically
    X = sp.vstack(sparse_blocks, format="csr")
    if verbose:
        print(f"{round(i * chunksize):_} rows parsed")
    return X


def _add_samples_df_metadata(obs, samples_df, slide, columns=None):
    if columns is None:
        columns = samples_df.columns.to_list()
    if "sample" not in columns:
        columns.append("sample")
    if "slide" in columns:
        columns.remove("slide")

    sdf = samples_df.loc[samples_df["slide"] == slide]
    fov2sample = _fovranges_to_dict(sdf)
    obs["sample"] = obs["fov"].replace(fov2sample)

    # merge samples_df to obs
    obs = obs.merge(sdf[columns], on="sample", how="left")
    return obs


def _fovranges_to_dict(df):
    fov2sample = {}
    for _, row in df.iterrows():
        sample = row["sample"]
        fovs = _parse_ranges(row["fovs"])
        for fov in fovs:
            fov2sample[fov] = sample
    return fov2sample


def _parse_ranges(s):
    out = []
    ranges = s.replace(" ", "").split(",")
    for item in ranges:
        if len(item) == 0:
            continue
        try:
            if "-" not in item:
                out.append(int(item))
            else:
                i1, i2 = item.split("-")
                start, end = sorted([int(i1), int(i2)])
                out.extend(range(start, end + 1))
        except (TypeError, ValueError) as e:
            print(
                f"FOV (range) '{item}' not recognized! FOVs must be written as ranges "
                f"(e.g. 1-3) or integers, and are comma separated (e.g.: 1-3,6)."
            )
            raise e
    return out


def _add_metadata(obs, file_md, slide, columns=None, data_dir: str = None, **kwargs):
    if data_dir is None:
        data_dir = ""
    if columns:
        for col in config.get("metadata_md_columns"):
            if col not in columns:
                columns.append(col)
    md = pd.read_csv(os.path.join(data_dir, file_md), usecols=columns, **kwargs)

    # add a unique identifier
    index = "slide-fov-cell_ID"
    md[index] = f"{slide}-" + md["fov"].astype(str) + "-" + md["cell_ID"].astype(str)

    # if both files are sorted, we can concatenate blindly
    assert len(md[index]) == len(
        obs[index]
    ), "exprMat and metadata files have different lengths!"
    assert (md[index] == obs[index]).all(), "exprMat and metadata files are not sorted!"
    obs = pd.concat(
        [obs, md.drop(columns=config.get("metadata_md_columns") + [index])], axis=1
    )
    return obs
