from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def paired_binomial_glm(
    df: pd.DataFrame,
    adata: ad.AnnData,
    samples_column: str,
    test_condition: str,
    reference_condition: str,
    condition_column: str = "condition",
    covariate_columns: str = None,
    random_state: int = 42,
) -> pd.DataFrame | None:
    """
    Runs paired sample-level binomial GLM:
        gene_detection_rate ~ condition + covariate(s)

    Args:
        df: dataframe with detection rates per gene per sample
        adata: the adata from which the detection rates were obtained
        samples_column: the column in adata.obs from which the detection rate df
         column names were obtained
        test_condition: the condition to compare (e.g., "treated")
        reference_condition: the baseline condition (e.g., "control")
        condition_column: column with the condition
        covariate_columns: column(s) with covariates (e.g. "donor")
        random_state: random seed value

    Returns:
        per-gene results including beta, odds_ratio, pval, padj
    """
    # optional dependency
    import statsmodels.api as sm  # noqa
    import statsmodels.formula.api as smf  # noqa
    from statsmodels.stats.multitest import multipletests  # noqa
    from statsmodels.tools.sm_exceptions import PerfectSeparationWarning  # noqa

    df = df.stack().reset_index()
    df.columns = ["gene", samples_column, "detection_rate"]
    df[samples_column] = df[samples_column].astype(str)

    # add the conditions per sample
    sample2condition = (
        adata.obs[[samples_column, condition_column]]
        .set_index(samples_column)[condition_column]
        .astype(str)
        .to_dict()
    )
    df[condition_column] = df[samples_column].map(sample2condition)

    # drop all samples not in adata
    df.dropna(subset=[condition_column], inplace=True)

    # add the number of cells per sample
    sample2ncells = adata.obs[samples_column].value_counts().astype(str).to_dict()
    df["ncells"] = df[samples_column].replace(sample2ncells).astype(int)

    # add all covariate columns
    if covariate_columns is None:
        covariate_columns = []
    elif isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]
    for col in covariate_columns:
        sample2covariate = (
            adata.obs[[samples_column, col]]
            .set_index(samples_column)[col]
            .astype(str)
            .to_dict()
        )
        df[col] = df[samples_column].map(sample2covariate)

    # convert all metadata columns to categorical
    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].astype("category")

    unique_conditions = df[condition_column].unique()
    if reference_condition not in unique_conditions:
        raise ValueError(f"{reference_condition=} not found in dataframe")
    if test_condition not in unique_conditions:
        raise ValueError(f"{test_condition=} not found in dataframe")

    # re-level the condition column so the reference condition is the baseline
    df = df[df[condition_column].isin(unique_conditions)]
    df[condition_column] = pd.Categorical(
        df[condition_column],
        categories=[reference_condition, test_condition],
        ordered=True,
    )
    df.dropna(subset=condition_column, inplace=True)

    design_formula = "detection_rate ~ " + " + ".join(
        [condition_column] + covariate_columns
    )

    def fit_one_gene(gene_df):
        perfect_sep = False
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always", PerfectSeparationWarning)

                model = smf.glm(
                    formula=design_formula,
                    data=gene_df,
                    family=sm.families.Binomial(),
                    var_weights=gene_df["ncells"],
                )
                result = model.fit()

                for warn in w:
                    if issubclass(warn.category, PerfectSeparationWarning):
                        perfect_sep = True
                        break

            # Coefficient name will be condition_column[T.condition_of_interest]
            coef_name = f"{condition_column}[T.{test_condition}]"

            beta = result.params.get(coef_name, np.nan)
            se = result.bse.get(coef_name, np.nan)
            pval = result.pvalues.get(coef_name, np.nan)
            odds_ratio = np.exp(beta) if pd.notnull(beta) else np.nan

            return pd.Series(
                {
                    "beta": beta,
                    "se": se,
                    "odds_ratio": odds_ratio,
                    "pval": pval,
                    "perfect_separation": perfect_sep,
                    "error": None,
                }
            )

        except Exception as exc:
            return pd.Series(
                {
                    "beta": np.nan,
                    "se": np.nan,
                    "odds_ratio": np.nan,
                    "pval": np.nan,
                    "perfect_separation": np.nan,
                    "error": str(exc),
                }
            )

    # run across genes
    np.random.seed(random_state)
    results = (
        df.groupby("gene", group_keys=False, observed=False)
        .apply(fit_one_gene, include_groups=False)
        .reset_index()
    )

    # drop failed fits
    results.dropna(subset=["pval"], inplace=True)
    if len(results) == 0:
        return None

    results["padj"] = multipletests(results["pval"], method="fdr_bh")[1]
    results["-log10(padj)"] = -np.log10(np.clip(results["padj"], 1e-9, None))
    results["log2(odds_ratio)"] = np.log2(results["odds_ratio"])
    results.sort_values("odds_ratio", inplace=True)

    n_perfect_sep = results["perfect_separation"].sum()
    if n_perfect_sep > 0:
        warnings.warn(
            f"Perfect separation detected in {int(n_perfect_sep)} genes. "
            "Parameter estimates may be unstable for these genes. "
            "Check the 'perfect_separation' column in the results.",
            RuntimeWarning,
        )

    return results


def plot_paired_binomial_glm_volcano(
    df: pd.DataFrame,
    drop_perfect_separation: bool = True,
    pval_thresh: float = 0.05,
    or_thresh: float = 0.75,
    to_label: int | list = 5,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[Figure, Axes]:
    """
    Generate a volcano plot from the detection_rates results dataframe.

    Args:
        df: a dataframe
        drop_perfect_separation: whether to drop the genes with perfect separations
        pval_thresh: threshold pvalue_column for genes to be significant
        or_thresh: threshold for the log2 odds ratios to be considered significant
        to_label: the number of top genes (down and up each) to be labeled
        subplot_kwargs: kwargs passed to plt.subplots
        plot_kwargs: kwargs passed to the main plotting function
        text_kwargs: kwargs passed to ax.text

    Returns:
        matplotlib figure and axis object
    """
    or_thresh = abs(or_thresh)
    if subplot_kwargs is None:
        subplot_kwargs = {}
    if plot_kwargs is None:
        plot_kwargs = {}
    if text_kwargs is None:
        text_kwargs = {}
    gene_column = "gene"
    or_column = "odds_ratio"
    pvalue_column = "padj"
    separation_column = "perfect_separation"

    df = df.dropna(subset=[pvalue_column, or_column, separation_column])
    if drop_perfect_separation:
        df = df.loc[~df[separation_column]]

    if df[pvalue_column].min() == 0:
        df[pvalue_column][df[pvalue_column] == 0] = 1e-9
    df["-log10(padj)"] = -np.log10(df[pvalue_column])
    df["log2(odds_ratio)"] = np.log2(df[or_column])

    symbol2dir = {gene: "no change" for gene in df[gene_column]}
    up = (
        df[gene_column]
        .loc[(df[pvalue_column] < pval_thresh) & (df["log2(odds_ratio)"] >= or_thresh)]
        .to_list()
    )
    for gene in up:
        symbol2dir[gene] = "up"
    down = (
        df[gene_column]
        .loc[(df[pvalue_column] < pval_thresh) & (df["log2(odds_ratio)"] <= -or_thresh)]
        .to_list()
    )
    for gene in down:
        symbol2dir[gene] = "down"
    df["change"] = df[gene_column].map(symbol2dir)

    dir2color = {
        "up": "indianred",
        "down": "cornflowerblue",
        "no change": "gainsboro",
    }

    fig, ax = plt.subplots(**subplot_kwargs)
    sns.scatterplot(
        df,
        x="log2(odds_ratio)",
        y="-log10(padj)",
        hue="change",
        palette=dir2color,
        ax=ax,
        **plot_kwargs,
    )
    ax.axhline(-np.log10(pval_thresh), color="black", linestyle="--", linewidth=0.5)
    ax.axvline(or_thresh, color="black", linestyle="--", linewidth=0.5)
    ax.axvline(-or_thresh, color="black", linestyle="--", linewidth=0.5)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    df["sorter"] = (
        df["-log10(padj)"] * df["log2(odds_ratio)"]
    )  # make a column to pick top genes
    df = df.sort_values(by="sorter", ascending=False)
    top_up = df[gene_column].head(to_label).to_list()
    top_down = df[gene_column].tail(to_label).to_list()

    texts = []
    for gene in top_up + top_down:
        x, y = df.loc[df[gene_column] == gene][
            ["log2(odds_ratio)", "-log10(padj)"]
        ].iloc[0, 0:2]
        txt = ax.text(
            x=x,
            y=y,
            s=gene,
            **text_kwargs,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
        texts.append(txt)
    _ = adjust_text(
        texts,
        arrowprops=dict(arrowstyle="-", color="gray", zorder=5),
        ax=ax,
    )

    return fig, ax
