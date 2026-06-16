from __future__ import annotations

import warnings

import anndata as ad
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import patheffects
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def paired_binomial_glm(
    df: pd.DataFrame,
    adata: ad.AnnData,
    column: str,
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
        column: the column in adata.obs from which the detection rate df
         column names were obtained
        test_condition: the condition to compare (e.g., "treated")
        reference_condition: the baseline condition (e.g., "control")
        condition_column: column with the conditions
        covariate_columns: column(s) with covariates (e.g. "batch")
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
    df.columns = ["gene", column, "detection_rate"]
    df[column] = df[column].astype(str)

    # add the condition per sample
    sample2condition = (
        adata.obs[[column, condition_column]]
        .set_index(column)[condition_column]
        .astype(str)
        .to_dict()
    )
    df[condition_column] = df[column].map(sample2condition)

    # sanity check
    unique_conditions = df[condition_column].unique()
    if reference_condition not in unique_conditions:
        raise ValueError(f"{reference_condition=} not found in dataframe:\n{df}")
    if test_condition not in unique_conditions:
        raise ValueError(f"{test_condition=} not found in dataframe:\n{df}")

    # add the number of cells per sample
    sample2ncells = adata.obs[column].value_counts().astype(str).to_dict()
    df["ncells"] = df[column].replace(sample2ncells).astype(int)

    # add all covariate columns
    if covariate_columns is None:
        covariate_columns = []
    elif isinstance(covariate_columns, str):
        covariate_columns = [covariate_columns]
    for col in covariate_columns:
        sample2covariate = (
            adata.obs[[column, col]].set_index(column)[col].astype(str).to_dict()
        )
        df[col] = df[column].map(sample2covariate)

    # convert all metadata columns to categorical
    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].astype("category")

    # drop all samples/conditions not in the contrast
    #  (if there are no covariates to take into account)
    if len(covariate_columns) == 0:
        df = df[df[condition_column].isin([reference_condition, test_condition])]

    # re-level the condition column so the reference condition is the baseline
    df[condition_column] = pd.Categorical(
        df[condition_column],
        categories=[reference_condition, test_condition],
        ordered=True,
    )

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
    results.set_index("gene", inplace=True)

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
    symbol_column: str = "index",
    or_column: str = "odds_ratio",
    pvalue_column: str = "padj",
    separation_column: str = "perfect_separation",
    pval_thresh: float = 0.05,
    l2or_thresh: float = 0.75,
    to_label: int | list | None = 5,
    drop_perfect_separation: bool = True,
    subplot_kwargs: dict = None,
    plot_kwargs: dict = None,
    text_kwargs: dict = None,
) -> tuple[Figure, Axes]:
    """
    Generate a volcano plot from the detection_rates results dataframe.

    Args:
        df: a dataframe
        symbol_column: column name of gene IDs to use
        or_column: column name of odds ratios
        pvalue_column: column name of the adjusted p values to be converted to -log10 p-values
        separation_column: boolean column denoting perfect separations
        pval_thresh: threshold pvalue_column for genes to be significant
        l2or_thresh: threshold for the log2 odds ratios to be considered significant
        to_label: the number of top genes (down and up each) to be labeled
        drop_perfect_separation: whether to drop the genes with perfect separations
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
    pval_thresh = -np.log10(pval_thresh)

    if symbol_column == "index":
        if df.index.name:
            symbol_column = df.index.name
        df = df.reset_index(drop=False)

    df = df.dropna(subset=[pvalue_column, or_column, separation_column])
    if drop_perfect_separation:
        df = df.loc[~df[separation_column]]
    min_value = min(1e-9, df[df[pvalue_column] > 0][pvalue_column].min() / 10)
    df["-log10(padj)"] = -np.log10(np.clip(df[pvalue_column], min_value, None))
    df["log2(odds_ratio)"] = np.log2(df[or_column])

    def map_genes(row):
        l2or, log10p = row
        if log10p <= pval_thresh:
            if abs(l2or) <= l2or_thresh:
                return "NS"
            else:
                return "Log2OR"
        else:
            if abs(l2or) <= l2or_thresh:
                return "p-value"
            else:
                return "p-value & Log2OR"

    df["cat"] = df[["log2(odds_ratio)", "-log10(padj)"]].apply(map_genes, axis=1)
    cat2color = {
        "NS": "lightgrey",
        "p-value": "tab:blue",
        "Log2OR": "tab:green",
        "p-value & Log2OR": "tab:red",
    }
    df["color"] = df["cat"].map(cat2color)

    fig, ax = plt.subplots(**subplot_kwargs)
    df1 = df[df[pvalue_column] > min_value]
    df2 = df[df[pvalue_column] <= min_value]
    ax.scatter(
        x=df1["log2(odds_ratio)"],
        y=df1["-log10(padj)"],
        c=df1["color"],
        marker="o",
        alpha=alpha,
        **plot_kwargs,
    )
    ax.scatter(
        x=df2["log2(odds_ratio)"],
        y=df2["-log10(padj)"],
        c=df2["color"],
        marker="^",
        alpha=alpha,
        **plot_kwargs,
    )
    ax.axhline(pval_thresh, zorder=-1, c="k", lw=1.5, ls="--", alpha=alpha)
    ax.axvline(l2or_thresh, zorder=-1, c="k", lw=1.5, ls="--", alpha=alpha)
    ax.axvline(-l2or_thresh, zorder=-1, c="k", lw=1.5, ls="--", alpha=alpha)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("$log_{2}$ odds ratio")
    ax.set_ylabel(f"-$log_{{10}}$ adjusted p-value")

    # legend with color descriptions
    legend_elements = []
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
        df["sorter"] = df["-log10(padj)"] * df["log2(odds_ratio)"]
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
                x=label_df.iloc[i]["log2(odds_ratio)"],
                y=label_df.iloc[i]["-log10(padj)"],
                s=label_df.iloc[i][symbol_column],
                **text_kwargs,
            )
            txt.set_path_effects([patheffects.withStroke(linewidth=3, foreground="w")])
            texts.append(txt)
        adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", zorder=5), ax=ax)

    return fig, ax
