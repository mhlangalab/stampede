import threading
import time
import warnings

import anndata as ad
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.path import Path
from matplotlib.widgets import LassoSelector


def lasso_clustering(
    adata: ad.AnnData,
    key_added: str = "lasso",
    color: str = None,
    basis: str = "umap",
    highlight_n: int = 1000,
    highlight_frac: float = 0.50,
    n_categories: int = 10,
    density_bins: int = None,
    fig_kwargs: dict = None,
    embedding_kwargs: dict = None,
    highlight_kwargs: dict = None,
    density_kwargs: dict = None,
    lasso_kwargs: dict = None,
):
    """
    Interactive lasso selection on a scanpy embedding plot.

    Left panel: static plot with either cell density, or colored by `color` (a gene name or column in adata.obs).
    Right panel: live view of `adata.obs[key_added]`, redrawn every time a selection is labeled.

    Draw a lasso on the left panel, then click a numbered button to label the selected cells.
    Click 'Close' to exit interactive mode.

    Args:
        adata: adata object
        key_added: column in adata.obs that will be added
        color: a gene name or column in adata.obs to color the UMAP with. If None, a cell density plot is shown instead
        basis: Name of the `obsm` basis to use
        highlight_n: minimum number of cells to highlight
        highlight_frac: maximum fraction of cells to highlight
        n_categories: maximum number of categories in the adata.obs columns
        density_bins: number of bins in the density plot
        fig_kwargs: kwargs passed to plt.Figure
        embedding_kwargs: kwargs passed to sc.pl.embedding (UMAP plots)
        highlight_kwargs: kwargs passed to plt.scatter (cell highlights)
        density_kwargs: kwargs passed to sns.histplot (density plot)
        lasso_kwargs: kwargs passed to mpl.widgets.LassoSelector

    Returns:
        Nothing, updates adata.obs
    """
    density_plot = False
    if color is None:
        density_plot = True
        color = "Cell density"
    if color and not isinstance(color, str):
        raise TypeError(f"{color=} must be a string (one color)!")
    if fig_kwargs is None:
        # increase the default figure size
        x, y = plt.rcParams["figure.figsize"]
        fig_kwargs = {"figsize": (x * 1.70, y * 1.0)}
    if embedding_kwargs is None:
        embedding_kwargs = {}
    if highlight_kwargs is None:
        highlight_kwargs = {
            "facecolors": "none",
            "edgecolors": "#e41a1c",  # red circles
        }
    if density_kwargs is None:
        density_kwargs = {
            "color": "#4daf4a",  # green squares
        }
    if lasso_kwargs is None:
        lasso_kwargs = {}  # blue lines

    categories = list(range(n_categories))
    adata.obs[key_added] = pd.Categorical([None] * adata.n_obs, categories=categories)

    if basis not in adata.obsm:
        basis = f"X_{basis}"
    if basis not in adata.obsm:
        raise KeyError(
            f"'{basis}' not found in adata.obsm. Available: {list(adata.obsm.keys())}"
        )
    coords = adata.obsm[basis][:, :2]

    # plot something with the previous backend
    #  this is needed to 'lock-in' that backend if nothing
    #  has been plotted yet.
    #  if not set, and this function is called,
    #  non-interactive plots stop working.
    fig, ax = plt.subplots()
    plt.close("all")

    # this import sets an interactive backend
    prev_backend = matplotlib.get_backend()
    # optional dependencies
    import ipywidgets as widgets  # noqa
    from ipympl.backend_nbagg import FigureCanvas, FigureManager  # noqa
    from IPython.display import display  # noqa

    fig = plt.Figure(**fig_kwargs)
    canvas = FigureCanvas(fig)
    manager = FigureManager(canvas, 1)
    canvas.toolbar_visible = False  # buttons unrelated to the lasso tool
    canvas.header_visible = False  # "figure 1"
    canvas.footer_visible = False  # x & y coordinates
    fig.suptitle(
        "In the left figure, draw a lasso around cells to select them. "
        + "Use the buttons below the figure to assign the selected cells to a cluster.\n",
        y=1,
        fontsize=10,
    )
    ax_left = fig.add_subplot(1, 2, 1)
    ax_right = fig.add_subplot(1, 2, 2)  # , sharex=ax_left, sharey=ax_left)

    # track all interactive elements
    state = {
        "indices": np.array([], dtype=int),
        "highlight_left": None,
        "highlight_right": None,
        "s": None,
        "lw": None,
        "idx_right": [],
    }

    # draw the left plot (once)
    if density_plot:
        if density_bins is None:
            density_bins = np.clip(
                round(len(adata.obs) ** 0.5),
                10,
                100,
            )
        sns.histplot(
            x=coords[:, 0],
            y=coords[:, 1],
            bins=density_bins,
            ax=ax_left,
            zorder=1,
            **density_kwargs,
        )
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Adding colorbar to a different Figure.*",
                category=UserWarning,
            )
            sc.pl.embedding(
                adata,
                basis=basis,
                color=color,
                show=False,
                ax=ax_left,
                zorder=0,
                **embedding_kwargs,
            )
    ax_left.set_title(f"{color} | Select cells here")

    # draw the right plot (once)
    # plot a subset of the cells in the right embedding plot (for speed)
    n = np.clip(
        len(adata.obs),
        a_min=min(1000, len(adata.obs)),
        a_max=10_000,
    )
    state["idx_right"] = np.random.choice(
        adata.obs.index,
        size=n,
        replace=False,
    )
    if color in adata.obs.columns:
        bdata = ad.AnnData(
            obs=adata.obs[[color, key_added]].copy(),
            obsm={basis: adata.obsm[basis]},
        )
    elif color in adata.var.columns:
        bdata = ad.AnnData(
            obs=adata.obs[[key_added]].copy(),
            var=adata.var[[color]].copy(),
            obsm={basis: adata.obsm[basis]},
        )
    else:
        bdata = ad.AnnData(
            obs=adata.obs[[key_added]].copy(),
            obsm={basis: adata.obsm[basis]},
        )
    bdata = bdata[state["idx_right"], :].copy()
    bdata.obs[key_added] = bdata.obs[key_added].cat.set_categories(categories)
    sc.pl.embedding(
        bdata,
        basis=basis,
        color=key_added,
        show=False,
        ax=ax_right,
        zorder=0,
        **embedding_kwargs,
    )
    # for some reason, this sizes variable is larger than the actual point sizes?
    state["s"] = ax_right.collections[0].get_sizes() / 3
    state["lw"] = ax_right.collections[0].get_linewidths()
    ax_right.set_title(key_added)

    def cache_ax(ax):
        """replace the plot on the given ax with an image"""
        renderer = canvas.get_renderer()
        bbox = ax.get_window_extent(renderer=renderer)
        buf = np.asarray(renderer.buffer_rgba())
        x0, y0, x1, y1 = int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)
        img = buf[
            round(fig.bbox.height - y1) : round(fig.bbox.height - y0), x0:x1
        ]  # flip y

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for artist in list(ax.collections) + list(ax.patches):
            artist.remove()
        ax.imshow(img, extent=[*xlim, *ylim], aspect="auto", zorder=0)

    # store the figures as images to improve the lasso draw speed
    canvas.draw()
    time.sleep(1)  # needed to finish the (blocking) draw call. don't ask.
    cache_ax(ax_left)
    cache_ax(ax_right)

    def remove_highlight(side):
        """remove cell highlights from either left or right side"""
        if state[f"highlight_{side}"] is not None:
            state[f"highlight_{side}"].remove()
            state[f"highlight_{side}"] = None

    def redraw_right():
        """draw the right plot"""
        remove_highlight("right")
        ax_right.clear()
        sc.pl.embedding(
            bdata,
            basis=basis,
            color=key_added,
            show=False,
            ax=ax_right,
            zorder=0,
            **embedding_kwargs,
        )
        ax_right.set_title(key_added)
        canvas.draw()
        cache_ax(ax_right)

    def on_select(verts):
        """track the selected cells & highlight a subset"""
        mask = Path(verts).contains_points(coords)
        state["indices"] = np.nonzero(mask)[0]

        # remove old highlights
        remove_highlight("left")
        remove_highlight("right")
        # draw new highlights
        if len(state["indices"]) > 0:
            # subset the highlights to draw
            n = max(round(highlight_frac * len(state["indices"])), highlight_n)
            if len(state["indices"]) > n:
                indices = np.random.choice(
                    state["indices"],
                    size=n,
                    replace=False,
                )
            else:
                indices = state["indices"]
            x = coords[indices, 0]
            y = coords[indices, 1]
            state["highlight_left"] = ax_left.scatter(  # noqa
                x,
                y,
                s=state["s"],
                linewidths=state["lw"],
                zorder=1,
                **highlight_kwargs,
            )
            state["highlight_right"] = ax_right.scatter(  # noqa
                x,
                y,
                s=state["s"],
                linewidths=state["lw"],
                zorder=1,
                **highlight_kwargs,
            )
        ax_left.set_title(f"{color} | {len(state['indices'])} cells selected")
        canvas.draw()

    fig._lasso = LassoSelector(ax_left, on_select, **lasso_kwargs)
    label_buttons = [
        widgets.Button(description=str(d), layout=widgets.Layout(width="35px"))
        for d in categories
    ]
    close_button = widgets.Button(
        description="Close", button_style="danger", layout=widgets.Layout(width="70px")
    )
    button_row = widgets.HBox(label_buttons + [close_button])
    container = widgets.VBox([canvas, button_row])
    display(container)

    def assign_cells_to_cluster(digit):
        """assign the currently highlighted cells to the selected cluster"""

        def handler(_):
            idx = state["indices"]
            adata.obs.loc[adata.obs_names[idx], key_added] = digit
            bdata.obs[key_added] = adata.obs.loc[state["idx_right"]][key_added]
            ax_left.set_title(f"{color} | labeled {len(idx)} cells cluster {digit}")
            redraw_right()

        return handler

    for button, d in zip(label_buttons, categories):
        button.on_click(assign_cells_to_cluster(d))

    def on_close(_):
        """clean up the interactive plots & unblock the GIL"""
        adata.obs[key_added] = adata.obs[key_added].cat.remove_unused_categories()
        manager.destroy()
        plt.close(fig)
        container.close()

        matplotlib.use(prev_backend, force=True)
        done.set()

    close_button.on_click(on_close)

    # block the notebook until the figure is closed
    done = threading.Event()
    done.wait()

    # plot the result
    if len(adata.obs[key_added].cat.categories) > 0:
        sc.pl.embedding(
            adata,
            basis=basis,
            color=key_added,
            **embedding_kwargs,
        )
    else:
        print("No clusters annotated (if no plot was shown, rerun the function)")
