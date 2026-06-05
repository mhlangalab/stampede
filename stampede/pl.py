"""plotting functions"""

from ._dim_red import plot_dim_red as dim_red
from ._dim_red import plot_scree as scree
from ._qc import plot_avg_per_pixel as avg_per_pixel
from ._qc import plot_column_distribution as column_distribution
from ._qc import plot_correlations as correlations
from ._qc import plot_ncell_per_condition as ncell_per_condition
from ._qc import plot_noise_threshold as noise_threshold
from ._qc import plot_slide_qc as slide_qc
from ._qc import plot_value_distribution as value_distribution
from ._qc import plot_violin as violin
from ._tools.geosketch import plot_sketch as sketch
from ._tools.pydeseq2 import plot_pydeseq2_volcano as pydeseq2_volcano
from ._tools.statsmodels import (
    plot_paired_binomial_glm_volcano as paired_binomial_glm_volcano,
)

__all__ = [
    "slide_qc",
    "correlations",
    "avg_per_pixel",
    "violin",
    "ncell_per_condition",
    "value_distribution",
    "column_distribution",
    "noise_threshold",
    "scree",
    "dim_red",
    "sketch",
    "pydeseq2_volcano",
    "paired_binomial_glm_volcano",
]
