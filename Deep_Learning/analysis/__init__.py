"""Dataset analysis for the paper: class-imbalance statistics, figures and tables.

Run from the project root:
    python -m Deep_Learning.analysis            # writes to reports/figures/imbalance/
"""
from .imbalance_report import (imbalance_metrics, plot_class_distribution, plot_class_weights,
                               plot_resampling_effect, plot_split_distribution, run)

__all__ = ["imbalance_metrics", "plot_class_distribution", "plot_class_weights",
           "plot_resampling_effect", "plot_split_distribution", "run"]
