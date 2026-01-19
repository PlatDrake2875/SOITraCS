"""
Evaluation package for SOITraCS experiments.

Provides tools for running reproducible experiments, statistical analysis,
and comparing algorithm performance.

Modules:
    experiment: ExperimentConfig, ExperimentResult, ExperimentRunner
    analysis: Statistical analysis and export functions
    self_organization: Metrics for emergent behavior
"""

from .experiment import (
    ExperimentConfig,
    ExperimentResult,
    ExperimentRunner,
    load_experiment_config,
)
from .analysis import (
    compute_statistics,
    paired_comparison,
    export_latex_table,
    export_csv_summary,
    format_comparison_results,
    compute_improvement_summary,

    compute_additive_contribution,
    compute_interaction_effect,
    compute_subtractive_importance,
)
from .self_organization import (
    SelfOrganizationAnalyzer,
    analyze_experiment_self_organization,
)

__all__ = [

    "ExperimentConfig",
    "ExperimentResult",
    "ExperimentRunner",
    "load_experiment_config",

    "compute_statistics",
    "paired_comparison",
    "export_latex_table",
    "export_csv_summary",
    "format_comparison_results",
    "compute_improvement_summary",

    "compute_additive_contribution",
    "compute_interaction_effect",
    "compute_subtractive_importance",

    "SelfOrganizationAnalyzer",
    "analyze_experiment_self_organization",
]
