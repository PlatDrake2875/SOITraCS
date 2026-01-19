"""
Statistical analysis functions for experiment results.

Provides tools for computing summary statistics, paired comparisons,
and exporting results in various formats (CSV, LaTeX).
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy import stats

def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean, std, 95% CI for each metric grouped by experiment.

    Args:
        df: DataFrame with columns: experiment, run, seed, and metric columns

    Returns:
        DataFrame with aggregated statistics per experiment
    """

    metadata_cols = {"experiment", "run", "seed", "description", "algorithms"}
    metric_cols = [col for col in df.columns if col not in metadata_cols]

    results = []

    for exp_name, group in df.groupby("experiment"):
        row = {"experiment": exp_name, "n_runs": len(group)}

        for metric in metric_cols:
            values = group[metric].dropna()

            if len(values) > 0:
                mean = values.mean()
                std = values.std()
                n = len(values)

                if n > 1:
                    ci = stats.t.ppf(0.975, n - 1) * std / np.sqrt(n)
                else:
                    ci = 0.0

                row[f"{metric}_mean"] = mean
                row[f"{metric}_std"] = std
                row[f"{metric}_ci95"] = ci
                row[f"{metric}_min"] = values.min()
                row[f"{metric}_max"] = values.max()

        results.append(row)

    return pd.DataFrame(results)

def paired_comparison(
    baseline_df: pd.DataFrame,
    treatment_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Welch's t-test between baseline and treatment conditions.

    Args:
        baseline_df: DataFrame with baseline experiment results
        treatment_df: DataFrame with treatment experiment results
        metrics: List of metrics to compare (auto-detected if None)

    Returns:
        Dictionary with p-values and effect sizes for each metric
    """
    if metrics is None:

        metadata_cols = {"experiment", "run", "seed"}
        metrics = [
            col for col in baseline_df.columns
            if col not in metadata_cols and not col.endswith(("_mean", "_std", "_ci95"))
        ]

    results = {}

    for metric in metrics:
        if metric not in baseline_df.columns or metric not in treatment_df.columns:
            continue

        baseline_vals = baseline_df[metric].dropna()
        treatment_vals = treatment_df[metric].dropna()

        if len(baseline_vals) < 2 or len(treatment_vals) < 2:
            continue

        t_stat, p_value = stats.ttest_ind(
            baseline_vals, treatment_vals, equal_var=False
        )

        pooled_std = np.sqrt(
            (baseline_vals.std() ** 2 + treatment_vals.std() ** 2) / 2
        )
        if pooled_std > 0:
            cohens_d = (treatment_vals.mean() - baseline_vals.mean()) / pooled_std
        else:
            cohens_d = 0.0

        baseline_mean = baseline_vals.mean()
        if baseline_mean != 0:
            pct_change = (
                (treatment_vals.mean() - baseline_mean) / abs(baseline_mean) * 100
            )
        else:
            pct_change = 0.0

        results[metric] = {
            "baseline_mean": float(baseline_vals.mean()),
            "baseline_std": float(baseline_vals.std()),
            "treatment_mean": float(treatment_vals.mean()),
            "treatment_std": float(treatment_vals.std()),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
            "pct_change": float(pct_change),
            "significant": p_value < 0.05,
        }

    return results

def export_latex_table(
    summary_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    caption: str = "Algorithm Comparison Results",
    label: str = "tab:comparison",
) -> str:
    """
    Generate LaTeX table from summary statistics.

    Args:
        summary_df: DataFrame from compute_statistics()
        metrics: Metrics to include (auto-detected if None)
        caption: Table caption
        label: LaTeX label for referencing

    Returns:
        LaTeX table string
    """
    if metrics is None:

        metrics = []
        for col in summary_df.columns:
            if col.endswith("_mean"):
                base_metric = col[:-5]
                if base_metric not in metrics:
                    metrics.append(base_metric)

    n_cols = len(metrics) + 1
    col_spec = "l" + "r" * len(metrics)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{ {caption}} ",
        f"\\label{ {label}} ",
        f"\\begin{ tabular} { {col_spec}} ",
        "\\toprule",
    ]

    header_parts = ["Experiment"]
    for metric in metrics:

        display_name = metric.replace("_", " ").title()
        header_parts.append(display_name)
    lines.append(" & ".join(header_parts) + " \\\\")
    lines.append("\\midrule")

    for _, row in summary_df.iterrows():
        parts = [str(row["experiment"])]

        for metric in metrics:
            mean_col = f"{metric}_mean"
            std_col = f"{metric}_std"

            if mean_col in row and std_col in row:
                mean_val = row[mean_col]
                std_val = row[std_col]
                parts.append(f"${mean_val:.2f} \\pm {std_val:.2f}$")
            else:
                parts.append("--")

        lines.append(" & ".join(parts) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(lines)

def export_csv_summary(summary_df: pd.DataFrame, path: str) -> None:
    """
    Export summary statistics to CSV file.

    Args:
        summary_df: DataFrame from compute_statistics()
        path: Output file path
    """
    summary_df.to_csv(path, index=False)

def format_comparison_results(
    comparison: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """
    Format paired comparison results as a DataFrame.

    Args:
        comparison: Results from paired_comparison()

    Returns:
        Formatted DataFrame
    """
    rows = []
    for metric, stats_dict in comparison.items():
        row = {"metric": metric, **stats_dict}
        rows.append(row)

    df = pd.DataFrame(rows)

    if "p_value" in df.columns:
        df["significance"] = df["p_value"].apply(
            lambda p: "***" if p < 0.001 else (
                "**" if p < 0.01 else ("*" if p < 0.05 else "")
            )
        )

    return df

def compute_improvement_summary(
    raw_df: pd.DataFrame,
    baseline_name: str,
    treatment_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute improvement percentages vs baseline for all treatments.

    Args:
        raw_df: Raw experiment data with all runs
        baseline_name: Name of baseline experiment
        treatment_names: Names of treatment experiments (all others if None)

    Returns:
        DataFrame with improvement percentages and significance
    """
    baseline_df = raw_df[raw_df["experiment"] == baseline_name]

    if treatment_names is None:
        treatment_names = [
            exp for exp in raw_df["experiment"].unique()
            if exp != baseline_name
        ]

    results = []

    for treatment_name in treatment_names:
        treatment_df = raw_df[raw_df["experiment"] == treatment_name]
        comparison = paired_comparison(baseline_df, treatment_df)

        row = {"treatment": treatment_name}
        for metric, stats_dict in comparison.items():
            row[f"{metric}_pct_change"] = stats_dict["pct_change"]
            row[f"{metric}_p_value"] = stats_dict["p_value"]
            row[f"{metric}_significant"] = stats_dict["significant"]

        results.append(row)

    return pd.DataFrame(results)

def compute_additive_contribution(
    raw_df: pd.DataFrame,
    baseline_name: str,
    treatment_name: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the additive contribution of an algorithm vs baseline.

    contribution(A) = metric(baseline + A) - metric(baseline)

    This measures the marginal improvement when adding a single algorithm
    to the baseline configuration.

    Args:
        raw_df: Raw experiment data with all runs
        baseline_name: Name of baseline experiment (e.g., "baseline_ca_only")
        treatment_name: Name of treatment experiment (e.g., "add_sotl")
        metrics: List of metrics to analyze (auto-detected if None)

    Returns:
        Dictionary with contribution values and statistics for each metric:
        {
            "metric_name": {
                "baseline_mean": float,
                "treatment_mean": float,
                "contribution": float,  # Absolute difference
                "pct_contribution": float,  # Percent change
                "p_value": float,
                "cohens_d": float,
                "significant": bool,
            }
        }
    """
    baseline_df = raw_df[raw_df["experiment"] == baseline_name]
    treatment_df = raw_df[raw_df["experiment"] == treatment_name]

    if baseline_df.empty:
        raise ValueError(f"Baseline experiment '{baseline_name}' not found")
    if treatment_df.empty:
        raise ValueError(f"Treatment experiment '{treatment_name}' not found")

    comparison = paired_comparison(baseline_df, treatment_df, metrics)

    results = {}
    for metric, stats in comparison.items():
        baseline_mean = stats["baseline_mean"]
        treatment_mean = stats["treatment_mean"]
        contribution = treatment_mean - baseline_mean

        if baseline_mean != 0:
            pct_contribution = (contribution / abs(baseline_mean)) * 100
        else:
            pct_contribution = 0.0

        results[metric] = {
            "baseline_mean": baseline_mean,
            "treatment_mean": treatment_mean,
            "contribution": contribution,
            "pct_contribution": pct_contribution,
            "p_value": stats["p_value"],
            "cohens_d": stats["cohens_d"],
            "significant": stats["significant"],
        }

    return results

def compute_interaction_effect(
    raw_df: pd.DataFrame,
    baseline_name: str,
    single_a_name: str,
    single_b_name: str,
    combination_name: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the interaction effect between two algorithms.

    interaction(A,B) = metric(A+B) - metric(A) - metric(B) + metric(baseline)

    Interpretation:
    - Positive interaction (for "lower is better" metrics): interference
    - Negative interaction (for "lower is better" metrics): synergy
    - Vice versa for "higher is better" metrics

    Args:
        raw_df: Raw experiment data with all runs
        baseline_name: Name of baseline experiment (e.g., "baseline_ca_only")
        single_a_name: Name of first single-algorithm experiment (e.g., "add_sotl")
        single_b_name: Name of second single-algorithm experiment (e.g., "add_aco")
        combination_name: Name of combined experiment (e.g., "sotl_aco")
        metrics: List of metrics to analyze (auto-detected if None)

    Returns:
        Dictionary with interaction values for each metric:
        {
            "metric_name": {
                "baseline_mean": float,
                "single_a_mean": float,
                "single_b_mean": float,
                "combination_mean": float,
                "interaction": float,
                "expected_additive": float,  # What we'd expect if no interaction
                "interaction_type": str,  # "synergy" or "interference"
            }
        }
    """
    if metrics is None:
        metadata_cols = {"experiment", "run", "seed", "description", "algorithms"}
        metrics = [
            col for col in raw_df.columns
            if col not in metadata_cols and not col.endswith(
                ("_mean", "_std", "_ci95", "_min", "_max")
            )
        ]

    exp_means = raw_df.groupby("experiment")[metrics].mean()

    for exp_name in [baseline_name, single_a_name, single_b_name, combination_name]:
        if exp_name not in exp_means.index:
            raise ValueError(f"Experiment '{exp_name}' not found")

    results = {}

    lower_is_better = {"average_delay_final", "average_queue_length_final"}

    for metric in metrics:
        if metric not in exp_means.columns:
            continue

        baseline_mean = exp_means.loc[baseline_name, metric]
        single_a_mean = exp_means.loc[single_a_name, metric]
        single_b_mean = exp_means.loc[single_b_name, metric]
        combination_mean = exp_means.loc[combination_name, metric]

        interaction = (
            combination_mean
            - single_a_mean
            - single_b_mean
            + baseline_mean
        )

        expected_additive = single_a_mean + single_b_mean - baseline_mean

        if metric in lower_is_better:

            interaction_type = "synergy" if interaction < 0 else "interference"
        else:

            interaction_type = "synergy" if interaction > 0 else "interference"

        results[metric] = {
            "baseline_mean": float(baseline_mean),
            "single_a_mean": float(single_a_mean),
            "single_b_mean": float(single_b_mean),
            "combination_mean": float(combination_mean),
            "interaction": float(interaction),
            "expected_additive": float(expected_additive),
            "interaction_type": interaction_type,
        }

    return results

def compute_subtractive_importance(
    raw_df: pd.DataFrame,
    full_stack_name: str,
    reduced_name: str,
    metrics: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute the subtractive importance of an algorithm in the full stack.

    importance(A) = metric(full_stack) - metric(full_stack - A)

    This measures how much removing algorithm A from the full stack
    hurts performance.

    Args:
        raw_df: Raw experiment data with all runs
        full_stack_name: Name of full stack experiment (e.g., "full_stack")
        reduced_name: Name of experiment with one algorithm removed
                     (e.g., "full_minus_sotl")
        metrics: List of metrics to analyze (auto-detected if None)

    Returns:
        Dictionary with importance values and statistics for each metric:
        {
            "metric_name": {
                "full_stack_mean": float,
                "reduced_mean": float,
                "importance": float,  # How much removing hurts
                "pct_impact": float,  # Percent degradation
                "p_value": float,
                "cohens_d": float,
                "significant": bool,
            }
        }
    """
    full_stack_df = raw_df[raw_df["experiment"] == full_stack_name]
    reduced_df = raw_df[raw_df["experiment"] == reduced_name]

    if full_stack_df.empty:
        raise ValueError(f"Full stack experiment '{full_stack_name}' not found")
    if reduced_df.empty:
        raise ValueError(f"Reduced experiment '{reduced_name}' not found")

    comparison = paired_comparison(full_stack_df, reduced_df, metrics)

    lower_is_better = {"average_delay_final", "average_queue_length_final"}

    results = {}
    for metric, stats in comparison.items():
        full_mean = stats["baseline_mean"]
        reduced_mean = stats["treatment_mean"]

        if metric in lower_is_better:
            importance = reduced_mean - full_mean
        else:
            importance = full_mean - reduced_mean

        if full_mean != 0:
            pct_impact = (importance / abs(full_mean)) * 100
        else:
            pct_impact = 0.0

        results[metric] = {
            "full_stack_mean": full_mean,
            "reduced_mean": reduced_mean,
            "importance": importance,
            "pct_impact": pct_impact,
            "p_value": stats["p_value"],
            "cohens_d": abs(stats["cohens_d"]),
            "significant": stats["significant"],
        }

    return results
