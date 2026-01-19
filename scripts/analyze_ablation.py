"""
Analyze ablation study results for SOITraCS.

This script analyzes the results from run_ablation.py and generates:
- Additive contribution analysis (each algorithm vs baseline)
- Interaction effects (2-way combinations vs sum of singles)
- Subtractive importance (full stack minus each component)
- Statistical significance tests with p-values and effect sizes
- Figures and LaTeX tables for thesis

Usage:
    uv run python scripts/analyze_ablation.py --input results/ablation/ --output results/ablation/

Outputs:
    additive_contributions.csv   - Per-algorithm contribution vs baseline
    interaction_effects.csv      - 2-way interaction effects (synergy/conflict)
    subtractive_importance.csv   - Importance of each algorithm in full stack
    pairwise_comparisons.csv     - All p-values and effect sizes
    ablation_table.tex           - LaTeX table for thesis
    figures/                     - Visualization plots
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.analysis import (
    paired_comparison,
    compute_statistics,
    export_latex_table,
)

ALGORITHM_MAP = {
    "baseline_ca_only": [],
    "add_sotl": ["sotl"],
    "add_aco": ["aco"],
    "add_pso": ["pso"],
    "add_som": ["som"],
    "sotl_aco": ["sotl", "aco"],
    "sotl_pso": ["sotl", "pso"],
    "sotl_som": ["sotl", "som"],
    "full_stack": ["sotl", "aco", "pso", "som"],
    "full_minus_sotl": ["aco", "pso", "som"],
    "full_minus_aco": ["sotl", "pso", "som"],
    "full_minus_pso": ["sotl", "aco", "som"],
    "full_minus_som": ["sotl", "aco", "pso"],
}

METRICS = ["average_delay_final", "throughput_final", "average_queue_length_final", "average_speed_final"]
LOWER_IS_BETTER = {"average_delay_final", "average_queue_length_final"}

def load_raw_data(input_dir: Path) -> pd.DataFrame:
    """Load raw data from ablation study."""
    raw_path = input_dir / "raw_data.csv"
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found: {raw_path}")
    return pd.read_csv(raw_path)

def compute_additive_contributions(
    df: pd.DataFrame,
    baseline_name: str = "baseline_ca_only",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute additive contribution of each single algorithm vs baseline.

    contribution(A) = metric(baseline + A) - metric(baseline)

    Args:
        df: Raw data from ablation study
        baseline_name: Name of baseline experiment
        metrics: Metrics to analyze

    Returns:
        DataFrame with contribution values and significance
    """
    if metrics is None:
        metrics = METRICS

    baseline_df = df[df["experiment"] == baseline_name]

    single_additions = {
        "add_sotl": "SOTL",
        "add_aco": "ACO",
        "add_pso": "PSO",
        "add_som": "SOM",
    }

    results = []

    for exp_name, algo_name in single_additions.items():
        treatment_df = df[df["experiment"] == exp_name]

        if treatment_df.empty:
            continue

        comparison = paired_comparison(baseline_df, treatment_df, metrics)

        row = {"algorithm": algo_name, "experiment": exp_name}

        for metric in metrics:
            if metric not in comparison:
                continue

            stats = comparison[metric]
            baseline_mean = stats["baseline_mean"]
            treatment_mean = stats["treatment_mean"]

            contribution = treatment_mean - baseline_mean

            if baseline_mean != 0:
                pct_contribution = (contribution / abs(baseline_mean)) * 100
            else:
                pct_contribution = 0.0

            metric_short = metric.replace("_final", "")

            row[f"{metric_short}_contribution"] = contribution
            row[f"{metric_short}_pct_change"] = pct_contribution
            row[f"{metric_short}_p_value"] = stats["p_value"]
            row[f"{metric_short}_cohens_d"] = stats["cohens_d"]
            row[f"{metric_short}_significant"] = stats["significant"]

        results.append(row)

    return pd.DataFrame(results)

def compute_interaction_effects(
    df: pd.DataFrame,
    baseline_name: str = "baseline_ca_only",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute interaction effects for 2-way algorithm combinations.

    interaction(A,B) = metric(A+B) - metric(A) - metric(B) + metric(baseline)

    Positive interaction (for "lower is better" metrics): synergy
    Negative interaction: interference/conflict

    Args:
        df: Raw data from ablation study
        baseline_name: Name of baseline experiment
        metrics: Metrics to analyze

    Returns:
        DataFrame with interaction effects (empty if required experiments not present)
    """
    if metrics is None:
        metrics = METRICS

    exp_means = df.groupby("experiment")[metrics].mean()

    if baseline_name not in exp_means.index:
        logging.warning(f"Baseline {baseline_name} not found, skipping interaction analysis")
        return pd.DataFrame()

    baseline_means = exp_means.loc[baseline_name]

    combinations = [
        ("sotl_aco", "add_sotl", "add_aco", "SOTL", "ACO"),
        ("sotl_pso", "add_sotl", "add_pso", "SOTL", "PSO"),
        ("sotl_som", "add_sotl", "add_som", "SOTL", "SOM"),
    ]

    results = []

    for combo_exp, single_a_exp, single_b_exp, algo_a, algo_b in combinations:
        if combo_exp not in exp_means.index:
            continue
        if single_a_exp not in exp_means.index:
            continue
        if single_b_exp not in exp_means.index:
            continue

        combo_means = exp_means.loc[combo_exp]
        single_a_means = exp_means.loc[single_a_exp]
        single_b_means = exp_means.loc[single_b_exp]

        row = {
            "combination": f"{algo_a}+{algo_b}",
            "experiment": combo_exp,
        }

        for metric in metrics:

            interaction = (
                combo_means[metric]
                - single_a_means[metric]
                - single_b_means[metric]
                + baseline_means[metric]
            )

            metric_short = metric.replace("_final", "")

            if metric in LOWER_IS_BETTER:

                interaction_type = "synergy" if interaction < 0 else "interference"
            else:

                interaction_type = "synergy" if interaction > 0 else "interference"

            row[f"{metric_short}_interaction"] = interaction
            row[f"{metric_short}_type"] = interaction_type

        combo_df = df[df["experiment"] == combo_exp]
        single_a_df = df[df["experiment"] == single_a_exp]
        single_b_df = df[df["experiment"] == single_b_exp]

        primary_metric = "average_delay_final"
        if primary_metric in metrics:
            a_mean = single_a_means[primary_metric]
            b_mean = single_b_means[primary_metric]

            if a_mean < b_mean:
                best_single_df = single_a_df
                best_single_name = algo_a
            else:
                best_single_df = single_b_df
                best_single_name = algo_b

            comparison = paired_comparison(best_single_df, combo_df, [primary_metric])
            if primary_metric in comparison:
                row["vs_best_single"] = best_single_name
                row["vs_best_single_p_value"] = comparison[primary_metric]["p_value"]
                row["vs_best_single_improvement"] = comparison[primary_metric]["pct_change"]

        results.append(row)

    return pd.DataFrame(results)

def compute_subtractive_importance(
    df: pd.DataFrame,
    full_stack_name: str = "full_stack",
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute subtractive importance of each algorithm in full stack.

    importance(A) = metric(full_stack) - metric(full_stack - A)

    Shows how much removing A hurts performance.

    Args:
        df: Raw data from ablation study
        full_stack_name: Name of full stack experiment
        metrics: Metrics to analyze

    Returns:
        DataFrame with importance values (empty if full_stack not present)
    """
    if metrics is None:
        metrics = METRICS

    full_stack_df = df[df["experiment"] == full_stack_name]

    if full_stack_df.empty:
        logging.warning(f"Full stack experiment {full_stack_name} not found, skipping subtractive analysis")
        return pd.DataFrame()

    subtractive_experiments = {
        "full_minus_sotl": "SOTL",
        "full_minus_aco": "ACO",
        "full_minus_pso": "PSO",
        "full_minus_som": "SOM",
    }

    results = []

    for exp_name, algo_name in subtractive_experiments.items():
        reduced_df = df[df["experiment"] == exp_name]

        if reduced_df.empty:
            continue

        comparison = paired_comparison(full_stack_df, reduced_df, metrics)

        row = {"algorithm": algo_name, "experiment": exp_name}

        for metric in metrics:
            if metric not in comparison:
                continue

            stats = comparison[metric]

            full_mean = stats["baseline_mean"]
            reduced_mean = stats["treatment_mean"]

            if metric in LOWER_IS_BETTER:
                importance = reduced_mean - full_mean
            else:
                importance = full_mean - reduced_mean

            if full_mean != 0:
                pct_importance = (importance / abs(full_mean)) * 100
            else:
                pct_importance = 0.0

            metric_short = metric.replace("_final", "")

            row[f"{metric_short}_importance"] = importance
            row[f"{metric_short}_pct_impact"] = pct_importance
            row[f"{metric_short}_p_value"] = stats["p_value"]
            row[f"{metric_short}_cohens_d"] = abs(stats["cohens_d"])
            row[f"{metric_short}_significant"] = stats["significant"]

        results.append(row)

    return pd.DataFrame(results)

def compute_all_pairwise_comparisons(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute pairwise comparisons between all experiments.

    Args:
        df: Raw data from ablation study
        metrics: Metrics to analyze

    Returns:
        DataFrame with all pairwise p-values and effect sizes
    """
    if metrics is None:
        metrics = METRICS

    experiments = df["experiment"].unique()
    results = []

    for i, exp_a in enumerate(experiments):
        for exp_b in experiments[i + 1:]:
            df_a = df[df["experiment"] == exp_a]
            df_b = df[df["experiment"] == exp_b]

            comparison = paired_comparison(df_a, df_b, metrics)

            for metric, stats in comparison.items():
                metric_short = metric.replace("_final", "")
                results.append({
                    "experiment_a": exp_a,
                    "experiment_b": exp_b,
                    "metric": metric_short,
                    "mean_a": stats["baseline_mean"],
                    "mean_b": stats["treatment_mean"],
                    "difference": stats["treatment_mean"] - stats["baseline_mean"],
                    "pct_change": stats["pct_change"],
                    "t_statistic": stats["t_statistic"],
                    "p_value": stats["p_value"],
                    "cohens_d": stats["cohens_d"],
                    "significant": stats["significant"],
                })

    return pd.DataFrame(results)

def generate_ablation_latex_table(
    additive_df: pd.DataFrame,
    subtractive_df: pd.DataFrame,
) -> str:
    """Generate LaTeX table summarizing ablation results."""
    if additive_df.empty:
        return "% No ablation data available"

    has_subtractive = not subtractive_df.empty and "algorithm" in subtractive_df.columns

    if has_subtractive:
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Ablation Study Results: Algorithm Contributions}",
            "\\label{tab:ablation}",
            "\\begin{tabular}{lrrrr}",
            "\\toprule",
            "Algorithm & \\multicolumn{2}{c}{Additive Contribution} & \\multicolumn{2}{c}{Subtractive Importance} \\\\",
            "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5}",
            " & Delay (\\%) & Throughput (\\%) & Delay (\\%) & Throughput (\\%) \\\\",
            "\\midrule",
        ]
    else:
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Ablation Study Results: Algorithm Contributions}",
            "\\label{tab:ablation}",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Algorithm & Delay (\\%) & Throughput (\\%) \\\\",
            "\\midrule",
        ]

    algorithms = ["SOTL", "ACO", "PSO", "SOM"]

    def fmt(val, sig):
        marker = "*" if sig else ""
        return f"{val:+.1f}{marker}"

    for algo in algorithms:
        add_row = additive_df[additive_df["algorithm"] == algo]

        if add_row.empty:
            continue

        add_delay = add_row["average_delay_pct_change"].values[0]
        add_throughput = add_row["throughput_pct_change"].values[0]
        add_delay_sig = add_row["average_delay_significant"].values[0]
        add_throughput_sig = add_row["throughput_significant"].values[0]

        if has_subtractive:
            sub_row = subtractive_df[subtractive_df["algorithm"] == algo]
            if sub_row.empty:
                lines.append(
                    f"{algo} & {fmt(add_delay, add_delay_sig)} & {fmt(add_throughput, add_throughput_sig)} "
                    f"& -- & -- \\\\"
                )
            else:
                sub_delay = sub_row["average_delay_pct_impact"].values[0]
                sub_throughput = sub_row["throughput_pct_impact"].values[0]
                sub_delay_sig = sub_row["average_delay_significant"].values[0]
                sub_throughput_sig = sub_row["throughput_significant"].values[0]

                lines.append(
                    f"{algo} & {fmt(add_delay, add_delay_sig)} & {fmt(add_throughput, add_throughput_sig)} "
                    f"& {fmt(sub_delay, sub_delay_sig)} & {fmt(sub_throughput, sub_throughput_sig)} \\\\"
                )
        else:
            lines.append(
                f"{algo} & {fmt(add_delay, add_delay_sig)} & {fmt(add_throughput, add_throughput_sig)} \\\\"
            )

    if has_subtractive:
        lines.extend([
            "\\bottomrule",
            "\\multicolumn{5}{l}{\\footnotesize * indicates $p < 0.05$. Negative delay = improvement.} \\\\",
            "\\end{tabular}",
            "\\end{table}",
        ])
    else:
        lines.extend([
            "\\bottomrule",
            "\\multicolumn{3}{l}{\\footnotesize * indicates $p < 0.05$. Negative delay = improvement.} \\\\",
            "\\end{tabular}",
            "\\end{table}",
        ])

    return "\n".join(lines)

def generate_figures(
    df: pd.DataFrame,
    additive_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    subtractive_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate visualization figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        logging.warning("matplotlib not available, skipping figure generation")
        return

    if additive_df.empty:
        logging.warning("No additive contribution data, skipping figure generation")
        return

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    algorithms = additive_df["algorithm"].tolist()
    x = np.arange(len(algorithms))
    width = 0.35

    delay_changes = additive_df["average_delay_pct_change"].values
    throughput_changes = additive_df["throughput_pct_change"].values

    bars1 = ax.bar(x - width/2, delay_changes, width, label="Delay Change (%)", color="steelblue")
    bars2 = ax.bar(x + width/2, throughput_changes, width, label="Throughput Change (%)", color="darkorange")

    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Percent Change vs Baseline")
    ax.set_title("Additive Contribution of Each Algorithm")
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "contribution_bar_chart.png", dpi=150)
    plt.close()

    if not interaction_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))

        combinations = interaction_df["combination"].tolist()
        interactions = interaction_df["average_delay_interaction"].values

        colors = ["green" if i < 0 else "red" for i in interactions]
        bars = ax.barh(combinations, interactions, color=colors, alpha=0.7)

        ax.set_xlabel("Interaction Effect (Delay)")
        ax.set_title("Two-Way Algorithm Interactions\n(Negative = Synergy, Positive = Interference)")
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(axis="x", alpha=0.3)

        plt.tight_layout()
        plt.savefig(figures_dir / "interaction_heatmap.png", dpi=150)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    experiments_order = [
        "baseline_ca_only", "add_sotl", "add_aco", "add_pso", "add_som",
        "sotl_aco", "sotl_pso", "sotl_som", "full_stack"
    ]

    existing_exps = [e for e in experiments_order if e in df["experiment"].values]
    data_delay = [df[df["experiment"] == exp]["average_delay_final"].values for exp in existing_exps]
    data_throughput = [df[df["experiment"] == exp]["throughput_final"].values for exp in existing_exps]

    labels = [e.replace("_", "\n") for e in existing_exps]

    axes[0].boxplot(data_delay, labels=labels)
    axes[0].set_ylabel("Average Delay")
    axes[0].set_title("Delay Distribution by Configuration")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].boxplot(data_throughput, labels=labels)
    axes[1].set_ylabel("Throughput")
    axes[1].set_title("Throughput Distribution by Configuration")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(figures_dir / "performance_comparison.png", dpi=150)
    plt.close()

    logging.info(f"Figures saved to {figures_dir}")

def print_summary(
    additive_df: pd.DataFrame,
    interaction_df: pd.DataFrame,
    subtractive_df: pd.DataFrame,
) -> None:
    """Print analysis summary to console."""
    print("\n" + "=" * 70)
    print("ABLATION STUDY ANALYSIS SUMMARY")
    print("=" * 70)

    print("\n1. ADDITIVE CONTRIBUTIONS (vs Baseline)")
    print("-" * 50)
    if additive_df.empty:
        print("  (No additive contribution data available)")
    else:
        for _, row in additive_df.iterrows():
            algo = row["algorithm"]
            delay_pct = row.get("average_delay_pct_change", 0)
            delay_sig = row.get("average_delay_significant", False)
            sig_marker = "*" if delay_sig else ""

            print(f"  {algo:8s}: Delay {delay_pct:+6.1f}%{sig_marker}")

    print("\n2. TWO-WAY INTERACTION EFFECTS")
    print("-" * 50)
    if interaction_df.empty:
        print("  (No interaction data available - need 2-way combination experiments)")
    else:
        for _, row in interaction_df.iterrows():
            combo = row["combination"]
            interaction = row.get("average_delay_interaction", 0)
            int_type = row.get("average_delay_type", "unknown")

            print(f"  {combo:12s}: {interaction:+6.2f} ({int_type})")

    print("\n3. SUBTRACTIVE IMPORTANCE (in Full Stack)")
    print("-" * 50)
    if subtractive_df.empty:
        print("  (No subtractive data available - need full_stack and full_minus_* experiments)")
    else:
        for _, row in subtractive_df.iterrows():
            algo = row["algorithm"]
            impact_pct = row.get("average_delay_pct_impact", 0)
            sig = row.get("average_delay_significant", False)
            sig_marker = "*" if sig else ""

            print(f"  {algo:8s}: Delay impact {impact_pct:+6.1f}%{sig_marker} when removed")

    print("\n* indicates statistically significant (p < 0.05)")
    print("=" * 70 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze SOITraCS ablation study results"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/ablation/",
        help="Input directory containing raw_data.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else input_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {input_dir}")

    df = load_raw_data(input_dir)
    logger.info(f"Loaded {len(df)} runs from {df['experiment'].nunique()} experiments")

    logger.info("Computing additive contributions...")
    additive_df = compute_additive_contributions(df)

    logger.info("Computing interaction effects...")
    interaction_df = compute_interaction_effects(df)

    logger.info("Computing subtractive importance...")
    subtractive_df = compute_subtractive_importance(df)

    logger.info("Computing pairwise comparisons...")
    pairwise_df = compute_all_pairwise_comparisons(df)

    additive_df.to_csv(output_dir / "additive_contributions.csv", index=False)
    logger.info(f"Saved additive_contributions.csv")

    interaction_df.to_csv(output_dir / "interaction_effects.csv", index=False)
    logger.info(f"Saved interaction_effects.csv")

    subtractive_df.to_csv(output_dir / "subtractive_importance.csv", index=False)
    logger.info(f"Saved subtractive_importance.csv")

    pairwise_df.to_csv(output_dir / "pairwise_comparisons.csv", index=False)
    logger.info(f"Saved pairwise_comparisons.csv")

    latex_content = generate_ablation_latex_table(additive_df, subtractive_df)
    latex_path = output_dir / "ablation_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"Saved ablation_table.tex")

    if not args.no_figures:
        logger.info("Generating figures...")
        generate_figures(df, additive_df, interaction_df, subtractive_df, output_dir)

    print_summary(additive_df, interaction_df, subtractive_df)

    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
