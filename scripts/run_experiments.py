#!/usr/bin/env python3
"""
CLI for running SOITCS experiments.

Usage:
    python scripts/run_experiments.py --config config/experiments/standard_comparison.yaml --output results/

Outputs:
    results/raw_data.csv         - Full data from all runs
    results/summary_table.csv    - Aggregated statistics
    results/comparison.tex       - LaTeX table for thesis
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run SOITCS benchmarking experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Output directory for results (default: results/)",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help="Override number of runs per experiment (default: from config)",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Base seed for random number generation (default: 0)",
    )
    parser.add_argument(
        "--network",
        type=str,
        default=None,
        help="Override network path (default: from config)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load configuration
    from src.evaluation.experiment import load_experiment_config, ExperimentRunner
    from src.evaluation.analysis import (
        compute_statistics,
        export_csv_summary,
        export_latex_table,
    )

    logger.info(f"Loading configuration from {args.config}")
    config = load_experiment_config(args.config)

    # Get settings
    experiments = config.get("experiments", [])
    settings = config.get("settings", {})

    n_runs = args.n_runs or settings.get("n_runs", 10)
    duration_ticks = settings.get("duration_ticks", 5400)
    metrics = config.get("metrics", ["average_delay", "throughput", "average_queue_length"])

    # Apply duration_ticks and metrics to each experiment
    for exp in experiments:
        if "duration_ticks" not in exp:
            exp["duration_ticks"] = duration_ticks
        if "metrics" not in exp:
            exp["metrics"] = metrics

    # Create runner
    network_path = args.network or settings.get("network_path")
    runner = ExperimentRunner(network_path=network_path)

    logger.info(f"Running {len(experiments)} experiments, {n_runs} runs each")
    logger.info(f"Duration: {duration_ticks} ticks per run")

    # Run experiments
    df = runner.run_comparison(
        configs=experiments,
        n_runs=n_runs,
        base_seed=args.base_seed,
    )

    # Save raw data
    raw_path = output_dir / "raw_data.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"Raw data saved to {raw_path}")

    # Compute and save summary statistics
    summary_df = compute_statistics(df)
    summary_path = output_dir / "summary_table.csv"
    export_csv_summary(summary_df, str(summary_path))
    logger.info(f"Summary table saved to {summary_path}")

    # Generate LaTeX table
    latex_path = output_dir / "comparison.tex"
    latex_content = export_latex_table(summary_df)
    with open(latex_path, "w") as f:
        f.write(latex_content)
    logger.info(f"LaTeX table saved to {latex_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print(summary_df.to_string())
    print("=" * 60 + "\n")

    logger.info("Experiments completed successfully!")


if __name__ == "__main__":
    main()
