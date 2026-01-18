#!/usr/bin/env python3
"""
Run ablation study experiments for SOITCS.

This script runs all 13 experiment configurations from the ablation study
with multiple seeds to measure the contribution of each algorithm component.

Usage:
    # Quick test run (3 runs per experiment)
    uv run python scripts/run_ablation.py --n-runs 3 --output results/ablation_test/

    # Full ablation study (20 runs per experiment = 260 total runs)
    uv run python scripts/run_ablation.py --output results/ablation/

Outputs:
    results/ablation/raw_data.csv         - Full data from all runs
    results/ablation/summary_statistics.csv - Aggregated statistics per experiment
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Run SOITCS ablation study experiments"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiments/ablation_study.yaml",
        help="Path to ablation study configuration YAML file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/ablation/",
        help="Output directory for results (default: results/ablation/)",
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
        "--experiments",
        type=str,
        nargs="+",
        default=None,
        help="Run only specific experiments by name (default: run all)",
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
    from src.evaluation.experiment import (
        load_experiment_config,
        ExperimentRunner,
        ExperimentConfig,
    )
    from src.evaluation.analysis import compute_statistics, export_csv_summary

    logger.info(f"Loading configuration from {args.config}")
    config = load_experiment_config(args.config)

    # Get settings
    experiments = config.get("experiments", [])
    settings = config.get("settings", {})

    # Filter experiments if specified
    if args.experiments:
        experiments = [
            exp for exp in experiments
            if exp["name"] in args.experiments
        ]
        if not experiments:
            logger.error(f"No matching experiments found for: {args.experiments}")
            sys.exit(1)

    n_runs = args.n_runs or settings.get("n_runs", 20)
    duration_ticks = settings.get("duration_ticks", 5400)
    metrics = config.get("metrics", [
        "average_delay",
        "throughput",
        "average_queue_length",
        "average_speed",
        "total_vehicles",
    ])

    # Apply duration_ticks and metrics to each experiment
    for exp in experiments:
        if "duration_ticks" not in exp:
            exp["duration_ticks"] = duration_ticks
        if "metrics" not in exp:
            exp["metrics"] = metrics

    # Create runner
    network_path = args.network or settings.get("network_path")
    runner = ExperimentRunner(network_path=network_path)

    total_runs = len(experiments) * n_runs
    logger.info(f"Ablation Study: {len(experiments)} experiments x {n_runs} runs = {total_runs} total runs")
    logger.info(f"Duration: {duration_ticks} ticks per run")
    logger.info(f"Output directory: {output_dir}")

    # Run experiments with progress tracking
    all_results = []
    run_count = 0
    start_time = time.time()

    for exp in experiments:
        exp_name = exp["name"]
        exp_desc = exp.get("description", "")
        algorithms = exp.get("algorithms", ["cellular_automata"])

        logger.info(f"\n{'='*60}")
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Description: {exp_desc}")
        logger.info(f"Algorithms: {algorithms}")
        logger.info(f"{'='*60}")

        for run_idx in range(n_runs):
            seed = args.base_seed + run_idx
            run_count += 1

            # Progress info
            elapsed = time.time() - start_time
            if run_count > 1:
                avg_time = elapsed / (run_count - 1)
                remaining = avg_time * (total_runs - run_count + 1)
                eta_str = f"ETA: {remaining/60:.1f} min"
            else:
                eta_str = ""

            logger.info(
                f"[{run_count}/{total_runs}] {exp_name} run {run_idx + 1}/{n_runs} "
                f"(seed={seed}) {eta_str}"
            )

            try:
                exp_config = ExperimentConfig.from_dict(exp, seed)
                result = runner.run(exp_config)

                row = {
                    "experiment": exp_name,
                    "description": exp_desc,
                    "algorithms": ",".join(algorithms),
                    "run": run_idx,
                    "seed": seed,
                    **result.summary,
                }
                all_results.append(row)

            except Exception as e:
                logger.error(f"Run failed: {e}")
                continue

    # Create DataFrame
    import pandas as pd
    df = pd.DataFrame(all_results)

    # Save raw data
    raw_path = output_dir / "raw_data.csv"
    df.to_csv(raw_path, index=False)
    logger.info(f"\nRaw data saved to {raw_path}")

    # Compute and save summary statistics
    summary_df = compute_statistics(df)
    summary_path = output_dir / "summary_statistics.csv"
    export_csv_summary(summary_df, str(summary_path))
    logger.info(f"Summary statistics saved to {summary_path}")

    # Print summary to console
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Total runs: {run_count}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"\nResults saved to: {output_dir}")
    print(f"\nNext step: Run analysis with:")
    print(f"  uv run python scripts/analyze_ablation.py --input {output_dir}")
    print(f"{'='*60}\n")

    logger.info("Ablation study completed successfully!")


if __name__ == "__main__":
    main()
