"""
Experiment configuration and runner for headless benchmarking.

Provides infrastructure for running reproducible experiments with controlled
random seeds, multiple runs, and metrics collection.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import copy

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""

    name: str
    seed: int
    duration_ticks: int
    enabled_algorithms: List[str]
    scenario: Optional[str] = None
    metrics_to_track: List[str] = field(default_factory=lambda: [
        "average_delay",
        "throughput",
        "average_queue_length",
        "average_speed",
        "total_vehicles",
    ])
    network_path: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any], seed: int) -> "ExperimentConfig":
        """Create config from dictionary with a specific seed."""
        return cls(
            name=data["name"],
            seed=seed,
            duration_ticks=data.get("duration_ticks", 5400),
            enabled_algorithms=data.get("algorithms", ["cellular_automata"]),
            scenario=data.get("scenario"),
            metrics_to_track=data.get("metrics", [
                "average_delay",
                "throughput",
                "average_queue_length",
            ]),
            network_path=data.get("network_path"),
        )


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    config: ExperimentConfig
    time_series: Dict[str, np.ndarray]  # metric -> values over time
    summary: Dict[str, float]  # mean, std, min, max per metric
    algorithm_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "config_name": self.config.name,
            "seed": self.config.seed,
            "duration_ticks": self.config.duration_ticks,
            "enabled_algorithms": self.config.enabled_algorithms,
            "summary": self.summary,
            "algorithm_metrics": self.algorithm_metrics,
        }


class ExperimentRunner:
    """
    Runs headless experiments for benchmarking.

    Handles simulation setup, seed control, metrics collection, and
    multi-run comparisons.
    """

    def __init__(self, network_path: Optional[str] = None):
        """
        Initialize experiment runner.

        Args:
            network_path: Optional path to network YAML file
        """
        self.network_path = network_path
        self._results: List[ExperimentResult] = []

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run a single experiment.

        Args:
            config: Experiment configuration

        Returns:
            ExperimentResult with time series and summary statistics
        """
        from src.core.simulation import Simulation
        from src.core.state import SimulationState
        from src.core.clock import SimulationClock
        from src.core.event_bus import get_event_bus
        from config.settings import get_settings

        logger.info(f"Running experiment: {config.name} (seed={config.seed})")

        # Create fresh simulation
        settings = get_settings()

        # Configure algorithms based on experiment
        self._configure_algorithms(settings, config.enabled_algorithms)

        sim = Simulation(
            settings=settings,
            state=SimulationState(),
            clock=SimulationClock(),
            event_bus=get_event_bus(),
            headless=True,
        )

        # Set seed before initialization
        sim.set_seed(config.seed)

        # Initialize with network
        network_path = config.network_path or self.network_path
        sim.initialize(network_path)

        # Activate scenario if specified
        if config.scenario:
            sim.activate_scenario(config.scenario)

        # Run simulation and collect metrics
        time_series: Dict[str, List[float]] = {
            metric: [] for metric in config.metrics_to_track
        }

        snapshots = sim.run_ticks(config.duration_ticks)

        # Extract time series from snapshots
        for snapshot in snapshots:
            for metric in config.metrics_to_track:
                if hasattr(snapshot, metric):
                    time_series[metric].append(getattr(snapshot, metric))

        # Convert to numpy arrays
        time_series_np = {
            metric: np.array(values) for metric, values in time_series.items()
        }

        # Compute summary statistics
        summary = self._compute_summary(time_series_np)

        # Collect algorithm-specific metrics
        algorithm_metrics = {}
        for algo_name, algo in sim.state.algorithms.items():
            if algo.enabled:
                algorithm_metrics[algo_name] = algo.get_metrics()

        result = ExperimentResult(
            config=config,
            time_series=time_series_np,
            summary=summary,
            algorithm_metrics=algorithm_metrics,
        )

        self._results.append(result)
        return result

    def _configure_algorithms(
        self, settings: Any, enabled_algorithms: List[str]
    ) -> None:
        """Configure which algorithms are enabled in settings."""
        all_algorithms = [
            "cellular_automata", "sotl", "aco", "pso", "som", "marl"
        ]

        for algo_name in all_algorithms:
            algo_settings = getattr(settings.algorithms, algo_name, None)
            if algo_settings is not None:
                algo_settings["enabled"] = algo_name in enabled_algorithms

    def _compute_summary(
        self, time_series: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute summary statistics for each metric."""
        summary = {}

        for metric, values in time_series.items():
            if len(values) > 0:
                summary[f"{metric}_mean"] = float(np.mean(values))
                summary[f"{metric}_std"] = float(np.std(values))
                summary[f"{metric}_min"] = float(np.min(values))
                summary[f"{metric}_max"] = float(np.max(values))
                # Also compute final value (last 10% average for stability)
                tail_size = max(1, len(values) // 10)
                summary[f"{metric}_final"] = float(np.mean(values[-tail_size:]))

        return summary

    def run_comparison(
        self,
        configs: List[Dict[str, Any]],
        n_runs: int = 10,
        base_seed: int = 0,
    ) -> pd.DataFrame:
        """
        Run multiple configs with multiple seeds, return comparison table.

        Args:
            configs: List of experiment configuration dictionaries
            n_runs: Number of runs per configuration (each with different seed)
            base_seed: Starting seed (seeds will be base_seed, base_seed+1, ...)

        Returns:
            DataFrame with comparison results
        """
        all_results: List[Dict[str, Any]] = []

        for config_dict in configs:
            for run_idx in range(n_runs):
                seed = base_seed + run_idx
                config = ExperimentConfig.from_dict(config_dict, seed)

                try:
                    result = self.run(config)

                    row = {
                        "experiment": config.name,
                        "run": run_idx,
                        "seed": seed,
                        **result.summary,
                    }
                    all_results.append(row)

                except Exception as e:
                    logger.error(
                        f"Experiment {config.name} run {run_idx} failed: {e}"
                    )
                    continue

        return pd.DataFrame(all_results)

    def get_all_results(self) -> List[ExperimentResult]:
        """Get all results from this runner."""
        return self._results

    def clear_results(self) -> None:
        """Clear all stored results."""
        self._results.clear()


def load_experiment_config(path: str) -> Dict[str, Any]:
    """
    Load experiment configuration from YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
