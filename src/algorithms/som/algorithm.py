"""
Self-Organizing Map (SOM) for traffic pattern recognition.

Implements Kohonen's Self-Organizing Map algorithm for unsupervised
clustering of traffic states into interpretable patterns.

Features:
- Online learning during simulation
- Quantization error tracking for map quality assessment
- Feature importance analysis via weight variance
- Calibration method for data-driven pattern labeling

References:
    [1] Kohonen, T. (1990). The self-organizing map.
        Proceedings of the IEEE, 78(9), 1464-1480.
    [2] Kohonen, T. (2001). Self-Organizing Maps (3rd ed.). Springer.
    [3] Ultsch, A., & Siemon, H. P. (1990). Kohonen's self organizing
        feature maps for exploratory data analysis. INNC.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import random
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState
from src.core.event_bus import Event, EventType, get_event_bus

@dataclass
class SOMMetrics:
    """Metrics for tracking SOM quality and behavior."""

    quantization_errors: List[float] = field(default_factory=list)
    classification_history: List[str] = field(default_factory=list)
    bmu_history: List[Tuple[int, int]] = field(default_factory=list)

    def record_step(
        self, qe: float, pattern: str, bmu: Tuple[int, int]
    ) -> None:
        """Record metrics for a single step."""
        self.quantization_errors.append(qe)
        self.classification_history.append(pattern)
        self.bmu_history.append(bmu)

        max_history = 1000
        if len(self.quantization_errors) > max_history:
            self.quantization_errors = self.quantization_errors[-max_history:]
            self.classification_history = self.classification_history[-max_history:]
            self.bmu_history = self.bmu_history[-max_history:]

    def get_mean_qe(self, window: int = 100) -> float:
        """Get mean quantization error over recent window."""
        if not self.quantization_errors:
            return 0.0
        recent = self.quantization_errors[-window:]
        return float(np.mean(recent))

class SOMAlgorithm(BaseAlgorithm):
    """
    Self-Organizing Map for traffic pattern recognition.

    Classifies current traffic state into patterns:
    - Free flow: Low density, high speeds
    - Moderate congestion: Medium density
    - Rush hour: High overall traffic volume
    - Incident: Localized congestion patterns

    The SOM can operate in two modes:
    1. Pre-trained: Load weights from file for consistent classification
    2. Online learning: Adapt to traffic patterns during simulation
    """

    @property
    def name(self) -> str:
        return "som"

    @property
    def display_name(self) -> str:
        return "Pattern Recognition (SOM)"

    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.SOM_COLOR

    @property
    def current_pattern(self) -> str:
        """Get current detected traffic pattern."""
        return self._current_pattern

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        grid_size = config.get("grid_size", [10, 10])
        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.learning_rate = config.get("learning_rate", 0.5)
        self.learning_rate_decay = config.get("learning_rate_decay", 0.9995)
        self.sigma = config.get("sigma", 3.0)
        self.sigma_decay = config.get("sigma_decay", 0.9995)
        self.pretrained_path = config.get("pretrained_path")
        self.online_learning = config.get("online_learning", False)

        self._weights: Optional[np.ndarray] = None
        self._feature_dim = 0
        self._feature_names: List[str] = []

        self._neuron_labels: Dict[Tuple[int, int], str] = {}
        self._pattern_labels = {
            0: "free_flow",
            1: "moderate",
            2: "rush_hour",
            3: "incident",
        }

        self._current_pattern = "free_flow"
        self._current_bmu = (0, 0)
        self._current_qe = 0.0

        self._metrics = SOMMetrics()

        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize SOM weights."""
        self._network = network

        n_roads = len(network.roads)
        self._feature_dim = n_roads + 3

        self._feature_names = [f"road_{i}_density" for i in range(n_roads)]
        self._feature_names.extend(["avg_speed", "total_vehicles", "avg_delay"])

        self._weights = self._rng.random(
            (self.grid_width, self.grid_height, self._feature_dim)
        )

        if self.pretrained_path:
            self._load_pretrained()

        self._initialized = True

    def _load_pretrained(self) -> bool:
        """Try to load pretrained weights from file."""
        import os

        if not self.pretrained_path or not os.path.exists(self.pretrained_path):
            return False

        try:
            data = np.load(self.pretrained_path, allow_pickle=True)

            if "weights" in data:
                loaded_weights = data["weights"]
                if loaded_weights.shape == self._weights.shape:
                    self._weights = loaded_weights
                else:
                    import logging
                    logging.warning(
                        f"SOM weight shape mismatch: expected {self._weights.shape}, "
                        f"got {loaded_weights.shape}"
                    )
                    return False

            if "neuron_labels" in data:
                self._neuron_labels = dict(data["neuron_labels"].item())

            return True

        except Exception as e:
            import logging
            logging.warning(f"Failed to load SOM weights: {e}")
            return False

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update SOM classification."""
        if not self.enabled or not self._initialized or self._weights is None:
            return {}

        features = self._extract_features(state)

        self._current_bmu, self._current_qe = self._find_bmu(features)

        self._current_pattern = self._classify_bmu(self._current_bmu)

        self._metrics.record_step(
            qe=self._current_qe,
            pattern=self._current_pattern,
            bmu=self._current_bmu,
        )

        if self.online_learning:
            self._update_weights(features, self._current_bmu)

        tick = state.current_metrics.tick
        history = self._metrics.classification_history
        pattern_changed = len(history) >= 2 and history[-1] != history[-2]

        if pattern_changed or (tick % 10 == 0):
            get_event_bus().publish(Event(
                type=EventType.PATTERN_RECOGNIZED,
                data={
                    "pattern": self._current_pattern,
                    "bmu": self._current_bmu,
                    "qe": self._current_qe,
                },
                source=self.name,
            ))

        self._update_visualization(state)

        return {
            "pattern": self._current_pattern,
            "bmu": self._current_bmu,
            "quantization_error": self._current_qe,
        }

    def _extract_features(self, state: SimulationState) -> np.ndarray:
        """Extract feature vector from current state."""
        features = []

        if self._network:

            for road in self._network.roads.values():
                features.append(road.get_density())

        features.append(state.current_metrics.average_speed / 5.0)
        features.append(min(1.0, state.current_metrics.total_vehicles / 200.0))
        features.append(min(1.0, state.current_metrics.average_delay / 30.0))

        return np.array(features, dtype=np.float32)

    def _find_bmu(self, features: np.ndarray) -> Tuple[Tuple[int, int], float]:
        """
        Find Best Matching Unit for feature vector.

        Returns:
            Tuple of (bmu_coordinates, quantization_error)
        """
        if self._weights is None:
            return (0, 0), 0.0

        diff = self._weights - features
        distances = np.linalg.norm(diff, axis=2)

        flat_idx = np.argmin(distances)
        bmu = np.unravel_index(flat_idx, distances.shape)
        qe = float(distances[bmu])

        return (int(bmu[0]), int(bmu[1])), qe

    def _update_weights(
        self, features: np.ndarray, bmu: Tuple[int, int]
    ) -> None:
        """Update SOM weights (online learning)."""
        if self._weights is None:
            return

        for i in range(self.grid_width):
            for j in range(self.grid_height):

                grid_dist = (i - bmu[0]) ** 2 + (j - bmu[1]) ** 2

                neighborhood = np.exp(-grid_dist / (2 * self.sigma ** 2))

                self._weights[i, j] += (
                    self.learning_rate * neighborhood *
                    (features - self._weights[i, j])
                )

        self.learning_rate *= self.learning_rate_decay
        self.sigma *= self.sigma_decay

        self.learning_rate = max(0.01, self.learning_rate)
        self.sigma = max(0.5, self.sigma)

    def _classify_bmu(self, bmu: Tuple[int, int]) -> str:
        """Classify pattern based on BMU position."""

        if bmu in self._neuron_labels:
            return self._neuron_labels[bmu]

        x, y = bmu
        mid_x = self.grid_width // 2
        mid_y = self.grid_height // 2

        if x < mid_x and y < mid_y:
            return "free_flow"
        elif x >= mid_x and y < mid_y:
            return "moderate"
        elif x < mid_x and y >= mid_y:
            return "rush_hour"
        else:
            return "incident"

    def calibrate_classification(
        self, labeled_data: List[Tuple[np.ndarray, str]]
    ) -> None:
        """
        Calibrate pattern labels using labeled examples.

        This method assigns pattern labels to neurons based on labeled
        training data. Each labeled example is mapped to its BMU, and
        the majority label for each neuron becomes its label.

        Args:
            labeled_data: List of (feature_vector, label) tuples
        """
        if not labeled_data or self._weights is None:
            return

        neuron_votes: Dict[Tuple[int, int], List[str]] = {}

        for features, label in labeled_data:
            bmu, _ = self._find_bmu(features)
            if bmu not in neuron_votes:
                neuron_votes[bmu] = []
            neuron_votes[bmu].append(label)

        self._neuron_labels = {}
        for bmu, labels in neuron_votes.items():
            most_common = Counter(labels).most_common(1)[0][0]
            self._neuron_labels[bmu] = most_common

        self._propagate_labels()

    def _propagate_labels(self) -> None:
        """Propagate labels to unlabeled neurons via nearest labeled neighbor."""
        if not self._neuron_labels:
            return

        labeled_neurons = list(self._neuron_labels.keys())

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                if (i, j) in self._neuron_labels:
                    continue

                min_dist = float('inf')
                nearest_label = "free_flow"

                for labeled_bmu in labeled_neurons:
                    dist = (i - labeled_bmu[0]) ** 2 + (j - labeled_bmu[1]) ** 2
                    if dist < min_dist:
                        min_dist = dist
                        nearest_label = self._neuron_labels[labeled_bmu]

                self._neuron_labels[(i, j)] = nearest_label

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Compute feature importance as weight variance across map.

        Higher variance = feature differentiates patterns more.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self._weights is None:
            return {}

        importance = {}

        for i, name in enumerate(self._feature_names):
            if i < self._weights.shape[2]:

                variance = float(np.var(self._weights[:, :, i]))
                importance[name] = variance

        max_importance = max(importance.values()) if importance else 1.0
        if max_importance > 0:
            importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def get_u_matrix(self) -> np.ndarray:
        """
        Compute U-matrix for visualization.

        The U-matrix shows distances between neighboring neurons,
        revealing cluster boundaries in the map.

        Returns:
            2D array of average distances to neighbors
        """
        if self._weights is None:
            return np.zeros((self.grid_width, self.grid_height))

        u_matrix = np.zeros((self.grid_width, self.grid_height))

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                neighbors = []

                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < self.grid_width and 0 <= nj < self.grid_height:
                            dist = np.linalg.norm(
                                self._weights[i, j] - self._weights[ni, nj]
                            )
                            neighbors.append(dist)

                if neighbors:
                    u_matrix[i, j] = np.mean(neighbors)

        return u_matrix

    def _update_visualization(self, _state: SimulationState) -> None:
        """Update visualization data for SOM overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network or self._weights is None:
            return

        bounds = self._network.get_bounds()
        self._visualization.heatmap_bounds = bounds

        pattern_colors = {
            "free_flow": Colors.CONGESTION_LOW,
            "moderate": Colors.CONGESTION_MEDIUM,
            "rush_hour": Colors.CONGESTION_HIGH,
            "incident": (200, 50, 200),
        }

        color = pattern_colors.get(self._current_pattern, Colors.SOM_COLOR)

        for road_id in self._network.roads:
            self._visualization.road_colors[road_id] = color

        center_x = (bounds[0] + bounds[2]) / 2
        center_y = bounds[1] + 30
        qe_str = f" (QE: {self._current_qe:.2f})"
        self._visualization.annotations.append(
            (center_x, center_y, f"Pattern: {self._current_pattern}{qe_str}")
        )

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get SOM visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get SOM-specific metrics."""
        metrics = {
            "current_bmu_x": float(self._current_bmu[0]),
            "current_bmu_y": float(self._current_bmu[1]),
            "quantization_error": self._current_qe,
            "mean_qe_last_100": self._metrics.get_mean_qe(100),
            "learning_rate": self.learning_rate,
            "sigma": self.sigma,
        }

        for pattern in self._pattern_labels.values():
            count = self._metrics.classification_history.count(pattern)
            metrics[f"pattern_{pattern}"] = float(count)

        pattern_id_map = {v: k for k, v in self._pattern_labels.items()}
        metrics["current_pattern_id"] = float(
            pattern_id_map.get(self._current_pattern, -1)
        )

        return metrics

    def reset(self) -> None:
        """Reset SOM algorithm."""
        super().reset()
        self._current_pattern = "free_flow"
        self._current_bmu = (0, 0)
        self._current_qe = 0.0
        self._metrics = SOMMetrics()

        self.learning_rate = self.config.get("learning_rate", 0.5)
        self.sigma = self.config.get("sigma", 3.0)

    def save_weights(self, path: str) -> None:
        """Save current SOM weights and labels to file."""
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        np.savez(
            path,
            weights=self._weights,
            neuron_labels=self._neuron_labels,
            feature_names=self._feature_names,
        )
