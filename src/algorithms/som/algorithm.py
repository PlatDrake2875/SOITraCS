"""Self-Organizing Map (SOM) algorithm for traffic pattern recognition."""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState
from src.core.event_bus import Event, EventType, get_event_bus


class SOMAlgorithm(BaseAlgorithm):
    """
    Self-Organizing Map for traffic pattern recognition.

    Classifies current traffic state into patterns:
    - Free flow
    - Moderate congestion
    - Rush hour
    - Incident

    Can use pre-trained weights or train online.
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

        # SOM parameters
        grid_size = config.get("grid_size", [10, 10])
        self.grid_width = grid_size[0]
        self.grid_height = grid_size[1]
        self.learning_rate = config.get("learning_rate", 0.5)
        self.sigma = config.get("sigma", 3.0)
        self.pretrained_path = config.get("pretrained_path")

        # SOM weights (will be initialized in initialize())
        self._weights: Optional[np.ndarray] = None
        self._feature_dim = 0

        # Pattern labels
        self._pattern_labels = {
            0: "free_flow",
            1: "moderate",
            2: "rush_hour",
            3: "incident",
        }

        # Current classification
        self._current_pattern = "free_flow"
        self._current_bmu = (0, 0)  # Best Matching Unit

        # Metrics
        self._classification_history: List[str] = []

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize SOM weights."""
        self._network = network

        # Feature dimension: one value per road (density) + global metrics
        self._feature_dim = len(network.roads) + 3  # +3 for avg speed, total vehicles, avg delay

        # Initialize weights randomly
        self._weights = np.random.rand(
            self.grid_width, self.grid_height, self._feature_dim
        )

        # Try to load pretrained weights
        if self.pretrained_path:
            import os
            if os.path.exists(self.pretrained_path):
                try:
                    data = np.load(self.pretrained_path, allow_pickle=False)
                    self._weights = data['weights']
                except Exception as e:
                    import logging
                    logging.warning(f"Failed to load SOM weights: {e}")

        self._initialized = True

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update SOM classification."""
        if not self.enabled or not self._initialized or self._weights is None:
            return {}

        # Extract features from current state
        features = self._extract_features(state)

        # Find Best Matching Unit
        self._current_bmu = self._find_bmu(features)

        # Classify pattern based on BMU position
        self._current_pattern = self._classify_bmu(self._current_bmu)

        # Store in history
        self._classification_history.append(self._current_pattern)
        if len(self._classification_history) > 300:
            self._classification_history = self._classification_history[-300:]

        # Publish pattern event if changed
        if len(self._classification_history) >= 2:
            if self._classification_history[-1] != self._classification_history[-2]:
                get_event_bus().publish(Event(
                    type=EventType.PATTERN_RECOGNIZED,
                    data={
                        "pattern": self._current_pattern,
                        "bmu": self._current_bmu,
                    },
                    source=self.name,
                ))

        # Update visualization
        self._update_visualization(state)

        return {
            "pattern": self._current_pattern,
            "bmu": self._current_bmu,
        }

    def _extract_features(self, state: SimulationState) -> np.ndarray:
        """Extract feature vector from current state."""
        features = []

        if self._network:
            # Road densities
            for road in self._network.roads.values():
                features.append(road.get_density())

        # Global metrics
        features.append(state.current_metrics.average_speed / 5.0)  # Normalize
        features.append(min(1.0, state.current_metrics.total_vehicles / 200.0))
        features.append(min(1.0, state.current_metrics.average_delay / 30.0))

        return np.array(features)

    def _find_bmu(self, features: np.ndarray) -> Tuple[int, int]:
        """Find Best Matching Unit for feature vector."""
        if self._weights is None:
            return (0, 0)

        min_dist = float('inf')
        bmu = (0, 0)

        for i in range(self.grid_width):
            for j in range(self.grid_height):
                dist = np.linalg.norm(features - self._weights[i, j])
                if dist < min_dist:
                    min_dist = dist
                    bmu = (i, j)

        return bmu

    def _classify_bmu(self, bmu: Tuple[int, int]) -> str:
        """Classify pattern based on BMU position."""
        # Simple quadrant-based classification
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

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for SOM overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network or self._weights is None:
            return

        # Create U-matrix heatmap
        bounds = self._network.get_bounds()
        self._visualization.heatmap_bounds = bounds

        # Simplified: color roads based on current pattern
        pattern_colors = {
            "free_flow": Colors.CONGESTION_LOW,
            "moderate": Colors.CONGESTION_MEDIUM,
            "rush_hour": Colors.CONGESTION_HIGH,
            "incident": (200, 50, 200),
        }

        color = pattern_colors.get(self._current_pattern, Colors.SOM_COLOR)

        for road_id in self._network.roads:
            self._visualization.road_colors[road_id] = color

        # Add pattern label annotation
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = bounds[1] + 30
        self._visualization.annotations.append(
            (center_x, center_y, f"Pattern: {self._current_pattern}")
        )

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get SOM visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get SOM-specific metrics."""
        # Count pattern occurrences
        pattern_counts = {}
        for pattern in self._pattern_labels.values():
            count = self._classification_history.count(pattern)
            pattern_counts[f"pattern_{pattern}"] = float(count)

        return {
            "current_bmu_x": float(self._current_bmu[0]),
            "current_bmu_y": float(self._current_bmu[1]),
            **pattern_counts,
        }

    def reset(self) -> None:
        """Reset SOM algorithm."""
        super().reset()
        self._current_pattern = "free_flow"
        self._current_bmu = (0, 0)
        self._classification_history.clear()
