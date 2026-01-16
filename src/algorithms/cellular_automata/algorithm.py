"""Cellular Automata (Nagel-Schreckenberg) traffic flow algorithm."""

from typing import Dict, Any, Tuple
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState


class CellularAutomataAlgorithm(BaseAlgorithm):
    """
    Cellular Automata algorithm implementing NaSch rules.

    This algorithm controls the fundamental traffic flow behavior:
    - Acceleration
    - Braking (based on gap to vehicle ahead)
    - Random slowdown (stochastic element)
    - Vehicle movement

    The actual vehicle updates happen in Vehicle.update(), but this
    algorithm provides the parameters and tracks CA-specific metrics.
    """

    @property
    def name(self) -> str:
        return "cellular_automata"

    @property
    def display_name(self) -> str:
        return "Cellular Automata"

    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.CA_COLOR

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)
        self.max_velocity = config.get("max_velocity", 5)
        self.slow_probability = config.get("slow_probability", 0.3)
        self.lane_change_threshold = config.get("lane_change_threshold", 2)

        # Metrics
        self._total_moves = 0
        self._total_stops = 0
        self._average_speed = 0.0
        self._flow_rate = 0.0

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize CA algorithm."""
        self._network = network
        self._initialized = True
        self._reset_metrics()

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """
        Update CA metrics (actual movement is in Vehicle.update).

        The CA rules are implemented in Vehicle._apply_nasch_rules().
        This method tracks aggregate metrics.
        """
        if not self.enabled or not self._initialized:
            return {}

        # Calculate metrics from current vehicle states
        vehicles = list(state.vehicles.values())
        if vehicles:
            speeds = [v.speed for v in vehicles]
            self._average_speed = sum(speeds) / len(speeds)

            moving = sum(1 for v in vehicles if v.speed > 0)
            self._total_moves += moving
            self._total_stops += len(vehicles) - moving

            # Flow rate (vehicles per unit time)
            self._flow_rate = moving / len(vehicles) if vehicles else 0

        # Update visualization data
        self._update_visualization(state)

        return {
            "average_speed": self._average_speed,
            "flow_rate": self._flow_rate,
        }

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for CA overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        # Color roads by density (red = congested)
        for road_id, road in self._network.roads.items():
            density = road.get_density()
            self._visualization.road_values[road_id] = density

            # Lerp from green (free flow) to red (congested)
            color = Colors.lerp(
                Colors.CONGESTION_LOW,
                Colors.CONGESTION_HIGH,
                density
            )
            self._visualization.road_colors[road_id] = color

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get CA visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get CA-specific metrics."""
        return {
            "avg_speed": self._average_speed,
            "flow_rate": self._flow_rate,
            "max_velocity": float(self.max_velocity),
            "slow_prob": self.slow_probability,
        }

    def _reset_metrics(self) -> None:
        """Reset metrics."""
        self._total_moves = 0
        self._total_stops = 0
        self._average_speed = 0.0
        self._flow_rate = 0.0

    def reset(self) -> None:
        """Reset algorithm."""
        super().reset()
        self._reset_metrics()
