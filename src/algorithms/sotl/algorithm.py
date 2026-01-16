"""Self-Organizing Traffic Lights (SOTL) algorithm."""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.entities.intersection import Intersection
from src.entities.traffic_light import Direction
from src.core.state import SimulationState
from src.core.event_bus import Event, EventType, get_event_bus


@dataclass
class IntersectionController:
    """SOTL controller for a single intersection."""

    intersection: Intersection
    theta: int = 5       # Queue threshold
    mu: int = 3          # Platoon detection threshold
    omega: int = 15      # Gap-out timer
    min_green: int = 10  # Minimum green time
    max_green: int = 60  # Maximum green time

    # State
    _gap_timer: int = field(default=0, init=False)
    _last_arrival_tick: int = field(default=0, init=False)
    _phase_changes: int = field(default=0, init=False)

    def update(self, tick: int) -> bool:
        """
        Apply SOTL rules to decide phase changes.

        Returns True if phase was changed.
        """
        light = self.intersection.traffic_light
        current_phase = light.current_phase
        time_in_phase = light.get_time_in_phase()

        # Don't change during yellow
        if light.is_yellow:
            return False

        # Minimum green time check
        if time_in_phase < self.min_green:
            return False

        # Maximum green time - force change
        if time_in_phase >= self.max_green:
            self._request_change()
            return True

        # Get queue lengths for red directions
        red_queue = 0
        green_queue = 0

        for direction in Direction:
            queue_len = self.intersection.get_queue_length(direction)
            if direction in current_phase.green_directions:
                green_queue += queue_len
            else:
                red_queue += queue_len

        # Rule 1: Queue threshold (theta)
        # Switch if red queue exceeds threshold and green queue is low
        if red_queue >= self.theta and green_queue < self.mu:
            self._request_change()
            return True

        # Rule 2: Gap-out (omega)
        # If no vehicles detected for omega ticks, switch
        if green_queue == 0:
            self._gap_timer += 1
            if self._gap_timer >= self.omega:
                self._gap_timer = 0
                self._request_change()
                return True
        else:
            self._gap_timer = 0

        # Rule 3: Platoon detection (mu)
        # Keep green if platoon is arriving (high green queue)
        if green_queue >= self.mu:
            return False  # Keep current phase for platoon

        return False

    def _request_change(self) -> None:
        """Request phase change."""
        self.intersection.request_phase_change()
        self._phase_changes += 1

    @property
    def phase_changes(self) -> int:
        """Get number of phase changes."""
        return self._phase_changes

    def reset(self) -> None:
        """Reset controller state."""
        self._gap_timer = 0
        self._last_arrival_tick = 0
        self._phase_changes = 0


class SOTLAlgorithm(BaseAlgorithm):
    """
    Self-Organizing Traffic Lights algorithm.

    Implements decentralized traffic signal control where each
    intersection makes local decisions based on:
    - Queue lengths at red approaches (theta threshold)
    - Platoon detection at green approaches (mu threshold)
    - Gap-out timing (omega timer)
    """

    @property
    def name(self) -> str:
        return "sotl"

    @property
    def display_name(self) -> str:
        return "Self-Organizing Lights"

    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.SOTL_COLOR

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # SOTL parameters from config
        self.theta = config.get("theta", 5)
        self.mu = config.get("mu", 3)
        self.omega = config.get("omega", 15)
        self.min_green = config.get("min_green_time", 10)
        self.max_green = config.get("max_green_time", 60)

        # Controllers for each intersection
        self._controllers: Dict[int, IntersectionController] = {}

        # Metrics
        self._total_phase_changes = 0
        self._average_green_time = 0.0

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize SOTL controllers for all intersections."""
        self._network = network
        self._controllers.clear()

        for int_id, intersection in network.intersections.items():
            # Enable SOTL control for this intersection
            intersection.enable_sotl(True)

            # Create controller
            controller = IntersectionController(
                intersection=intersection,
                theta=self.theta,
                mu=self.mu,
                omega=self.omega,
                min_green=self.min_green,
                max_green=self.max_green,
            )
            self._controllers[int_id] = controller

        self._initialized = True

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update all SOTL controllers."""
        if not self.enabled or not self._initialized:
            return {}

        tick = state.current_metrics.tick
        phase_changes = 0

        # First, update queue counts from actual vehicle positions
        self._update_queue_counts(state)

        # Then update each controller
        for controller in self._controllers.values():
            if controller.update(tick):
                phase_changes += 1

                # Publish event
                get_event_bus().publish(Event(
                    type=EventType.SIGNAL_PHASE_CHANGED,
                    data={
                        "intersection_id": controller.intersection.id,
                        "new_phase": controller.intersection.traffic_light.current_phase_idx,
                    },
                    source=self.name,
                ))

        self._total_phase_changes += phase_changes

        # Update visualization
        self._update_visualization(state)

        return {
            "phase_changes": phase_changes,
            "total_phase_changes": self._total_phase_changes,
        }

    def _update_queue_counts(self, state: SimulationState) -> None:
        """Update queue counts at each intersection from vehicle positions."""
        if not self._network:
            return

        for intersection in self._network.intersections.values():
            for direction in Direction:
                road_id = intersection.get_incoming_road(direction)
                if road_id is not None:
                    road = self._network.get_road(road_id)
                    if road:
                        queue_len = road.get_queue_length()
                        intersection.update_queue_count(direction, queue_len)

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for SOTL overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        for int_id, intersection in self._network.intersections.items():
            # Show queue pressure as color intensity
            total_queue = intersection.get_total_queue()
            pressure = min(1.0, total_queue / (self.theta * 4))

            self._visualization.intersection_values[int_id] = pressure
            self._visualization.intersection_colors[int_id] = Colors.lerp(
                Colors.SOTL_COLOR,
                Colors.CONGESTION_HIGH,
                pressure
            )

            # Add annotation showing current phase info
            phase_info = intersection.get_phase_info()
            phase_text = f"Q:{total_queue}"
            pos = intersection.position
            self._visualization.annotations.append((pos[0], pos[1] - 20, phase_text))

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get SOTL visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get SOTL-specific metrics."""
        # Calculate average phase changes per intersection
        avg_changes = 0.0
        if self._controllers:
            total = sum(c.phase_changes for c in self._controllers.values())
            avg_changes = total / len(self._controllers)

        return {
            "total_phase_changes": float(self._total_phase_changes),
            "avg_changes_per_int": avg_changes,
            "theta": float(self.theta),
            "mu": float(self.mu),
            "omega": float(self.omega),
        }

    def reset(self) -> None:
        """Reset SOTL algorithm."""
        super().reset()
        for controller in self._controllers.values():
            controller.reset()
        self._total_phase_changes = 0

    def toggle(self, enabled: bool) -> None:
        """Enable/disable SOTL control."""
        super().toggle(enabled)

        # Update all intersections
        if self._network:
            for intersection in self._network.intersections.values():
                intersection.enable_sotl(enabled)
