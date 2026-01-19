"""Self-Organizing Traffic Lights (SOTL) algorithm."""

from typing import Dict, Any, Tuple, List, Callable
from dataclasses import dataclass, field
from config.colors import Colors
from config.constants import SOTL_PSO_SMOOTHING_FACTOR, SOTL_SUGGESTION_COOLDOWN
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.entities.intersection import Intersection
from src.entities.traffic_light import Direction
from src.core.state import SimulationState
from src.core.event_bus import Event, EventType, EventHandler, get_event_bus

@dataclass
class IntersectionController:
    """SOTL controller for a single intersection."""

    intersection: Intersection
    theta: int = 5
    mu: int = 3
    omega: int = 15
    min_green: int = 10
    max_green: int = 60

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

        if light.is_yellow:
            return False

        if time_in_phase < self.min_green:
            return False

        if time_in_phase >= self.max_green:
            self._request_change()
            return True

        red_queue = 0
        green_queue = 0

        for direction in Direction:
            queue_len = self.intersection.get_queue_length(direction)
            if direction in current_phase.green_directions:
                green_queue += queue_len
            else:
                red_queue += queue_len

        if red_queue >= self.theta and green_queue < self.mu:
            self._request_change()
            return True

        if green_queue == 0:
            self._gap_timer += 1
            if self._gap_timer >= self.omega:
                self._gap_timer = 0
                self._request_change()
                return True
        else:
            self._gap_timer = 0

        if green_queue >= self.mu:
            return False

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

        self.theta = config.get("theta", 5)
        self.mu = config.get("mu", 3)
        self.omega = config.get("omega", 15)
        self.min_green = config.get("min_green_time", 10)
        self.max_green = config.get("max_green_time", 60)

        self._controllers: Dict[int, IntersectionController] = {}

        self._total_phase_changes = 0
        self._average_green_time = 0.0

        self._subscribed = False
        self._event_handlers: Dict[EventType, EventHandler] = {}
        self._timing_suggestions_received = 0
        self._timing_suggestions_applied = 0
        self._current_tick = 0
        self._last_suggestion_tick: Dict[int, int] = {}

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize SOTL controllers for all intersections."""
        self._network = network
        self._controllers.clear()

        self._base_min_green = self.config.get("min_green_time", 10)
        self._base_max_green = self.config.get("max_green_time", 60)
        self._base_theta = self.config.get("theta", 5)

        for int_id, intersection in network.intersections.items():

            intersection.enable_sotl(True)

            controller = IntersectionController(
                intersection=intersection,
                theta=self.theta,
                mu=self.mu,
                omega=self.omega,
                min_green=self.min_green,
                max_green=self.max_green,
            )
            self._controllers[int_id] = controller

        self._subscribe_events()

        self._initialized = True

    def _subscribe_events(self) -> None:
        """Subscribe to relevant events."""
        if self._subscribed:
            return

        event_bus = get_event_bus()
        self._event_handlers = {
            EventType.SIGNAL_TIMING_SUGGESTED: self._on_timing_suggested,
            EventType.PATTERN_RECOGNIZED: self._on_pattern_recognized,
        }
        for event_type, handler in self._event_handlers.items():
            event_bus.subscribe(event_type, handler)
        self._subscribed = True

    def _unsubscribe_events(self) -> None:
        """Unsubscribe from all events."""
        event_bus = get_event_bus()
        for event_type, handler in self._event_handlers.items():
            event_bus.unsubscribe(event_type, handler)
        self._event_handlers.clear()
        self._subscribed = False

    def _on_timing_suggested(self, event: Event) -> None:
        """
        Apply PSO timing suggestion with smoothing.

        Uses smoothing factor from config to prevent oscillation.
        Includes per-intersection cooldown for rate limiting.
        """
        if not self.enabled:
            return

        self._timing_suggestions_received += 1

        int_id = event.data.get("intersection_id")
        if int_id not in self._controllers:
            return

        last_tick = self._last_suggestion_tick.get(int_id, 0)
        if self._current_tick - last_tick < SOTL_SUGGESTION_COOLDOWN:
            return

        controller = self._controllers[int_id]
        new_min = event.data.get("min_green", controller.min_green)
        new_max = event.data.get("max_green", controller.max_green)

        smoothing_new = SOTL_PSO_SMOOTHING_FACTOR
        smoothing_old = 1.0 - smoothing_new
        controller.min_green = round(smoothing_old * controller.min_green + smoothing_new * new_min)
        controller.max_green = round(smoothing_old * controller.max_green + smoothing_new * new_max)

        if controller.min_green > controller.max_green:
            controller.min_green = max(5, controller.max_green - 10)

        self._timing_suggestions_applied += 1
        self._last_suggestion_tick[int_id] = self._current_tick

    def _on_pattern_recognized(self, event: Event) -> None:
        """Adjust signal parameters based on SOM pattern."""
        if not self.enabled:
            return

        pattern = event.data.get("pattern")

        if pattern == "rush_hour":

            min_mult, max_mult, theta_mult = 1.2, 1.3, 0.8
        elif pattern == "incident":

            min_mult, max_mult, theta_mult = 1.3, 1.0, 1.2
        elif pattern == "free_flow":

            min_mult, max_mult, theta_mult = 0.9, 0.9, 1.0
        else:

            min_mult, max_mult, theta_mult = 1.0, 1.0, 1.0

        smoothing = 0.3
        for controller in self._controllers.values():
            target_min = self._base_min_green * min_mult
            target_max = self._base_max_green * max_mult
            target_theta = self._base_theta * theta_mult

            controller.min_green = round((1 - smoothing) * controller.min_green + smoothing * target_min)
            controller.max_green = round((1 - smoothing) * controller.max_green + smoothing * target_max)
            controller.theta = round((1 - smoothing) * controller.theta + smoothing * target_theta)

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update all SOTL controllers."""
        if not self.enabled or not self._initialized:
            return {}

        tick = state.current_metrics.tick
        self._current_tick = tick
        phase_changes = 0

        self._update_queue_counts(state)

        for controller in self._controllers.values():
            if controller.update(tick):
                phase_changes += 1

                get_event_bus().publish(Event(
                    type=EventType.SIGNAL_PHASE_CHANGED,
                    data={
                        "intersection_id": controller.intersection.id,
                        "new_phase": controller.intersection.traffic_light.current_phase_idx,
                    },
                    source=self.name,
                ))

        self._total_phase_changes += phase_changes

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

            total_queue = intersection.get_total_queue()
            pressure = min(1.0, total_queue / (self.theta * 4))

            self._visualization.intersection_values[int_id] = pressure
            self._visualization.intersection_colors[int_id] = Colors.lerp(
                Colors.SOTL_COLOR,
                Colors.CONGESTION_HIGH,
                pressure
            )

            phase_info = intersection.get_phase_info()
            phase_text = f"Q:{total_queue}"
            pos = intersection.position
            self._visualization.annotations.append((pos[0], pos[1] - 20, phase_text))

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get SOTL visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get SOTL-specific metrics."""

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
            "pso_suggestions_received": float(self._timing_suggestions_received),
            "pso_suggestions_applied": float(self._timing_suggestions_applied),
        }

    def reset(self) -> None:
        """Reset SOTL algorithm."""
        self._unsubscribe_events()
        super().reset()
        for controller in self._controllers.values():
            controller.reset()
        self._total_phase_changes = 0
        self._timing_suggestions_received = 0
        self._timing_suggestions_applied = 0
        self._current_tick = 0
        self._last_suggestion_tick.clear()

    def toggle(self, enabled: bool) -> None:
        """Enable/disable SOTL control."""
        was_enabled = self.enabled
        super().toggle(enabled)

        if enabled and not was_enabled:
            self._subscribe_events()
        elif not enabled and was_enabled:
            self._unsubscribe_events()

        if self._network:
            for intersection in self._network.intersections.values():
                intersection.enable_sotl(enabled)
