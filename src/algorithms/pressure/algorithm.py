"""
Pressure-Based Traffic Signal Control Algorithm.

Implements queue pressure-based signal control that adapts signal timing
based on the differential pressure between competing traffic streams.

Features:
- Queue pressure calculation (queue + weighted average)
- Hysteresis to prevent rapid switching
- Starvation prevention (maximum wait time)
- Gridlock detection with all-red clearance
- Overload handling (simultaneous green when safe)

References:
    [1] Varaiya, P. (2013). Max pressure control of a network of signalized
        intersections. Transportation Research Part C.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import logging
import numpy as np

from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.entities.traffic_light import Direction
from src.core.state import SimulationState

logger = logging.getLogger(__name__)

@dataclass
class IntersectionState:
    """Per-intersection state tracking for pressure control."""

    last_switch: int = 0
    last_throughput: int = 0
    staleness: int = 0
    all_red_until: int = 0
    last_granted: Optional[str] = None
    wait_ticks: Dict[str, int] = field(default_factory=lambda: {"ns": 0, "ew": 0})

class PressureBasedAlgorithm(BaseAlgorithm):
    """
    Pressure-Based Traffic Signal Control.

    Uses queue pressure differential to adaptively control traffic signals.
    Pressure is computed as: queue + (0.5 * average_queue)

    The algorithm grants green to the direction with higher pressure,
    with hysteresis to prevent oscillation and starvation prevention
    to ensure fairness.
    """

    @property
    def name(self) -> str:
        return "pressure"

    @property
    def display_name(self) -> str:
        return "Pressure-Based Control"

    @property
    def color(self) -> Tuple[int, int, int]:
        return (100, 200, 200)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.overload_threshold = config.get("overload_threshold", 20)
        self.both_overload_threshold = config.get("both_overload_threshold", 15)
        self.stall_staleness = config.get("stall_staleness", 40)
        self.all_red_clear_ticks = config.get("all_red_clear_ticks", 3)
        self.hysteresis_delta = config.get("hysteresis_delta", 3)
        self.starvation_ticks = config.get("starvation_ticks", 120)

        self._intersection_states: Dict[int, IntersectionState] = {}

        self._total_reward = 0.0
        self._tick = 0
        self._reward_history: deque = deque(maxlen=1000)
        self._pressure_history: deque = deque(maxlen=1000)

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize state tracking for each intersection."""
        self._network = network
        self._intersection_states.clear()

        for int_id in network.intersections:
            self._intersection_states[int_id] = IntersectionState()

            network.intersections[int_id].traffic_light.sotl_controlled = True

        self._initialized = True

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

    def _get_queue_ns(self, intersection) -> int:
        """Get total NS queue for intersection."""
        queues = intersection.get_queue_lengths()
        return queues.get(Direction.NORTH, 0) + queues.get(Direction.SOUTH, 0)

    def _get_queue_ew(self, intersection) -> int:
        """Get total EW queue for intersection."""
        queues = intersection.get_queue_lengths()
        return queues.get(Direction.EAST, 0) + queues.get(Direction.WEST, 0)

    def _is_ns_green(self, intersection) -> bool:
        """Check if NS direction has green."""
        tl = intersection.traffic_light
        return Direction.NORTH in tl.current_phase.green_directions

    def _is_ew_green(self, intersection) -> bool:
        """Check if EW direction has green."""
        tl = intersection.traffic_light
        return Direction.EAST in tl.current_phase.green_directions

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update pressure-based signal control."""
        if not self.enabled or not self._initialized:
            return {}

        self._tick += 1
        tick_reward = 0.0
        total_pressure = 0.0

        self._update_queue_counts(state)

        for int_id, intersection in self._network.intersections.items():

            int_state = self._intersection_states.get(int_id)
            if int_state is None:
                int_state = IntersectionState()
                self._intersection_states[int_id] = int_state

            ns_queue = self._get_queue_ns(intersection)
            ew_queue = self._get_queue_ew(intersection)

            ns_avg = ns_queue * 0.8
            ew_avg = ew_queue * 0.8

            ns_green = self._is_ns_green(intersection)
            ew_green = self._is_ew_green(intersection)

            time_in_phase = intersection.traffic_light.get_time_in_phase()

            current_throughput = intersection.get_total_queue()
            if current_throughput < int_state.last_throughput:

                int_state.last_throughput = current_throughput
                int_state.staleness = 0
            else:
                int_state.staleness += 1

            if ns_green:
                int_state.wait_ticks["ns"] = 0
                int_state.wait_ticks["ew"] += 1
            elif ew_green:
                int_state.wait_ticks["ew"] = 0
                int_state.wait_ticks["ns"] += 1
            else:

                int_state.wait_ticks["ns"] += 1
                int_state.wait_ticks["ew"] += 1

            if (
                ns_queue >= self.both_overload_threshold
                and ew_queue >= self.both_overload_threshold
                and int_state.staleness >= self.stall_staleness
            ):

                int_state.all_red_until = max(
                    int_state.all_red_until, self._tick + self.all_red_clear_ticks
                )

                intersection.traffic_light.sotl_controlled = True
                if ns_green:
                    intersection.traffic_light.request_phase_change()
                if ew_green:
                    intersection.traffic_light.request_phase_change()
                int_state.staleness = 0
                logger.debug(f"Intersection {int_id}: Gridlock detected, all-red clearance")

            if int_state.all_red_until > self._tick:
                tick_reward += -ns_queue - 0.5 * ew_queue
                continue
            else:
                int_state.all_red_until = 0

            pressure = (ns_queue + 0.5 * ns_avg) - (ew_queue + 0.5 * ew_avg)
            total_pressure += abs(pressure)

            if int_state.wait_ticks["ns"] >= self.starvation_ticks and ns_queue > 0:
                desired_ns = True
                desired_ew = False
            elif int_state.wait_ticks["ew"] >= self.starvation_ticks and ew_queue > 0:
                desired_ns = False
                desired_ew = True
            else:

                if abs(pressure) <= self.hysteresis_delta:

                    if ns_green or ew_green:
                        desired_ns = ns_green
                        desired_ew = ew_green
                    else:

                        desired_ns = ns_queue >= ew_queue
                        desired_ew = ew_queue > ns_queue
                else:

                    if pressure > 0:
                        desired_ns = True
                        desired_ew = False
                    else:
                        desired_ns = False
                        desired_ew = True

                if ns_queue >= self.overload_threshold and ew_queue < (ns_queue * 0.2):

                    desired_ns = True
                    desired_ew = True
                elif ew_queue >= self.overload_threshold and ns_queue < (ew_queue * 0.2):

                    desired_ns = True
                    desired_ew = True

            intersection.traffic_light.sotl_controlled = True

            if desired_ns != ns_green:
                intersection.traffic_light.request_phase_change()
                int_state.last_switch = self._tick
            elif desired_ew != ew_green:

                intersection.traffic_light.request_phase_change()
                int_state.last_switch = self._tick

            reward = -ns_queue - 0.5 * ew_queue

            if desired_ns != ns_green or desired_ew != ew_green:
                reward -= 0.5

            if ns_queue > self.overload_threshold:
                reward -= (ns_queue - self.overload_threshold) * 0.2
            if ew_queue > self.overload_threshold:
                reward -= (ew_queue - self.overload_threshold) * 0.2

            tick_reward += reward

        self._total_reward += tick_reward
        self._reward_history.append(tick_reward)
        n_intersections = len(self._network.intersections)
        self._pressure_history.append(total_pressure / max(1, n_intersections))

        self._update_visualization()

        return {
            "total_reward": tick_reward,
            "cumulative_reward": self._total_reward,
            "avg_pressure": total_pressure / max(1, n_intersections),
        }

    def _update_visualization(self) -> None:
        """Update visualization data."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        for int_id, intersection in self._network.intersections.items():
            ns_queue = self._get_queue_ns(intersection)
            ew_queue = self._get_queue_ew(intersection)

            pressure = (ns_queue - ew_queue) / max(1, ns_queue + ew_queue)

            if pressure > 0.2:

                color = (200, 150, 100)
            elif pressure < -0.2:

                color = (100, 150, 200)
            else:

                color = self.color

            intensity = min(1.0, (ns_queue + ew_queue) / 20.0)

            self._visualization.intersection_values[int_id] = intensity
            self._visualization.intersection_colors[int_id] = color

            pos = intersection.position
            self._visualization.points.append(pos)
            self._visualization.point_colors.append(self.color)
            self._visualization.point_sizes.append(6.0 + intensity * 6)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get algorithm-specific metrics."""
        avg_reward = 0.0
        if self._reward_history:
            avg_reward = np.mean(list(self._reward_history)[-100:])

        avg_pressure = 0.0
        if self._pressure_history:
            avg_pressure = np.mean(list(self._pressure_history)[-100:])

        return {
            "cumulative_reward": self._total_reward,
            "avg_reward": avg_reward,
            "avg_pressure": avg_pressure,
            "overload_threshold": float(self.overload_threshold),
            "starvation_ticks": float(self.starvation_ticks),
        }

    def reset(self) -> None:
        """Reset algorithm state."""
        super().reset()

        self._intersection_states.clear()
        if self._network:
            for int_id in self._network.intersections:
                self._intersection_states[int_id] = IntersectionState()

        self._total_reward = 0.0
        self._tick = 0
        self._reward_history.clear()
        self._pressure_history.clear()
