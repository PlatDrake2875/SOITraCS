"""Intersection entity with traffic signal control."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from enum import Enum

from .traffic_light import TrafficLight, SignalPhase, Direction, SignalState

if TYPE_CHECKING:
    from .road import Road


@dataclass
class Intersection:
    """
    An intersection where roads meet.

    Manages traffic signals and tracks queue lengths for each approach.
    """

    id: int
    position: Tuple[float, float]  # (x, y) in pixels

    # Traffic signal
    traffic_light: TrafficLight = field(default=None, init=False)

    # Connected roads (by direction)
    incoming_roads: Dict[Direction, int] = field(default_factory=dict)  # direction -> road_id
    outgoing_roads: Dict[Direction, int] = field(default_factory=dict)  # direction -> road_id

    # Queue tracking
    _queue_counts: Dict[Direction, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize traffic light."""
        if self.traffic_light is None:
            self.traffic_light = TrafficLight(intersection_id=self.id)

    def set_signal_phases(self, phases: List[SignalPhase]) -> None:
        """Set signal phases for this intersection."""
        self.traffic_light.phases = phases
        self.traffic_light.reset()

    def connect_incoming(self, direction: Direction, road_id: int) -> None:
        """Connect an incoming road from a direction."""
        self.incoming_roads[direction] = road_id

    def connect_outgoing(self, direction: Direction, road_id: int) -> None:
        """Connect an outgoing road in a direction."""
        self.outgoing_roads[direction] = road_id

    def get_incoming_road(self, direction: Direction) -> int | None:
        """Get incoming road ID for a direction."""
        return self.incoming_roads.get(direction)

    def get_outgoing_road(self, direction: Direction) -> int | None:
        """Get outgoing road ID for a direction."""
        return self.outgoing_roads.get(direction)

    def can_proceed(self, from_direction: Direction) -> bool:
        """Check if traffic from a direction can proceed."""
        return self.traffic_light.is_green(from_direction)

    def get_signal_state(self, direction: Direction) -> SignalState:
        """Get signal state for a direction."""
        return self.traffic_light.get_state(direction)

    def update(self, dt: float = 1.0) -> bool:
        """
        Update intersection state.

        Returns:
            True if signal phase changed
        """
        return self.traffic_light.update(dt)

    def update_queue_count(self, direction: Direction, count: int) -> None:
        """Update queue count for a direction."""
        self._queue_counts[direction] = count

    def get_queue_length(self, direction: Direction) -> int:
        """Get queue length for a direction."""
        return self._queue_counts.get(direction, 0)

    def get_queue_lengths(self) -> Dict[Direction, int]:
        """Get all queue lengths."""
        return dict(self._queue_counts)

    def get_total_queue(self) -> int:
        """Get total queue across all directions."""
        return sum(self._queue_counts.values())

    def get_waiting_vehicles(self, direction: Direction) -> int:
        """Get count of waiting vehicles (alias for queue length)."""
        return self.get_queue_length(direction)

    def get_phase_info(self) -> Dict:
        """Get current phase information for display."""
        return {
            "phase_index": self.traffic_light.current_phase_idx,
            "green_directions": [d.value for d in self.traffic_light.current_phase.green_directions],
            "time_in_phase": self.traffic_light.get_time_in_phase(),
            "time_to_change": self.traffic_light.get_time_to_change(),
            "is_yellow": self.traffic_light.is_yellow,
        }

    def enable_sotl(self, enabled: bool = True) -> None:
        """Enable/disable SOTL control for this intersection."""
        self.traffic_light.sotl_controlled = enabled

    def request_phase_change(self) -> None:
        """Request a phase change (for SOTL)."""
        self.traffic_light.request_phase_change()

    def reset(self) -> None:
        """Reset intersection to initial state."""
        self.traffic_light.reset()
        self._queue_counts.clear()

    @classmethod
    def from_dict(cls, data: dict) -> "Intersection":
        """Create intersection from dictionary (YAML parsing)."""
        intersection = cls(
            id=data["id"],
            position=tuple(data["position"]),
        )

        # Parse signal phases
        if "signal_phases" in data:
            phases = [SignalPhase.from_dict(p) for p in data["signal_phases"]]
            intersection.set_signal_phases(phases)

        return intersection
