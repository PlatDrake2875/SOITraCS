"""Traffic light and signal phase management."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Set, Tuple


class SignalState(Enum):
    """Traffic signal states."""
    RED = auto()
    YELLOW = auto()
    GREEN = auto()


class Direction(Enum):
    """Approach directions to intersection."""
    NORTH = "north"
    SOUTH = "south"
    EAST = "east"
    WEST = "west"

    @classmethod
    def opposite(cls, direction: "Direction") -> "Direction":
        """Get opposite direction."""
        opposites = {
            cls.NORTH: cls.SOUTH,
            cls.SOUTH: cls.NORTH,
            cls.EAST: cls.WEST,
            cls.WEST: cls.EAST,
        }
        return opposites[direction]

    @classmethod
    def from_string(cls, s: str) -> "Direction":
        """Create from string."""
        return cls(s.lower())


@dataclass
class SignalPhase:
    """A signal phase defining which directions have green."""

    green_directions: Set[Direction]
    duration: int  # Ticks
    yellow_duration: int = 3  # Ticks for yellow transition

    @classmethod
    def from_dict(cls, data: dict) -> "SignalPhase":
        """Create from dictionary (YAML parsing)."""
        directions = {Direction.from_string(d) for d in data.get("green_directions", [])}
        return cls(
            green_directions=directions,
            duration=data.get("duration", 30),
            yellow_duration=data.get("yellow_duration", 3),
        )


@dataclass
class TrafficLight:
    """
    Traffic light controller for an intersection.

    Manages signal phases and provides current state for each direction.
    """

    intersection_id: int
    phases: List[SignalPhase] = field(default_factory=list)

    # State
    current_phase_idx: int = field(default=0, init=False)
    phase_timer: int = field(default=0, init=False)
    is_yellow: bool = field(default=False, init=False)
    yellow_timer: int = field(default=0, init=False)

    # SOTL control (can be updated by algorithm)
    sotl_controlled: bool = False
    force_phase_change: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Set up default phases if none provided."""
        if not self.phases:
            # Default: NS/EW alternating with 30-tick phases
            self.phases = [
                SignalPhase(green_directions={Direction.NORTH, Direction.SOUTH}, duration=30),
                SignalPhase(green_directions={Direction.EAST, Direction.WEST}, duration=30),
            ]

    @property
    def current_phase(self) -> SignalPhase:
        """Get current signal phase."""
        return self.phases[self.current_phase_idx]

    def get_state(self, direction: Direction) -> SignalState:
        """Get signal state for a direction."""
        if self.is_yellow:
            # Yellow for directions that were green
            prev_phase = self.phases[(self.current_phase_idx - 1) % len(self.phases)]
            if direction in prev_phase.green_directions:
                return SignalState.YELLOW
            return SignalState.RED

        if direction in self.current_phase.green_directions:
            return SignalState.GREEN
        return SignalState.RED

    def is_green(self, direction: Direction) -> bool:
        """Check if direction has green light."""
        return self.get_state(direction) == SignalState.GREEN

    def update(self, dt: float = 1.0) -> bool:
        """
        Update signal timing.

        Returns:
            True if phase changed
        """
        if not self.phases:
            return False

        phase_changed = False

        if self.is_yellow:
            self.yellow_timer += 1
            if self.yellow_timer >= self.current_phase.yellow_duration:
                self.is_yellow = False
                self.yellow_timer = 0
                phase_changed = True
        else:
            self.phase_timer += 1

            # Check for phase change
            should_change = self.phase_timer >= self.current_phase.duration
            if self.sotl_controlled and self.force_phase_change:
                should_change = True
                self.force_phase_change = False

            if should_change:
                self._start_yellow_transition()

        return phase_changed

    def _start_yellow_transition(self) -> None:
        """Start transition to next phase with yellow light."""
        self.is_yellow = True
        self.yellow_timer = 0
        # Advance to next phase
        self.current_phase_idx = (self.current_phase_idx + 1) % len(self.phases)
        self.phase_timer = 0

    def request_phase_change(self) -> None:
        """Request a phase change (for SOTL control)."""
        if self.sotl_controlled:
            self.force_phase_change = True

    def set_phase(self, phase_idx: int) -> None:
        """Force a specific phase (with yellow transition)."""
        if 0 <= phase_idx < len(self.phases) and phase_idx != self.current_phase_idx:
            self._start_yellow_transition()
            # Override to target phase
            self.current_phase_idx = phase_idx

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_phase_idx = 0
        self.phase_timer = 0
        self.is_yellow = False
        self.yellow_timer = 0
        self.force_phase_change = False

    def get_time_in_phase(self) -> int:
        """Get time spent in current phase."""
        return self.phase_timer

    def get_time_to_change(self) -> int:
        """Get estimated time until next phase change."""
        if self.is_yellow:
            return self.current_phase.yellow_duration - self.yellow_timer
        return self.current_phase.duration - self.phase_timer
