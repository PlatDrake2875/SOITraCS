"""Simulation timing and speed control."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List
import time


class SimulationSpeed(Enum):
    """Preset simulation speeds."""
    PAUSED = 0.0
    SLOW = 0.5
    NORMAL = 1.0
    FAST = 2.0
    VERY_FAST = 5.0
    MAX = 10.0


@dataclass
class SimulationClock:
    """
    Manages simulation timing with fixed timestep.

    Separates simulation time from real time to allow speed control.
    """

    ticks_per_second: int = 30  # Simulation ticks per real second at 1x speed
    speed: float = 1.0  # Speed multiplier

    # Internal state
    _tick: int = field(default=0, init=False)
    _accumulated_time: float = field(default=0.0, init=False)
    _last_update: float = field(default_factory=time.time, init=False)
    _paused: bool = field(default=False, init=False)
    _tick_times: List[float] = field(default_factory=list, init=False)

    @property
    def tick(self) -> int:
        """Current simulation tick."""
        return self._tick

    @property
    def dt(self) -> float:
        """Time delta per tick in simulation seconds."""
        return 1.0 / self.ticks_per_second

    @property
    def simulation_time(self) -> float:
        """Total simulation time in seconds."""
        return self._tick * self.dt

    @property
    def is_paused(self) -> bool:
        """Whether simulation is paused."""
        return self._paused

    @property
    def fps(self) -> float:
        """Current frames/ticks per second."""
        if len(self._tick_times) < 2:
            return 0.0
        duration = self._tick_times[-1] - self._tick_times[0]
        if duration <= 0:
            return 0.0
        return (len(self._tick_times) - 1) / duration

    def update(self, real_dt: float) -> int:
        """
        Update clock and return number of simulation ticks to process.

        Args:
            real_dt: Real elapsed time since last update in seconds

        Returns:
            Number of simulation ticks to process this frame
        """
        if self._paused or self.speed <= 0:
            return 0

        # Scale real time by speed multiplier
        scaled_dt = real_dt * self.speed

        # Accumulate time
        self._accumulated_time += scaled_dt

        # Calculate ticks to process
        tick_duration = self.dt
        ticks_to_process = int(self._accumulated_time / tick_duration)

        # Cap ticks to prevent spiral of death
        max_ticks = 10  # Maximum ticks per frame
        ticks_to_process = min(ticks_to_process, max_ticks)

        # Consume processed time
        self._accumulated_time -= ticks_to_process * tick_duration

        # Update tick counter
        self._tick += ticks_to_process

        # Track timing for FPS calculation
        now = time.time()
        self._tick_times.append(now)
        # Keep only last second of timing data
        cutoff = now - 1.0
        self._tick_times = [t for t in self._tick_times if t > cutoff]

        return ticks_to_process

    def pause(self) -> None:
        """Pause the simulation."""
        self._paused = True

    def resume(self) -> None:
        """Resume the simulation."""
        self._paused = False
        # Reset accumulated time to prevent time jump
        self._accumulated_time = 0.0
        self._last_update = time.time()

    def toggle_pause(self) -> bool:
        """Toggle pause state. Returns new pause state."""
        if self._paused:
            self.resume()
        else:
            self.pause()
        return self._paused

    def set_speed(self, speed: float | SimulationSpeed) -> None:
        """Set simulation speed multiplier."""
        if isinstance(speed, SimulationSpeed):
            self.speed = speed.value
        else:
            self.speed = max(0.0, min(speed, SimulationSpeed.MAX.value))

    def cycle_speed(self) -> float:
        """Cycle through preset speeds. Returns new speed."""
        speeds = [s.value for s in SimulationSpeed if s.value > 0]
        current_idx = -1

        for i, s in enumerate(speeds):
            if abs(self.speed - s) < 0.01:
                current_idx = i
                break

        next_idx = (current_idx + 1) % len(speeds)
        self.speed = speeds[next_idx]
        return self.speed

    def reset(self) -> None:
        """Reset clock to initial state."""
        self._tick = 0
        self._accumulated_time = 0.0
        self._last_update = time.time()
        self._paused = False
        self._tick_times.clear()

    def format_time(self) -> str:
        """Format simulation time as HH:MM:SS."""
        total_seconds = int(self.simulation_time)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        seconds = total_seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
