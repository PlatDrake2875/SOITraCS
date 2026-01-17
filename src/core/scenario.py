"""Scenario management for traffic presets."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .simulation import Simulation


@dataclass
class ScenarioConfig:
    """Configuration for a traffic scenario."""
    name: str
    spawn_rate_multiplier: float = 1.0
    blocked_roads: List[int] = field(default_factory=list)
    duration: int = 0  # 0 = permanent until cancelled

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "ScenarioConfig":
        """Create from dictionary."""
        blocked = data.get("blocked_roads", [])
        if "blocked_road" in data:  # Singular form from YAML
            blocked = [data["blocked_road"]]
        return cls(
            name=name,
            spawn_rate_multiplier=data.get("spawn_rate_multiplier", 1.0),
            blocked_roads=blocked,
            duration=data.get("duration", 0),
        )


class ScenarioManager:
    """Manages scenario activation and lifecycle."""

    def __init__(self, scenarios: Dict[str, ScenarioConfig]) -> None:
        self.scenarios = scenarios
        self.active_scenario: Optional[str] = None
        self._start_tick: int = 0
        self._active_incidents: List[int] = []

    def activate(self, name: str, simulation: "Simulation", tick: int) -> bool:
        """Activate a scenario."""
        if name not in self.scenarios:
            return False

        # Deactivate current first
        if self.active_scenario:
            self.deactivate(simulation)

        config = self.scenarios[name]
        self.active_scenario = name
        self._start_tick = tick

        # Apply spawn rate multiplier
        simulation.set_spawn_multiplier(config.spawn_rate_multiplier)

        # Block roads (inject incidents)
        for road_id in config.blocked_roads:
            simulation.inject_incident(road_id, duration=99999)
            self._active_incidents.append(road_id)

        return True

    def deactivate(self, simulation: "Simulation") -> None:
        """Deactivate current scenario."""
        if not self.active_scenario:
            return

        # Reset spawn rate
        simulation.set_spawn_multiplier(1.0)

        # Clear incidents we created
        for road_id in self._active_incidents:
            road = simulation.state.network.get_road(road_id)
            if road:
                road.clear_incident()
        self._active_incidents.clear()

        self.active_scenario = None

    def update(self, simulation: "Simulation", tick: int) -> None:
        """Check if scenario should expire."""
        if not self.active_scenario:
            return

        config = self.scenarios[self.active_scenario]
        if config.duration > 0:
            elapsed = tick - self._start_tick
            if elapsed >= config.duration:
                self.deactivate(simulation)

    def get_active(self) -> Optional[str]:
        """Get active scenario name."""
        return self.active_scenario
