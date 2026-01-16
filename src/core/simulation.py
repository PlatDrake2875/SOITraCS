"""Main simulation loop controller."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import time

from .event_bus import EventBus, Event, EventType, get_event_bus
from .state import SimulationState
from .clock import SimulationClock
from config.settings import Settings, get_settings


@dataclass
class Simulation:
    """
    Main simulation controller.

    Orchestrates the simulation loop, coordinates algorithms,
    and manages the overall simulation lifecycle.
    """

    settings: Settings = field(default_factory=get_settings)
    state: SimulationState = field(default_factory=SimulationState)
    clock: SimulationClock = field(default_factory=SimulationClock)
    event_bus: EventBus = field(default_factory=get_event_bus)

    # Flags
    running: bool = field(default=False, init=False)
    _initialized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        """Set up event subscriptions."""
        self.event_bus.subscribe(EventType.SIMULATION_PAUSE, self._on_pause)
        self.event_bus.subscribe(EventType.SIMULATION_RESUME, self._on_resume)
        self.event_bus.subscribe(EventType.SIMULATION_RESET, self._on_reset)
        self.event_bus.subscribe(EventType.SPEED_CHANGED, self._on_speed_changed)

    def initialize(self, network_path: Path | str | None = None) -> None:
        """
        Initialize the simulation with a network.

        Args:
            network_path: Path to network YAML file, or None for default
        """
        from src.entities.network import RoadNetwork

        # Load network
        if network_path:
            self.state.network = RoadNetwork.from_yaml(Path(network_path))
        else:
            # Create default grid network
            self.state.network = RoadNetwork.create_default_grid()

        # Initialize algorithms
        self._init_algorithms()

        self._initialized = True

        # Fire start event
        self.event_bus.publish(Event(
            type=EventType.SIMULATION_START,
            source="simulation",
        ))

    def _init_algorithms(self) -> None:
        """Initialize all enabled algorithms."""
        from src.algorithms.registry import get_algorithm_registry

        registry = get_algorithm_registry()

        # Initialize each algorithm based on settings
        algo_settings = self.settings.algorithms

        for algo_name in ["cellular_automata", "sotl", "aco", "pso", "som", "marl"]:
            config = getattr(algo_settings, algo_name, {})
            if config.get("enabled", False):
                algo_class = registry.get(algo_name)
                if algo_class:
                    algo = algo_class(config)
                    algo.initialize(self.state.network, self.state)
                    self.state.register_algorithm(algo_name, algo)

    def update(self, real_dt: float) -> int:
        """
        Main update loop.

        Args:
            real_dt: Real time elapsed since last update

        Returns:
            Number of simulation ticks processed
        """
        if not self._initialized:
            return 0

        # Get number of ticks to process
        ticks = self.clock.update(real_dt)

        for _ in range(ticks):
            self._tick()

        return ticks

    def _tick(self) -> None:
        """Process a single simulation tick."""
        tick = self.clock.tick
        dt = self.clock.dt

        # Fire tick event
        self.event_bus.publish(Event(
            type=EventType.TICK,
            data={"tick": tick, "dt": dt},
            source="simulation",
        ))

        # Update algorithms in priority order
        algorithm_outputs: Dict[str, Any] = {}

        for algo in self.state.get_active_algorithms():
            try:
                output = algo.update(self.state, dt)
                if output:
                    algorithm_outputs[algo.name] = output
            except Exception as e:
                print(f"Error updating algorithm {algo.name}: {e}")

        # Process pending events
        self.event_bus.flush()

        # Update vehicles
        self._update_vehicles(dt)

        # Spawn new vehicles
        self._spawn_vehicles()

        # Update metrics
        self.state.update_metrics(tick)

    def _update_vehicles(self, dt: float) -> None:
        """Update all vehicles."""
        from src.entities.vehicle import VehicleState

        vehicles_to_remove = []

        for vehicle in list(self.state.vehicles.values()):
            vehicle.update(dt, self.state)

            # Check for arrival
            if vehicle.state == VehicleState.ARRIVED:
                vehicles_to_remove.append(vehicle.id)
                self.state.record_arrival()

        # Remove arrived vehicles
        for vid in vehicles_to_remove:
            self.state.remove_vehicle(vid)
            self.event_bus.publish(Event(
                type=EventType.VEHICLE_ARRIVED,
                data={"vehicle_id": vid},
                source="simulation",
            ))

    def _spawn_vehicles(self) -> None:
        """Spawn new vehicles at spawn points."""
        from src.entities.vehicle import Vehicle
        import random

        if not self.state.network:
            return

        # Check vehicle limit
        if len(self.state.vehicles) >= self.settings.simulation.max_vehicles:
            return

        # Try each spawn point
        for spawn_point in self.state.network.spawn_points:
            if random.random() < spawn_point.rate:
                # Create vehicle
                vehicle = Vehicle.spawn(
                    vehicle_id=self.state.get_next_vehicle_id(),
                    spawn_point=spawn_point,
                    network=self.state.network,
                )

                if vehicle:
                    self.state.add_vehicle(vehicle)
                    self.state.record_spawn()

                    self.event_bus.publish(Event(
                        type=EventType.VEHICLE_SPAWNED,
                        data={"vehicle_id": vehicle.id},
                        source="simulation",
                    ))

    def pause(self) -> None:
        """Pause the simulation."""
        self.clock.pause()
        self.event_bus.publish(Event(
            type=EventType.SIMULATION_PAUSE,
            source="simulation",
        ))

    def resume(self) -> None:
        """Resume the simulation."""
        self.clock.resume()
        self.event_bus.publish(Event(
            type=EventType.SIMULATION_RESUME,
            source="simulation",
        ))

    def toggle_pause(self) -> bool:
        """Toggle pause state."""
        if self.clock.is_paused:
            self.resume()
        else:
            self.pause()
        return self.clock.is_paused

    def reset(self) -> None:
        """Reset the simulation to initial state."""
        self.state.reset()
        self.clock.reset()
        self.event_bus.clear_pending()

        self.event_bus.publish(Event(
            type=EventType.SIMULATION_RESET,
            source="simulation",
        ))

    def set_speed(self, speed: float) -> None:
        """Set simulation speed."""
        self.clock.set_speed(speed)
        self.event_bus.publish(Event(
            type=EventType.SPEED_CHANGED,
            data={"speed": speed},
            source="simulation",
        ))

    def toggle_algorithm(self, name: str) -> bool | None:
        """Toggle an algorithm on/off. Returns new state or None if not found."""
        algo = self.state.get_algorithm(name)
        if algo:
            algo.toggle(not algo.enabled)
            self.event_bus.publish(Event(
                type=EventType.ALGORITHM_TOGGLED,
                data={"algorithm": name, "enabled": algo.enabled},
                source="simulation",
            ))
            return algo.enabled
        return None

    def inject_incident(self, road_id: int, duration: int = 100) -> bool:
        """Inject an incident on a road."""
        if not self.state.network:
            return False

        road = self.state.network.get_road(road_id)
        if road:
            road.set_incident(duration)
            self.event_bus.publish(Event(
                type=EventType.INCIDENT_INJECTED,
                data={"road_id": road_id, "duration": duration},
                source="simulation",
            ))
            return True
        return False

    def _on_pause(self, event: Event) -> None:
        """Handle pause event."""
        pass  # Clock already handles pause

    def _on_resume(self, event: Event) -> None:
        """Handle resume event."""
        pass  # Clock already handles resume

    def _on_reset(self, event: Event) -> None:
        """Handle reset event."""
        pass  # Already handled in reset()

    def _on_speed_changed(self, event: Event) -> None:
        """Handle speed change event."""
        pass  # Already handled in set_speed()
