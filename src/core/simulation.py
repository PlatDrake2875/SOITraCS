"""Main simulation loop controller."""

import logging
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import time

import numpy as np

logger = logging.getLogger(__name__)

from .event_bus import EventBus, Event, EventType, get_event_bus
from .state import SimulationState
from .clock import SimulationClock
from .scenario import ScenarioManager, ScenarioConfig
from config.settings import Settings, get_settings
from config.constants import CONGESTION_THRESHOLD, CONGESTION_CLEAR_THRESHOLD

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
    headless: bool = field(default=False)

    running: bool = field(default=False, init=False)
    _initialized: bool = field(default=False, init=False)

    _congested_roads: Set[int] = field(default_factory=set, init=False)

    _scenario_manager: Optional[ScenarioManager] = field(default=None, init=False)
    _spawn_multiplier: float = field(default=1.0, init=False)

    _seed: Optional[int] = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Set up event subscriptions."""
        self.event_bus.subscribe(EventType.SIMULATION_PAUSE, self._on_pause)
        self.event_bus.subscribe(EventType.SIMULATION_RESUME, self._on_resume)
        self.event_bus.subscribe(EventType.SIMULATION_RESET, self._on_reset)
        self.event_bus.subscribe(EventType.SPEED_CHANGED, self._on_speed_changed)

    def set_seed(self, seed: int) -> None:
        """
        Set random seed for reproducibility.

        Seeds Python's random module, NumPy, and algorithm-specific RNGs.

        Args:
            seed: Random seed value
        """
        self._seed = seed
        random.seed(seed)
        np.random.seed(seed)

        for algo in self.state.algorithms.values():
            if hasattr(algo, "set_seed"):
                algo.set_seed(seed)

        logger.debug(f"Simulation random seed set to {seed}")

    def get_seed(self) -> Optional[int]:
        """Get the current random seed, if set."""
        return self._seed

    def initialize(self, network_path: Path | str | None = None) -> None:
        """
        Initialize the simulation with a network.

        Args:
            network_path: Path to network YAML file, or None for default
        """
        from src.entities.network import RoadNetwork

        if network_path:
            self.state.network = RoadNetwork.from_yaml(Path(network_path))
        else:

            self.state.network = RoadNetwork.create_default_grid()

        self._init_algorithms()

        if self.state.network and self.state.network.scenarios:
            scenarios = {
                name: ScenarioConfig.from_dict(name, data)
                for name, data in self.state.network.scenarios.items()
            }
            self._scenario_manager = ScenarioManager(scenarios)

        self._initialized = True

        self.event_bus.publish(Event(
            type=EventType.SIMULATION_START,
            source="simulation",
        ))

    def _init_algorithms(self) -> None:
        """Initialize all enabled algorithms."""
        from src.algorithms.registry import get_algorithm_registry

        registry = get_algorithm_registry()

        algo_settings = self.settings.algorithms

        for algo_name in ["cellular_automata", "sotl", "aco", "pso", "som", "marl", "marl_enhanced", "pressure"]:
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

        ticks = self.clock.update(real_dt)

        for _ in range(ticks):
            self._tick()

        return ticks

    def _tick(self) -> None:
        """Process a single simulation tick."""
        tick = self.clock.tick
        dt = self.clock.dt

        self.event_bus.publish(Event(
            type=EventType.TICK,
            data={"tick": tick, "dt": dt},
            source="simulation",
        ))

        if self.state.network:
            self.state.network.update(dt)

        self._check_congestion()

        algorithm_outputs: Dict[str, Any] = {}

        for algo in self.state.get_active_algorithms():
            try:
                output = algo.update(self.state, dt)
                if output:
                    algorithm_outputs[algo.name] = output
            except Exception as e:
                print(f"Error updating algorithm {algo.name}: {e}")

        self.event_bus.flush()

        self._reroute_vehicles(tick)

        self._update_vehicles(dt)

        self._spawn_vehicles()

        self.state.update_metrics(tick)

        if self._scenario_manager:
            self._scenario_manager.update(self, tick)

    def _update_vehicles(self, dt: float) -> None:
        """Update all vehicles."""
        from src.entities.vehicle import VehicleState

        vehicles_to_remove = []

        for vehicle in list(self.state.vehicles.values()):
            vehicle.update(dt, self.state)

            if vehicle.state == VehicleState.ARRIVED:
                vehicles_to_remove.append(vehicle.id)
                self.state.record_arrival()

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

        if len(self.state.vehicles) >= self.settings.simulation.max_vehicles:
            return

        for spawn_point in self.state.network.spawn_points:
            effective_rate = spawn_point.rate * self._spawn_multiplier
            if random.random() < effective_rate:

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

    def _check_congestion(self) -> None:
        """Check roads for congestion changes and publish events."""
        if not self.state.network:
            return

        for road_id, road in self.state.network.roads.items():
            density = road.get_density()
            was_congested = road_id in self._congested_roads

            if not was_congested and density >= CONGESTION_THRESHOLD:

                self._congested_roads.add(road_id)
                self.event_bus.publish(Event(
                    type=EventType.CONGESTION_DETECTED,
                    data={"road_id": road_id, "density": density},
                    source="simulation",
                ))
            elif was_congested and density < CONGESTION_CLEAR_THRESHOLD:

                self._congested_roads.discard(road_id)
                self.event_bus.publish(Event(
                    type=EventType.CONGESTION_CLEARED,
                    data={"road_id": road_id, "density": density},
                    source="simulation",
                ))

    def _reroute_vehicles(self, tick: int) -> None:
        """Reroute vehicles using ACO weights if available."""
        if not self.state.network:
            return

        aco = self.state.get_algorithm("aco")
        if not aco or not aco.enabled:
            return

        weights = aco.get_route_weights()
        reroute_interval = self.settings.simulation.reroute_interval

        for vehicle in self.state.vehicles.values():

            if tick - vehicle._last_reroute_tick < reroute_interval:
                continue

            if vehicle.reroute(self.state.network, weights, tick):

                aco.record_reroute()

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
        self._congested_roads.clear()

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

    def inject_incident(
        self, road_id: int, duration: int = 100, source: str = "user"
    ) -> bool:
        """Inject an incident on a road."""
        if not self.state.network:
            return False

        road = self.state.network.get_road(road_id)
        if road:
            road.set_incident(duration, source=source)
            self.event_bus.publish(Event(
                type=EventType.INCIDENT_INJECTED,
                data={"road_id": road_id, "duration": duration, "source": source},
                source="simulation",
            ))
            return True
        return False

    def _on_pause(self, event: Event) -> None:
        """Handle pause event."""
        pass

    def _on_resume(self, event: Event) -> None:
        """Handle resume event."""
        pass

    def _on_reset(self, event: Event) -> None:
        """Handle reset event."""
        pass

    def _on_speed_changed(self, event: Event) -> None:
        """Handle speed change event."""
        pass

    def set_spawn_multiplier(self, multiplier: float) -> None:
        """Set spawn rate multiplier."""
        clamped = max(0.0, min(5.0, multiplier))
        if clamped != multiplier:
            logger.warning(f"Spawn multiplier {multiplier} clamped to {clamped}")
        self._spawn_multiplier = clamped

    def activate_scenario(self, name: str) -> bool:
        """Activate a named scenario."""
        if not self._scenario_manager:
            return False
        success = self._scenario_manager.activate(name, self, self.clock.tick)
        if success:
            self.event_bus.publish(Event(
                type=EventType.SCENARIO_STARTED,
                data={"scenario": name},
                source="simulation",
            ))
        return success

    def deactivate_scenario(self) -> None:
        """Deactivate current scenario."""
        if self._scenario_manager and self._scenario_manager.active_scenario:
            scenario_name = self._scenario_manager.active_scenario
            self._scenario_manager.deactivate(self)
            self.event_bus.publish(Event(
                type=EventType.SCENARIO_ENDED,
                data={"scenario": scenario_name},
                source="simulation",
            ))

    def get_active_scenario(self) -> Optional[str]:
        """Get currently active scenario name."""
        return self._scenario_manager.get_active() if self._scenario_manager else None

    def run_ticks(self, n_ticks: int) -> List["MetricsSnapshot"]:
        """
        Run the simulation for a specified number of ticks (headless mode).

        This method is designed for batch experiments where GUI is not needed.
        Collects and returns metrics snapshots for each tick.

        Args:
            n_ticks: Number of simulation ticks to run

        Returns:
            List of MetricsSnapshot objects from each tick
        """
        from .state import MetricsSnapshot

        if not self._initialized:
            raise RuntimeError("Simulation must be initialized before running ticks")

        snapshots: List[MetricsSnapshot] = []

        for _ in range(n_ticks):

            self.clock._tick += 1
            self._tick()

            m = self.state.current_metrics
            snapshot = MetricsSnapshot(
                tick=m.tick,
                total_vehicles=m.total_vehicles,
                vehicles_arrived=m.vehicles_arrived,
                vehicles_spawned=m.vehicles_spawned,
                average_speed=m.average_speed,
                average_delay=m.average_delay,
                average_queue_length=m.average_queue_length,
                throughput=m.throughput,
                total_wait_time=m.total_wait_time,
            )
            snapshots.append(snapshot)

        return snapshots

    def get_current_metrics_dict(self) -> Dict[str, float]:
        """
        Get current simulation metrics as a dictionary.

        Returns:
            Dictionary mapping metric names to values
        """
        m = self.state.current_metrics
        return {
            "tick": float(m.tick),
            "total_vehicles": float(m.total_vehicles),
            "vehicles_arrived": float(m.vehicles_arrived),
            "vehicles_spawned": float(m.vehicles_spawned),
            "average_speed": m.average_speed,
            "average_delay": m.average_delay,
            "average_queue_length": m.average_queue_length,
            "throughput": m.throughput,
            "total_wait_time": m.total_wait_time,
        }
