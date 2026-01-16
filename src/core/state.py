"""Simulation state container."""

from dataclasses import dataclass, field
from typing import Dict, List, Any, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from src.entities.vehicle import Vehicle
    from src.entities.network import RoadNetwork
    from src.algorithms.base import BaseAlgorithm


@dataclass
class MetricsSnapshot:
    """Snapshot of simulation metrics at a point in time."""

    tick: int = 0
    total_vehicles: int = 0
    vehicles_arrived: int = 0
    vehicles_spawned: int = 0
    average_speed: float = 0.0
    average_delay: float = 0.0
    average_queue_length: float = 0.0
    throughput: float = 0.0  # Vehicles/minute
    total_wait_time: float = 0.0


@dataclass
class SimulationState:
    """
    Central state container for the simulation.

    Provides a single source of truth that all components can reference.
    """

    # Core entities
    vehicles: Dict[int, "Vehicle"] = field(default_factory=dict)
    network: "RoadNetwork | None" = None

    # Algorithm instances
    algorithms: Dict[str, "BaseAlgorithm"] = field(default_factory=dict)

    # Metrics tracking
    current_metrics: MetricsSnapshot = field(default_factory=MetricsSnapshot)
    metrics_history: List[MetricsSnapshot] = field(default_factory=list)
    metrics_history_limit: int = 3600  # Store ~2 minutes at 30 ticks/sec

    # Vehicle ID counter
    _next_vehicle_id: int = field(default=0, init=False)

    # Comparison mode
    comparison_mode: bool = False
    baseline_metrics: MetricsSnapshot | None = None

    def get_next_vehicle_id(self) -> int:
        """Get next unique vehicle ID."""
        vid = self._next_vehicle_id
        self._next_vehicle_id += 1
        return vid

    def add_vehicle(self, vehicle: "Vehicle") -> None:
        """Add a vehicle to the simulation."""
        self.vehicles[vehicle.id] = vehicle

    def remove_vehicle(self, vehicle_id: int) -> "Vehicle | None":
        """Remove and return a vehicle from the simulation."""
        return self.vehicles.pop(vehicle_id, None)

    def get_vehicle(self, vehicle_id: int) -> "Vehicle | None":
        """Get a vehicle by ID."""
        return self.vehicles.get(vehicle_id)

    def update_metrics(self, tick: int) -> None:
        """Update current metrics from simulation state."""
        vehicles = list(self.vehicles.values())
        num_vehicles = len(vehicles)

        if num_vehicles > 0:
            speeds = [v.speed for v in vehicles]
            delays = [v.delay for v in vehicles]
            self.current_metrics.average_speed = float(np.mean(speeds))
            self.current_metrics.average_delay = float(np.mean(delays))
        else:
            self.current_metrics.average_speed = 0.0
            self.current_metrics.average_delay = 0.0

        self.current_metrics.tick = tick
        self.current_metrics.total_vehicles = num_vehicles

        # Calculate queue lengths from network
        if self.network:
            queue_lengths = []
            for intersection in self.network.intersections.values():
                queue_lengths.extend(intersection.get_queue_lengths().values())
            if queue_lengths:
                self.current_metrics.average_queue_length = float(np.mean(queue_lengths))

        # Store in history
        snapshot = MetricsSnapshot(
            tick=self.current_metrics.tick,
            total_vehicles=self.current_metrics.total_vehicles,
            vehicles_arrived=self.current_metrics.vehicles_arrived,
            vehicles_spawned=self.current_metrics.vehicles_spawned,
            average_speed=self.current_metrics.average_speed,
            average_delay=self.current_metrics.average_delay,
            average_queue_length=self.current_metrics.average_queue_length,
            throughput=self.current_metrics.throughput,
            total_wait_time=self.current_metrics.total_wait_time,
        )
        self.metrics_history.append(snapshot)

        # Trim history
        if len(self.metrics_history) > self.metrics_history_limit:
            self.metrics_history = self.metrics_history[-self.metrics_history_limit:]

    def record_arrival(self) -> None:
        """Record a vehicle arrival."""
        self.current_metrics.vehicles_arrived += 1

    def record_spawn(self) -> None:
        """Record a vehicle spawn."""
        self.current_metrics.vehicles_spawned += 1

    def get_algorithm(self, name: str) -> "BaseAlgorithm | None":
        """Get an algorithm by name."""
        return self.algorithms.get(name)

    def register_algorithm(self, name: str, algorithm: "BaseAlgorithm") -> None:
        """Register an algorithm."""
        self.algorithms[name] = algorithm

    def get_active_algorithms(self) -> List["BaseAlgorithm"]:
        """Get list of enabled algorithms."""
        return [algo for algo in self.algorithms.values() if algo.enabled]

    def reset(self) -> None:
        """Reset simulation state."""
        self.vehicles.clear()
        self._next_vehicle_id = 0
        self.current_metrics = MetricsSnapshot()
        self.metrics_history.clear()
        self.baseline_metrics = None

        # Reset network if present
        if self.network:
            self.network.reset()

        # Reset all algorithms
        for algo in self.algorithms.values():
            algo.reset()

    def get_metrics_array(self, metric_name: str, limit: int = 300) -> np.ndarray:
        """Get array of historical metric values for graphing."""
        history = self.metrics_history[-limit:]
        return np.array([getattr(m, metric_name, 0) for m in history])
