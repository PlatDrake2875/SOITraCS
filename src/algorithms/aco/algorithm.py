"""Ant Colony Optimization (ACO) algorithm for dynamic vehicle routing."""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
import random
import math
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.entities.road import Road
from src.core.state import SimulationState
from src.core.event_bus import Event, EventType, get_event_bus


class ACOAlgorithm(BaseAlgorithm):
    """
    Ant Colony Optimization for dynamic vehicle routing.

    Maintains a pheromone matrix on roads that guides vehicle route selection.
    - Vehicles deposit pheromone based on travel time
    - Pheromone evaporates over time
    - Routes with high pheromone are preferred

    This creates emergent rerouting around congestion.
    """

    @property
    def name(self) -> str:
        return "aco"

    @property
    def display_name(self) -> str:
        return "Ant Colony Routing"

    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.ACO_COLOR

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        # ACO parameters
        self.alpha = config.get("alpha", 1.0)  # Pheromone importance
        self.beta = config.get("beta", 2.0)    # Heuristic importance
        self.rho = config.get("rho", 0.1)      # Evaporation rate
        self.Q = config.get("Q", 100)          # Pheromone deposit constant
        self.reroute_frequency = config.get("reroute_frequency", 10)

        # Pheromone matrix (road_id -> pheromone level)
        self._pheromone: Dict[int, float] = {}
        self._min_pheromone = 0.1
        self._max_pheromone = 10.0

        # Metrics
        self._total_reroutes = 0
        self._avg_pheromone = 1.0
        self._tick_counter = 0

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize pheromone matrix."""
        self._network = network
        self._pheromone.clear()

        # Initialize pheromone on all roads
        for road_id in network.roads:
            self._pheromone[road_id] = 1.0

        # Subscribe to events
        event_bus = get_event_bus()
        event_bus.subscribe(EventType.VEHICLE_ARRIVED, self._on_vehicle_arrived)
        event_bus.subscribe(EventType.CONGESTION_DETECTED, self._on_congestion)

        self._initialized = True

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update ACO algorithm - evaporate and deposit pheromone."""
        if not self.enabled or not self._initialized:
            return {}

        self._tick_counter += 1

        # Evaporate pheromone
        self._evaporate()

        # Deposit pheromone from moving vehicles
        self._deposit_from_vehicles(state)

        # Update visualization
        self._update_visualization(state)

        # Calculate metrics
        if self._pheromone:
            self._avg_pheromone = sum(self._pheromone.values()) / len(self._pheromone)

        return {
            "avg_pheromone": self._avg_pheromone,
            "total_reroutes": self._total_reroutes,
        }

    def _evaporate(self) -> None:
        """Apply pheromone evaporation."""
        for road_id in self._pheromone:
            self._pheromone[road_id] *= (1 - self.rho)
            self._pheromone[road_id] = max(self._min_pheromone, self._pheromone[road_id])

    def _deposit_from_vehicles(self, state: SimulationState) -> None:
        """Deposit pheromone from moving vehicles."""
        for vehicle in state.vehicles.values():
            if vehicle.speed > 0:
                road_id = vehicle.current_road_id
                if road_id in self._pheromone:
                    # Deposit inversely proportional to travel time
                    deposit = self.Q / max(1, vehicle.total_time - vehicle.wait_time)
                    self._pheromone[road_id] += deposit * 0.01  # Small deposits per tick
                    self._pheromone[road_id] = min(self._max_pheromone, self._pheromone[road_id])

    def get_route_weights(self) -> Dict[int, float]:
        """
        Get routing weights for pathfinding.

        Lower weight = more attractive (inverse pheromone).
        """
        weights = {}
        if not self._network:
            return weights

        for road_id, road in self._network.roads.items():
            pheromone = self._pheromone.get(road_id, 1.0)
            travel_time = road.get_travel_time()

            # Combined weight: low pheromone and high travel time = high weight (avoid)
            # High pheromone and low travel time = low weight (prefer)
            pheromone_factor = 1.0 / (pheromone ** self.alpha)
            heuristic_factor = travel_time ** self.beta

            weights[road_id] = pheromone_factor * heuristic_factor

        return weights

    def _on_vehicle_arrived(self, event: Event) -> None:
        """Handle vehicle arrival - deposit pheromone on route."""
        # Could enhance to deposit along entire route
        pass

    def _on_congestion(self, event: Event) -> None:
        """Handle congestion detection - reduce pheromone on congested roads."""
        road_id = event.data.get("road_id")
        if road_id in self._pheromone:
            # Reduce pheromone on congested road
            self._pheromone[road_id] *= 0.5
            self._pheromone[road_id] = max(self._min_pheromone, self._pheromone[road_id])

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for ACO overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        # Draw pheromone trails on roads
        for road_id, road in self._network.roads.items():
            pheromone = self._pheromone.get(road_id, 1.0)
            normalized = min(1.0, pheromone / self._max_pheromone)

            self._visualization.road_values[road_id] = normalized

            # Green color with intensity based on pheromone
            alpha = int(50 + normalized * 150)  # 50-200 alpha range
            self._visualization.road_colors[road_id] = Colors.ACO_COLOR

            # Add line overlay for strong pheromone trails
            if normalized > 0.3:
                start = road.start_pos
                end = road.end_pos
                self._visualization.lines.append((start, end))
                self._visualization.line_colors.append(
                    (Colors.ACO_COLOR[0], Colors.ACO_COLOR[1], Colors.ACO_COLOR[2], alpha)
                )

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get ACO visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get ACO-specific metrics."""
        return {
            "avg_pheromone": self._avg_pheromone,
            "total_reroutes": float(self._total_reroutes),
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
        }

    def reset(self) -> None:
        """Reset ACO algorithm."""
        super().reset()

        # Reset pheromone to initial levels
        for road_id in self._pheromone:
            self._pheromone[road_id] = 1.0

        self._total_reroutes = 0
        self._avg_pheromone = 1.0
        self._tick_counter = 0
