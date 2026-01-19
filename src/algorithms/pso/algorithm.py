"""Particle Swarm Optimization (PSO) algorithm for signal timing optimization."""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
import random
import numpy as np
from config.colors import Colors
from config.constants import (
    PSO_HIGH_QUEUE_THRESHOLD, PSO_VERY_HIGH_QUEUE_THRESHOLD,
    PSO_SLOW_MIN_GREEN_THRESHOLD, PSO_LONG_MAX_GREEN_THRESHOLD,
    PSO_LOW_QUEUE_THRESHOLD, PSO_QUICK_MIN_GREEN,
    PSO_HIGH_QUEUE_PENALTY, PSO_LONG_PHASE_PENALTY,
    PSO_BASELINE_PENALTY, PSO_INVALID_CONFIG_PENALTY, PSO_QUICK_SWITCH_BONUS,
)
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState
from src.core.event_bus import Event, EventType, get_event_bus

@dataclass
class Particle:
    """A particle in the swarm."""

    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_fitness: float = float('inf')

class PSOAlgorithm(BaseAlgorithm):
    """
    Particle Swarm Optimization for signal timing.

    Uses a swarm of particles to search for optimal signal timing
    configurations that minimize total delay.

    Publishes SIGNAL_TIMING_SUGGESTED events for SOTL integration.
    """

    @property
    def name(self) -> str:
        return "pso"

    @property
    def display_name(self) -> str:
        return "Signal Optimization (PSO)"

    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.PSO_COLOR

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.swarm_size = config.get("swarm_size", 30)
        self.c1 = config.get("c1", 2.0)
        self.c2 = config.get("c2", 2.0)
        self.w = config.get("w", 0.7)
        self.optimization_interval = config.get("optimization_interval", 50)

        self._particles: List[Particle] = []
        self._global_best_position: np.ndarray | None = None
        self._global_best_fitness = float('inf')

        self._queues_snapshot: Dict[int, int] = {}
        self._timings_snapshot: Dict[int, Tuple[int, int]] = {}

        self._swarm_centroid: np.ndarray | None = None
        self._swarm_spread: float = 0.0

        self._iterations = 0
        self._convergence_history: List[float] = []
        self._tick_counter = 0
        self._last_publish_tick = 0

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize PSO swarm."""
        self._network = network
        self._initialized = True

        n_intersections = len(network.intersections)
        dimension = n_intersections * 2

        self._particles = []
        for _ in range(self.swarm_size):

            position = np.zeros(dimension)
            for i in range(n_intersections):
                position[i * 2] = np.random.uniform(10, 30)
                position[i * 2 + 1] = np.random.uniform(30, 60)

            velocity = np.random.uniform(-3, 3, dimension)

            particle = Particle(
                position=position,
                velocity=velocity,
                best_position=position.copy(),
            )
            self._particles.append(particle)

        self._global_best_position = self._particles[0].position.copy()

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update PSO algorithm."""
        if not self.enabled or not self._initialized:
            return {}

        self._tick_counter += 1

        if self._tick_counter % self.optimization_interval != 0:
            self._update_visualization(state)
            return {}

        self._capture_snapshot(state)

        self._run_iteration(state)

        self._publish_timing_suggestions()

        self._update_visualization(state)

        return {
            "iterations": self._iterations,
            "best_fitness": self._global_best_fitness,
            "swarm_spread": self._swarm_spread,
        }

    def _capture_snapshot(self, state: SimulationState) -> None:
        """Capture immutable queue and timing snapshot for thread-safe evaluation."""
        self._queues_snapshot.clear()
        self._timings_snapshot.clear()

        if not self._network:
            return

        for int_id, intersection in self._network.intersections.items():

            self._queues_snapshot[int_id] = intersection.get_total_queue()

            light = intersection.traffic_light
            self._timings_snapshot[int_id] = (
                light.get_time_in_phase(),
                60
            )

    def _evaluate_particle(self, position: np.ndarray) -> float:
        """
        Estimate delay penalty for this timing configuration.

        Each particle's position encodes [min_green_1, max_green_1, min_green_2, max_green_2, ...]
        Fitness estimates how well these timings match current traffic conditions.
        """
        fitness = 0.0
        int_ids = list(self._queues_snapshot.keys())

        for i, int_id in enumerate(int_ids):
            if i * 2 + 1 >= len(position):
                break

            min_green = position[i * 2]
            max_green = position[i * 2 + 1]
            queue = self._queues_snapshot.get(int_id, 0)

            if queue > PSO_HIGH_QUEUE_THRESHOLD and min_green > PSO_SLOW_MIN_GREEN_THRESHOLD:

                fitness += queue * (min_green - PSO_QUICK_MIN_GREEN) * PSO_HIGH_QUEUE_PENALTY
            elif queue < PSO_LOW_QUEUE_THRESHOLD and max_green > PSO_LONG_MAX_GREEN_THRESHOLD:

                fitness += (max_green - 40) * PSO_LONG_PHASE_PENALTY
            else:

                fitness += queue * PSO_BASELINE_PENALTY

            if min_green > max_green:
                fitness += PSO_INVALID_CONFIG_PENALTY

            if queue > PSO_VERY_HIGH_QUEUE_THRESHOLD and min_green < PSO_QUICK_MIN_GREEN:
                fitness -= PSO_QUICK_SWITCH_BONUS

        return fitness

    def _run_iteration(self, state: SimulationState) -> None:
        """Run one PSO iteration with per-particle fitness evaluation."""

        for particle in self._particles:
            fitness = self._evaluate_particle(particle.position)

            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            if fitness < self._global_best_fitness:
                self._global_best_fitness = fitness
                self._global_best_position = particle.position.copy()

        for particle in self._particles:
            if self._global_best_position is None:
                continue

            r1, r2 = random.random(), random.random()

            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self._global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social

            particle.velocity = np.clip(particle.velocity, -5, 5)

            particle.position = particle.position + particle.velocity

            n_intersections = len(particle.position) // 2
            for i in range(n_intersections):

                particle.position[i * 2] = np.clip(particle.position[i * 2], 10, 30)

                particle.position[i * 2 + 1] = np.clip(particle.position[i * 2 + 1], 30, 60)

        self._iterations += 1
        self._convergence_history.append(self._global_best_fitness)

        if len(self._convergence_history) > 100:
            self._convergence_history = self._convergence_history[-100:]

    def _publish_timing_suggestions(self) -> None:
        """Publish optimized timing suggestions to SOTL via EventBus."""
        if self._global_best_position is None:
            return

        event_bus = get_event_bus()
        int_ids = list(self._network.intersections.keys()) if self._network else []

        for i, int_id in enumerate(int_ids):
            if i * 2 + 1 >= len(self._global_best_position):
                break

            min_green = int(np.clip(self._global_best_position[i * 2], 10, 30))
            max_green = int(np.clip(self._global_best_position[i * 2 + 1], 30, 60))

            if min_green >= max_green:
                min_green = max(10, max_green - 20)

            event_bus.publish(Event(
                type=EventType.SIGNAL_TIMING_SUGGESTED,
                data={
                    "intersection_id": int_id,
                    "min_green": min_green,
                    "max_green": max_green,
                },
                source=self.name,
            ))

        self._last_publish_tick = self._tick_counter

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for PSO overlay with fitness coloring."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        if self._particles:
            positions = np.array([p.position for p in self._particles])
            if len(positions) > 0:
                self._swarm_centroid = positions.mean(axis=0)
                self._swarm_spread = positions.std()

        int_list = list(self._network.intersections.values())

        for i, particle in enumerate(self._particles[:20]):

            int_idx = i % len(int_list)
            intersection = int_list[int_idx]

            offset_x = (particle.position[0] - 20) * 3 if len(particle.position) > 0 else 0
            offset_y = (particle.position[1] - 45) * 2 if len(particle.position) > 1 else 0

            pos = (
                intersection.position[0] + offset_x,
                intersection.position[1] + offset_y
            )

            if particle.best_fitness < float('inf'):
                normalized_fitness = 1.0 - min(1.0, particle.best_fitness / 50.0)
            else:
                normalized_fitness = 0.0

            color = Colors.lerp(Colors.CONGESTION_HIGH, Colors.PSO_COLOR, normalized_fitness)

            size = 3.0 + normalized_fitness * 4.0

            self._visualization.points.append(pos)
            self._visualization.point_colors.append(color)
            self._visualization.point_sizes.append(size)

        if self._global_best_position is not None and int_list:
            best_intersection = int_list[0]
            best_pos = (
                best_intersection.position[0],
                best_intersection.position[1]
            )

            self._visualization.points.insert(0, best_pos)
            self._visualization.point_colors.insert(0, Colors.PSO_COLOR)
            self._visualization.point_sizes.insert(0, 8.0)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get PSO visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get PSO-specific metrics."""
        convergence_rate = 0.0
        if len(self._convergence_history) >= 2:
            recent = self._convergence_history[-10:]
            if len(recent) >= 2:
                convergence_rate = (recent[0] - recent[-1]) / max(1, len(recent))

        return {
            "iterations": float(self._iterations),
            "best_fitness": self._global_best_fitness if self._global_best_fitness != float('inf') else 0.0,
            "swarm_size": float(self.swarm_size),
            "swarm_spread": self._swarm_spread,
            "convergence_rate": convergence_rate,
        }

    def reset(self) -> None:
        """Reset PSO algorithm."""
        super().reset()
        self._particles.clear()
        self._global_best_position = None
        self._global_best_fitness = float('inf')
        self._iterations = 0
        self._convergence_history.clear()
        self._tick_counter = 0
        self._queues_snapshot.clear()
        self._timings_snapshot.clear()
        self._swarm_centroid = None
        self._swarm_spread = 0.0
        self._last_publish_tick = 0
