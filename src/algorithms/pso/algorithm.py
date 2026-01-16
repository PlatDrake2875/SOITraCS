"""Particle Swarm Optimization (PSO) algorithm for signal timing optimization."""

from typing import Dict, Any, Tuple, List
from dataclasses import dataclass, field
import random
import numpy as np
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState


@dataclass
class Particle:
    """A particle in the swarm."""

    position: np.ndarray  # Current position (signal timings)
    velocity: np.ndarray  # Current velocity
    best_position: np.ndarray  # Personal best position
    best_fitness: float = float('inf')  # Personal best fitness


class PSOAlgorithm(BaseAlgorithm):
    """
    Particle Swarm Optimization for signal timing.

    Uses a swarm of particles to search for optimal signal timing
    configurations that minimize total delay.

    Note: This runs in background and periodically updates signal timings.
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

        # PSO parameters
        self.swarm_size = config.get("swarm_size", 30)
        self.c1 = config.get("c1", 2.0)  # Cognitive coefficient
        self.c2 = config.get("c2", 2.0)  # Social coefficient
        self.w = config.get("w", 0.7)    # Inertia weight
        self.optimization_interval = config.get("optimization_interval", 50)

        # Swarm state
        self._particles: List[Particle] = []
        self._global_best_position: np.ndarray | None = None
        self._global_best_fitness = float('inf')

        # Metrics
        self._iterations = 0
        self._convergence_history: List[float] = []
        self._tick_counter = 0

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize PSO swarm."""
        self._network = network
        self._initialized = True

        # Dimension = number of signal phases to optimize
        n_intersections = len(network.intersections)
        dimension = n_intersections * 2  # 2 phases per intersection

        # Initialize swarm
        self._particles = []
        for _ in range(self.swarm_size):
            # Random initial position (timing values between 10 and 60)
            position = np.random.uniform(10, 60, dimension)
            velocity = np.random.uniform(-5, 5, dimension)

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

        # Only optimize periodically
        if self._tick_counter % self.optimization_interval != 0:
            self._update_visualization(state)
            return {}

        # Run one PSO iteration
        self._run_iteration(state)

        self._update_visualization(state)

        return {
            "iterations": self._iterations,
            "best_fitness": self._global_best_fitness,
        }

    def _run_iteration(self, state: SimulationState) -> None:
        """Run one PSO iteration."""
        for particle in self._particles:
            # Evaluate fitness (simplified: use current average delay)
            fitness = state.current_metrics.average_delay

            # Update personal best
            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_position = particle.position.copy()

            # Update global best
            if fitness < self._global_best_fitness:
                self._global_best_fitness = fitness
                self._global_best_position = particle.position.copy()

        # Update velocities and positions
        for particle in self._particles:
            if self._global_best_position is None:
                continue

            r1, r2 = random.random(), random.random()

            # Velocity update
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self._global_best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social

            # Clamp velocity
            particle.velocity = np.clip(particle.velocity, -10, 10)

            # Position update
            particle.position = particle.position + particle.velocity

            # Clamp position to valid range
            particle.position = np.clip(particle.position, 10, 60)

        self._iterations += 1
        self._convergence_history.append(self._global_best_fitness)

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for PSO overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        # Show particles as points around intersections
        for i, particle in enumerate(self._particles[:20]):  # Limit for performance
            # Map particle position to screen coordinates
            int_idx = i % len(self._network.intersections)
            intersection = list(self._network.intersections.values())[int_idx]

            # Offset particle from intersection based on position values
            offset_x = (particle.position[0] - 35) * 2
            offset_y = (particle.position[1] - 35) * 2 if len(particle.position) > 1 else 0

            pos = (
                intersection.position[0] + offset_x,
                intersection.position[1] + offset_y
            )

            self._visualization.points.append(pos)
            self._visualization.point_colors.append(Colors.PSO_COLOR)
            self._visualization.point_sizes.append(4.0)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get PSO visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get PSO-specific metrics."""
        return {
            "iterations": float(self._iterations),
            "best_fitness": self._global_best_fitness if self._global_best_fitness != float('inf') else 0.0,
            "swarm_size": float(self.swarm_size),
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
