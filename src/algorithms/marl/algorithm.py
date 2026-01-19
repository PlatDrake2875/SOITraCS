"""
Multi-Agent Reinforcement Learning (MARL) algorithm for signal control.

Implements independent Q-learning agents for each intersection with:
- 8-dimensional state representation (queue lengths, phase, time, neighbor pressure)
- Local reward function based on queue reduction
- Learning curve tracking for training analysis

References:
    [1] Wiering, M. (2000). Multi-agent reinforcement learning for traffic
        light control. ICML.
    [2] Arel, I., Liu, C., Urbanik, T., & Kohls, A. G. (2010). Reinforcement
        learning-based multi-agent system for network traffic signal control.
        IET Intelligent Transport Systems, 4(2), 128-135.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import os
import random
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.entities.traffic_light import Direction
from src.core.state import SimulationState

@dataclass
class MARLMetrics:
    """Metrics for tracking MARL learning progress."""

    episode_rewards: List[float] = field(default_factory=list)
    per_agent_rewards: Dict[int, List[float]] = field(default_factory=dict)
    q_value_means: List[float] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    td_errors: List[float] = field(default_factory=list)

    def record_step(
        self,
        total_reward: float,
        agent_rewards: Dict[int, float],
        mean_q: float,
        epsilon: float,
        td_error: float = 0.0,
    ) -> None:
        """Record metrics for a single step."""
        self.episode_rewards.append(total_reward)
        self.q_value_means.append(mean_q)
        self.epsilon_values.append(epsilon)
        self.td_errors.append(td_error)

        for agent_id, reward in agent_rewards.items():
            if agent_id not in self.per_agent_rewards:
                self.per_agent_rewards[agent_id] = []
            self.per_agent_rewards[agent_id].append(reward)

        max_history = 1000
        if len(self.episode_rewards) > max_history:
            self.episode_rewards = self.episode_rewards[-max_history:]
            self.q_value_means = self.q_value_means[-max_history:]
            self.epsilon_values = self.epsilon_values[-max_history:]
            self.td_errors = self.td_errors[-max_history:]
            for agent_id in self.per_agent_rewards:
                self.per_agent_rewards[agent_id] = (
                    self.per_agent_rewards[agent_id][-max_history:]
                )

    def get_learning_curves(self) -> Dict[str, np.ndarray]:
        """Export learning curves for plotting."""
        return {
            "episode_rewards": np.array(self.episode_rewards),
            "q_value_means": np.array(self.q_value_means),
            "epsilon_values": np.array(self.epsilon_values),
            "td_errors": np.array(self.td_errors),
        }

class MARLAlgorithm(BaseAlgorithm):
    """
    Multi-Agent Reinforcement Learning for traffic signal control.

    Each intersection is an independent agent using tabular Q-learning.
    State representation includes local queue information, current phase,
    time in phase, and neighbor pressure for coordination.

    Features:
    - 8-dimensional discretized state space
    - Local reward function (queue reduction at each intersection)
    - Epsilon-greedy exploration with decay
    - Q-table initialization with traffic engineering heuristics
    """

    @property
    def name(self) -> str:
        return "marl"

    @property
    def display_name(self) -> str:
        return "Learning Agents (MARL)"

    @property
    def color(self) -> Tuple[int, int, int]:
        return Colors.MARL_COLOR

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.learning_rate = config.get("learning_rate", 0.01)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 0.1)
        self.epsilon_min = config.get("epsilon_min", 0.01)
        self.epsilon_decay = config.get("epsilon_decay", 0.9995)
        self.pretrained_path = config.get("pretrained_path")
        self.inference_only = config.get("inference_only", True)

        self.queue_bins = config.get("queue_bins", 5)
        self.time_bins = config.get("time_bins", 4)
        self.pressure_bins = config.get("pressure_bins", 3)

        self._q_tables: Dict[int, np.ndarray] = {}

        self._agent_states: Dict[int, Tuple] = {}
        self._agent_actions: Dict[int, int] = {}
        self._agent_confidences: Dict[int, float] = {}
        self._prev_queues: Dict[int, Dict[Direction, int]] = {}

        self._metrics = MARLMetrics()
        self._total_rewards = 0.0
        self._step_count = 0

        self._rng = np.random.default_rng()

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = np.random.default_rng(seed)
        random.seed(seed)

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize MARL agents."""
        self._network = network
        self._q_tables.clear()

        state_shape = (
            self.queue_bins,
            self.queue_bins,
            self.queue_bins,
            self.queue_bins,
            2,
            self.time_bins,
            self.pressure_bins,
        )
        n_actions = 2

        for int_id in network.intersections:
            self._q_tables[int_id] = np.zeros((*state_shape, n_actions))
            self._agent_states[int_id] = (0,) * 7
            self._agent_actions[int_id] = 0
            self._agent_confidences[int_id] = 0.0
            self._prev_queues[int_id] = {}

        self._neighbor_map = self._build_neighbor_map(network)

        loaded = False
        if self.pretrained_path:
            loaded = self._load_pretrained()

        if not loaded:
            self._initialize_heuristic_qtables()

        self._initialized = True

    def _build_neighbor_map(self, network: RoadNetwork) -> Dict[int, List[int]]:
        """Build map of neighboring intersections for each intersection."""
        neighbors: Dict[int, List[int]] = {
            int_id: [] for int_id in network.intersections
        }

        for road in network.roads.values():
            from_int = road.from_intersection
            to_int = road.to_intersection

            if from_int in neighbors and to_int is not None:
                if to_int not in neighbors[from_int]:
                    neighbors[from_int].append(to_int)

            if to_int in neighbors and from_int is not None:
                if from_int not in neighbors[to_int]:
                    neighbors[to_int].append(from_int)

        return neighbors

    def _load_pretrained(self) -> bool:
        """Try to load pretrained Q-tables from file."""
        try:
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                data = np.load(self.pretrained_path, allow_pickle=True)
                for int_id in self._q_tables:
                    key = str(int_id)
                    if key in data:
                        loaded_table = data[key]
                        if loaded_table.shape == self._q_tables[int_id].shape:
                            self._q_tables[int_id] = loaded_table
                return True
        except (FileNotFoundError, ValueError, KeyError) as e:
            import logging
            logging.warning(f"Failed to load MARL weights: {e}")
        return False

    def _initialize_heuristic_qtables(self) -> None:
        """
        Initialize Q-tables with traffic engineering heuristics.

        Based on traffic flow theory:
        - High queue differential -> favor switching to serve longer queue
        - Low time in phase -> favor keeping (stability)
        - High neighbor pressure -> consider switching to distribute load
        """
        for int_id in self._q_tables:
            q = self._q_tables[int_id]

            it = np.nditer(q[..., 0], flags=['multi_index'])
            while not it.finished:
                idx = it.multi_index
                q_n, q_s, q_e, q_w, phase, time_bin, pressure = idx

                ns_queue = q_n + q_s
                ew_queue = q_e + q_w

                if phase == 0:
                    served_queue = ns_queue
                    waiting_queue = ew_queue
                else:
                    served_queue = ew_queue
                    waiting_queue = ns_queue

                queue_diff = (waiting_queue - served_queue) / max(1, self.queue_bins)

                time_factor = time_bin / max(1, self.time_bins - 1)

                switch_preference = 0.3 * queue_diff + 0.2 * time_factor

                q[idx + (0,)] = 0.5 - switch_preference
                q[idx + (1,)] = 0.5 + switch_preference

                it.iternext()

    def _get_state(self, intersection, int_id: int) -> Tuple:
        """
        Get discretized state for an intersection.

        State vector (7 dimensions):
        - queue_north (discretized 0 to queue_bins-1)
        - queue_south (discretized)
        - queue_east (discretized)
        - queue_west (discretized)
        - current_phase (0=NS, 1=EW)
        - time_in_phase (discretized)
        - neighbor_pressure (discretized)
        """
        queues = intersection.get_queue_lengths()

        def discretize_queue(q: int) -> int:

            if q == 0:
                return 0
            elif q <= 2:
                return min(1, self.queue_bins - 1)
            elif q <= 4:
                return min(2, self.queue_bins - 1)
            elif q <= 7:
                return min(3, self.queue_bins - 1)
            else:
                return self.queue_bins - 1

        q_n = discretize_queue(queues.get(Direction.NORTH, 0))
        q_s = discretize_queue(queues.get(Direction.SOUTH, 0))
        q_e = discretize_queue(queues.get(Direction.EAST, 0))
        q_w = discretize_queue(queues.get(Direction.WEST, 0))

        tl = intersection.traffic_light
        if tl.current_phase.green_directions & {Direction.NORTH, Direction.SOUTH}:
            phase = 0
        else:
            phase = 1

        time_in_phase = tl.get_time_in_phase()
        phase_duration = tl.current_phase.duration
        time_ratio = min(1.0, time_in_phase / max(1, phase_duration))
        time_bin = min(int(time_ratio * self.time_bins), self.time_bins - 1)

        neighbor_pressure = self._get_neighbor_pressure(int_id)
        pressure_bin = min(
            int(neighbor_pressure / 5 * self.pressure_bins),
            self.pressure_bins - 1
        )

        return (q_n, q_s, q_e, q_w, phase, time_bin, pressure_bin)

    def _get_neighbor_pressure(self, int_id: int) -> float:
        """Get average queue pressure from neighboring intersections."""
        neighbors = self._neighbor_map.get(int_id, [])
        if not neighbors:
            return 0.0

        total_queue = 0
        count = 0

        for neighbor_id in neighbors:
            neighbor_int = self._network.intersections.get(neighbor_id)
            if neighbor_int:
                total_queue += neighbor_int.get_total_queue()
                count += 1

        return total_queue / max(1, count)

    def _compute_reward(
        self, int_id: int, intersection, old_queues: Dict[Direction, int]
    ) -> float:
        """
        Compute local reward based on queue reduction at this intersection.

        reward = queue_reduction_bonus - congestion_penalty
        """
        current_queues = intersection.get_queue_lengths()
        current_total = sum(current_queues.values())
        old_total = sum(old_queues.values())

        queue_reward = (old_total - current_total) * 0.5

        congestion_penalty = 0.0
        for direction, queue in current_queues.items():
            if queue > 8:
                congestion_penalty += 0.1 * (queue - 8) ** 1.5

        if current_total > 12:
            congestion_penalty += 0.2

        return queue_reward - congestion_penalty

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update MARL agents."""
        if not self.enabled or not self._initialized:
            return {}

        total_reward = 0.0
        agent_rewards: Dict[int, float] = {}
        q_values_sum = 0.0
        td_error_sum = 0.0
        n_agents = 0

        for int_id, intersection in self._network.intersections.items():

            current_state = self._get_state(intersection, int_id)

            if not self.inference_only and self._rng.random() < self.epsilon:
                action = int(self._rng.integers(2))
            else:
                q_values = self._q_tables[int_id][current_state]
                action = int(np.argmax(q_values))

            q_values = self._q_tables[int_id][current_state]
            q_values_sum += np.mean(np.abs(q_values))
            confidence = abs(q_values[0] - q_values[1])
            self._agent_confidences[int_id] = min(1.0, confidence)
            n_agents += 1

            if not self.inference_only:
                if action == 1:
                    intersection.request_phase_change()

                old_queues = self._prev_queues.get(int_id, {})
                if not old_queues:
                    old_queues = intersection.get_queue_lengths()

                reward = self._compute_reward(int_id, intersection, old_queues)
                agent_rewards[int_id] = reward
                total_reward += reward

                old_state = self._agent_states[int_id]
                old_action = self._agent_actions[int_id]

                if old_state != (0,) * 7:

                    old_q = self._q_tables[int_id][old_state + (old_action,)]
                    max_future_q = np.max(self._q_tables[int_id][current_state])
                    td_target = reward + self.gamma * max_future_q
                    td_error = td_target - old_q
                    new_q = old_q + self.learning_rate * td_error
                    self._q_tables[int_id][old_state + (old_action,)] = new_q
                    td_error_sum += abs(td_error)

                self._prev_queues[int_id] = dict(intersection.get_queue_lengths())

            self._agent_states[int_id] = current_state
            self._agent_actions[int_id] = action

        self._total_rewards += total_reward
        self._step_count += 1

        mean_q = q_values_sum / max(1, n_agents)
        mean_td = td_error_sum / max(1, n_agents)

        self._metrics.record_step(
            total_reward=total_reward,
            agent_rewards=agent_rewards,
            mean_q=mean_q,
            epsilon=self.epsilon,
            td_error=mean_td,
        )

        if not self.inference_only and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self._update_visualization(state)

        return {
            "total_reward": total_reward,
            "cumulative_reward": self._total_rewards,
            "epsilon": self.epsilon,
            "mean_q_value": mean_q,
            "inference_only": self.inference_only,
        }

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for MARL overlay with Q-value confidence."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        for int_id, intersection in self._network.intersections.items():
            current_state = self._agent_states.get(int_id, (0,) * 7)
            q_values = self._q_tables[int_id][current_state]

            confidence = abs(np.max(q_values) - np.min(q_values))
            normalized_conf = min(1.0, confidence / 1.0)

            best_action = np.argmax(q_values)

            if best_action == 0:
                color = Colors.MARL_COLOR
            else:
                color = (200, 140, 80)

            self._visualization.intersection_values[int_id] = normalized_conf
            self._visualization.intersection_colors[int_id] = color

            pos = intersection.position
            self._visualization.points.append(pos)
            self._visualization.point_colors.append(color)
            self._visualization.point_sizes.append(6.0 + normalized_conf * 4)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get MARL visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get MARL-specific metrics."""
        curves = self._metrics.get_learning_curves()

        avg_reward = 0.0
        if len(curves["episode_rewards"]) > 0:
            avg_reward = float(np.mean(curves["episode_rewards"][-100:]))

        avg_confidence = 0.0
        if self._agent_confidences:
            avg_confidence = sum(self._agent_confidences.values()) / len(
                self._agent_confidences
            )

        avg_td_error = 0.0
        if len(curves["td_errors"]) > 0:
            avg_td_error = float(np.mean(curves["td_errors"][-100:]))

        return {
            "cumulative_reward": self._total_rewards,
            "avg_reward": avg_reward,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "avg_confidence": avg_confidence,
            "avg_td_error": avg_td_error,
            "step_count": float(self._step_count),
            "inference_only": 1.0 if self.inference_only else 0.0,
        }

    def get_learning_curves(self) -> Dict[str, np.ndarray]:
        """Get learning curves for plotting."""
        return self._metrics.get_learning_curves()

    def reset(self) -> None:
        """Reset MARL algorithm."""
        super().reset()

        if self._q_tables:
            self._initialize_heuristic_qtables()

        self._total_rewards = 0.0
        self._step_count = 0
        self._metrics = MARLMetrics()
        self._agent_states.clear()
        self._agent_actions.clear()
        self._agent_confidences.clear()
        self._prev_queues.clear()

        self.epsilon = self.config.get("epsilon", 0.1)

    def save_qtables(self, path: str) -> None:
        """Save current Q-tables to file."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, **{str(k): v for k, v in self._q_tables.items()})

    def enable_training(self) -> None:
        """Enable training mode (actions affect signals, Q-tables update)."""
        self.inference_only = False
        self.epsilon = self.config.get("epsilon", 0.1)

    def enable_inference(self) -> None:
        """Enable inference mode (visualization only, no signal control)."""
        self.inference_only = True
        self.epsilon = 0.0
