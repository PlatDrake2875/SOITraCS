"""
Enhanced Multi-Agent Reinforcement Learning (MARL) Algorithm.

Implements enhanced Q-learning with:
- Double Q-learning (reduces overestimation bias)
- Prioritized Experience Replay (improves sample efficiency)
- Target network updates (stabilizes learning)
- Per-direction (NS/EW) agents

References:
    [1] Van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double
        Q-learning. AAAI.
    [2] Schaul, T., et al. (2015). Prioritized Experience Replay. ICLR.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from ..base import BaseAlgorithm, AlgorithmVisualization
from .replay_buffer import PrioritizedReplayBuffer, Experience

from src.entities.network import RoadNetwork
from src.entities.traffic_light import Direction
from src.core.state import SimulationState

@dataclass
class EnhancedMARLMetrics:
    """Metrics tracking for Enhanced MARL."""

    reward_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    epsilon_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    q_value_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    episode_rewards: deque = field(default_factory=lambda: deque(maxlen=100))

    def record(
        self, avg_reward: float, epsilon: float, avg_q: float
    ) -> None:
        """Record metrics for current step."""
        self.reward_history.append(avg_reward)
        self.epsilon_history.append(epsilon)
        self.q_value_history.append(avg_q)

    def get_learning_curves(self) -> Dict[str, np.ndarray]:
        """Export learning curves for plotting."""
        return {
            "rewards": np.array(self.reward_history),
            "epsilon": np.array(self.epsilon_history),
            "q_values": np.array(self.q_value_history),
        }

class EnhancedMARLAlgorithm(BaseAlgorithm):
    """
    Enhanced Multi-Agent Reinforcement Learning for traffic signal control.

    Features:
    - Double Q-learning: Uses online network for action selection,
      target network for evaluation
    - Prioritized Experience Replay: Samples more important transitions
      more frequently
    - Per-direction agents: Separate agents for NS and EW phases
    - Target network soft updates: Periodic copy of Q-table to target
    """

    @property
    def name(self) -> str:
        return "marl_enhanced"

    @property
    def display_name(self) -> str:
        return "Enhanced MARL (DQN)"

    @property
    def color(self) -> Tuple[int, int, int]:
        return (255, 200, 100)

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__(config)

        self.learning_rate = config.get("learning_rate", 0.01)
        self.gamma = config.get("gamma", 0.95)
        self.epsilon = config.get("epsilon", 1.0)
        self.epsilon_decay = config.get("epsilon_decay", 0.9995)
        self.epsilon_min = config.get("epsilon_min", 0.01)

        self.double_q = config.get("double_q", True)
        self.experience_replay = config.get("experience_replay", True)
        self.prioritized_replay = config.get("prioritized_replay", True)

        self.buffer_size = config.get("buffer_size", 10000)
        self.batch_size = config.get("batch_size", 32)

        self.target_update_interval = config.get("target_update_interval", 100)

        self._q_tables: Dict[str, np.ndarray] = {}
        self._target_q_tables: Dict[str, np.ndarray] = {}

        self._replay_buffers: Dict[str, PrioritizedReplayBuffer] = {}

        self._agent_states: Dict[str, Tuple] = {}
        self._agent_actions: Dict[str, int] = {}
        self._prev_states: Dict[str, Optional[Tuple]] = {}

        self._metrics = EnhancedMARLMetrics()
        self._total_reward = 0.0
        self._tick = 0
        self._update_count = 0

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize agents for each intersection direction."""
        self._network = network

        for int_id, intersection in network.intersections.items():
            intersection.traffic_light.sotl_controlled = True

        n_states = 21 * 11 * 2 * 4 * 11 * 11
        n_actions = 2

        for int_id in network.intersections:
            for direction_group in ["ns", "ew"]:
                agent_id = f"{int_id}_{direction_group}"

                self._q_tables[agent_id] = np.random.uniform(-1, 1, (n_states, n_actions))
                self._target_q_tables[agent_id] = self._q_tables[agent_id].copy()

                if self.prioritized_replay:
                    self._replay_buffers[agent_id] = PrioritizedReplayBuffer(
                        self.buffer_size
                    )

                self._agent_states[agent_id] = (0, 0, 0, 0, 0, 0)
                self._agent_actions[agent_id] = 0
                self._prev_states[agent_id] = None

        self._initialized = True

    def _update_queue_counts(self, state: SimulationState) -> None:
        """Update queue counts at each intersection from vehicle positions."""
        if not self._network:
            return

        for intersection in self._network.intersections.values():
            for direction in Direction:
                road_id = intersection.get_incoming_road(direction)
                if road_id is not None:
                    road = self._network.get_road(road_id)
                    if road:
                        queue_len = road.get_queue_length()
                        intersection.update_queue_count(direction, queue_len)

    def _get_state_index(
        self,
        queue: int,
        avg_queue: float,
        phase: int,
        time_in_phase: int,
        incoming1: int = 0,
        incoming2: int = 0,
    ) -> int:
        """
        Convert state features to flat index.

        Args:
            queue: Current queue length (capped at 20)
            avg_queue: Moving average queue length (capped at 10)
            phase: Current phase (0 or 1)
            time_in_phase: Time in current phase
            incoming1: Incoming vehicles from first direction
            incoming2: Incoming vehicles from second direction

        Returns:
            Flat state index
        """
        q_idx = min(20, queue)
        avg_idx = min(10, int(avg_queue))
        inc1_idx = min(10, incoming1)
        inc2_idx = min(10, incoming2)

        time_bin = 0 if time_in_phase < 8 else 1 if time_in_phase < 16 else 2 if time_in_phase < 24 else 3

        return (
            q_idx * 11 * 2 * 4 * 11 * 11
            + avg_idx * 2 * 4 * 11 * 11
            + phase * 4 * 11 * 11
            + time_bin * 11 * 11
            + inc1_idx * 11
            + inc2_idx
        )

    def _get_state(
        self, intersection, direction_group: str
    ) -> Tuple[int, float, int, int, int, int]:
        """
        Get state features for an agent.

        Args:
            intersection: The intersection
            direction_group: "ns" or "ew"

        Returns:
            Tuple of (queue, avg_queue, phase, time, incoming1, incoming2)
        """
        queues = intersection.get_queue_lengths()
        tl = intersection.traffic_light

        if direction_group == "ns":

            queue = queues.get(Direction.NORTH, 0) + queues.get(Direction.SOUTH, 0)
            phase = 0 if Direction.NORTH in tl.current_phase.green_directions else 1
            incoming1 = queues.get(Direction.NORTH, 0)
            incoming2 = queues.get(Direction.SOUTH, 0)
        else:

            queue = queues.get(Direction.EAST, 0) + queues.get(Direction.WEST, 0)
            phase = 0 if Direction.EAST in tl.current_phase.green_directions else 1
            incoming1 = queues.get(Direction.EAST, 0)
            incoming2 = queues.get(Direction.WEST, 0)

        avg_queue = queue * 0.8

        time_in_phase = tl.get_time_in_phase()

        return (queue, avg_queue, phase, time_in_phase, incoming1, incoming2)

    def _compute_reward(
        self, intersection, action: int, prev_queue: int, direction_group: str
    ) -> float:
        """
        Compute reward for the agent.

        Rewards queue reduction, penalizes growing queues and frequent switching.
        """
        queues = intersection.get_queue_lengths()

        if direction_group == "ns":
            current_queue = queues.get(Direction.NORTH, 0) + queues.get(Direction.SOUTH, 0)
        else:
            current_queue = queues.get(Direction.EAST, 0) + queues.get(Direction.WEST, 0)

        reward = -current_queue

        if current_queue < prev_queue:
            reward += 2.0
        elif current_queue > prev_queue:
            reward -= 1.0

        if action == 1:
            reward -= 2.0

        if current_queue > 15:
            reward -= (current_queue - 15) * 0.5

        if current_queue < 5:
            reward += 1.0

        return reward

    def _update_q_value(
        self,
        agent_id: str,
        state: int,
        action: int,
        reward: float,
        next_state: int,
    ) -> None:
        """
        Update Q-value using Double Q-learning if enabled.

        Args:
            agent_id: Agent identifier
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
        """
        old_q = self._q_tables[agent_id][state, action]

        if self.double_q:

            best_action = np.argmax(self._q_tables[agent_id][next_state])

            max_future_q = self._target_q_tables[agent_id][next_state, best_action]
        else:

            max_future_q = np.max(self._target_q_tables[agent_id][next_state])

        self._q_tables[agent_id][state, action] = (
            old_q + self.learning_rate * (reward + self.gamma * max_future_q - old_q)
        )

    def _learn_from_replay(self, agent_id: str) -> None:
        """Learn from prioritized experience replay."""
        if not self._replay_buffers[agent_id].is_ready(self.batch_size):
            return

        beta = min(1.0, 0.4 + self._update_count * 0.0001)

        experiences, indices, weights = self._replay_buffers[agent_id].sample(
            self.batch_size, beta
        )

        td_errors = []

        for i, exp in enumerate(experiences):
            old_q = self._q_tables[agent_id][exp.state, exp.action]

            if self.double_q:
                best_action = np.argmax(self._q_tables[agent_id][exp.next_state])
                max_future_q = self._target_q_tables[agent_id][exp.next_state, best_action]
            else:
                max_future_q = np.max(self._target_q_tables[agent_id][exp.next_state])

            td_error = exp.reward + self.gamma * max_future_q - old_q

            self._q_tables[agent_id][exp.state, exp.action] = (
                old_q + self.learning_rate * weights[i] * td_error
            )

            td_errors.append(abs(td_error))

        self._replay_buffers[agent_id].update_priorities(indices, np.array(td_errors))

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update all agents."""
        if not self.enabled or not self._initialized:
            return {}

        self._tick += 1
        tick_reward = 0.0

        self._update_queue_counts(state)

        for agent_id in self._q_tables:

            int_id_str, direction_group = agent_id.rsplit("_", 1)
            int_id = int(int_id_str)

            intersection = self._network.intersections.get(int_id)
            if intersection is None:
                continue

            state_tuple = self._get_state(intersection, direction_group)
            queue, avg_q, phase, time_in_phase, inc1, inc2 = state_tuple
            current_state_idx = self._get_state_index(
                queue, avg_q, phase, time_in_phase, inc1, inc2
            )

            if np.random.random() < self.epsilon:
                action = np.random.randint(0, 2)
            else:
                action = int(np.argmax(self._q_tables[agent_id][current_state_idx]))

            if action == 1:
                intersection.traffic_light.sotl_controlled = True
                intersection.traffic_light.request_phase_change()

            prev_state_tuple = self._prev_states.get(agent_id)
            prev_action = self._agent_actions[agent_id]

            if prev_state_tuple is not None:
                prev_queue = prev_state_tuple[0]
                reward = self._compute_reward(intersection, prev_action, prev_queue, direction_group)
                tick_reward += reward

                prev_state_idx = self._get_state_index(*prev_state_tuple)

                if self.experience_replay and self.prioritized_replay:

                    experience = Experience(
                        prev_state_idx, prev_action, reward, current_state_idx, False
                    )

                    if self.double_q:
                        best_a = np.argmax(self._q_tables[agent_id][current_state_idx])
                        max_fq = self._target_q_tables[agent_id][current_state_idx, best_a]
                    else:
                        max_fq = np.max(self._target_q_tables[agent_id][current_state_idx])

                    old_q = self._q_tables[agent_id][prev_state_idx, prev_action]
                    td_error = reward + self.gamma * max_fq - old_q

                    self._replay_buffers[agent_id].add(experience, td_error)

                    self._learn_from_replay(agent_id)
                else:

                    self._update_q_value(
                        agent_id, prev_state_idx, prev_action, reward, current_state_idx
                    )

            self._prev_states[agent_id] = state_tuple
            self._agent_actions[agent_id] = action

        if self._update_count % self.target_update_interval == 0:
            for agent_id in self._q_tables:
                self._target_q_tables[agent_id] = self._q_tables[agent_id].copy()

        self._update_count += 1
        self._total_reward += tick_reward

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        avg_q = np.mean([np.max(q) for q in self._q_tables.values()])
        self._metrics.record(tick_reward, self.epsilon, avg_q)
        self._metrics.episode_rewards.append(tick_reward)

        self._update_visualization()

        return {
            "total_reward": tick_reward,
            "cumulative_reward": self._total_reward,
            "epsilon": self.epsilon,
            "update_count": self._update_count,
        }

    def _update_visualization(self) -> None:
        """Update visualization data."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        for int_id, intersection in self._network.intersections.items():

            for direction_group in ["ns", "ew"]:
                agent_id = f"{int_id}_{direction_group}"
                if agent_id not in self._q_tables:
                    continue

                state_tuple = self._get_state(intersection, direction_group)
                state_idx = self._get_state_index(*state_tuple)
                q_values = self._q_tables[agent_id][state_idx]

                confidence = min(1.0, abs(q_values[0] - q_values[1]) / 2.0)
                best_action = np.argmax(q_values)

                if best_action == 0:
                    color = self.color
                else:
                    color = (255, 140, 80)

                self._visualization.intersection_values[int_id] = confidence
                self._visualization.intersection_colors[int_id] = color

            pos = intersection.position
            self._visualization.points.append(pos)
            self._visualization.point_colors.append(self.color)
            self._visualization.point_sizes.append(8.0)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get algorithm-specific metrics."""
        avg_reward = 0.0
        if self._metrics.episode_rewards:
            avg_reward = np.mean(list(self._metrics.episode_rewards)[-100:])

        avg_q = 0.0
        if self._metrics.q_value_history:
            avg_q = np.mean(list(self._metrics.q_value_history)[-100:])

        return {
            "cumulative_reward": self._total_reward,
            "avg_reward": avg_reward,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "avg_q_value": avg_q,
            "update_count": float(self._update_count),
            "double_q": 1.0 if self.double_q else 0.0,
            "prioritized_replay": 1.0 if self.prioritized_replay else 0.0,
        }

    def reset(self) -> None:
        """Reset algorithm state."""
        super().reset()

        for agent_id in self._q_tables:
            self._q_tables[agent_id] = np.random.uniform(
                -1, 1, self._q_tables[agent_id].shape
            )
            self._target_q_tables[agent_id] = self._q_tables[agent_id].copy()

            if agent_id in self._replay_buffers:
                self._replay_buffers[agent_id] = PrioritizedReplayBuffer(self.buffer_size)

        self._prev_states.clear()
        self._agent_actions.clear()
        self._total_reward = 0.0
        self._tick = 0
        self._update_count = 0
        self._metrics = EnhancedMARLMetrics()
        self.epsilon = self.config.get("epsilon", 1.0)
