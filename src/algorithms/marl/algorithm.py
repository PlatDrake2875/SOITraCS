"""Multi-Agent Reinforcement Learning (MARL) algorithm for signal control."""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState


class MARLAlgorithm(BaseAlgorithm):
    """
    Multi-Agent Reinforcement Learning for traffic signal control.

    Each intersection is an agent that learns to optimize signal timing.
    Uses pre-trained weights for inference (no online learning by default).
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

        # MARL parameters
        self.learning_rate = config.get("learning_rate", 0.001)
        self.gamma = config.get("gamma", 0.99)
        self.epsilon = config.get("epsilon", 0.1)
        self.pretrained_path = config.get("pretrained_path")
        self.inference_only = config.get("inference_only", True)

        # Q-tables per intersection (simplified tabular Q-learning)
        self._q_tables: Dict[int, np.ndarray] = {}

        # Agent states
        self._agent_states: Dict[int, int] = {}
        self._agent_actions: Dict[int, int] = {}

        # Metrics
        self._total_rewards = 0.0
        self._episode_rewards: List[float] = []

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize MARL agents."""
        self._network = network
        self._q_tables.clear()

        # Create Q-table for each intersection
        # State space: discretized queue length (0-10)
        # Action space: 2 (keep phase, switch phase)
        n_states = 11
        n_actions = 2

        for int_id in network.intersections:
            self._q_tables[int_id] = np.zeros((n_states, n_actions))
            self._agent_states[int_id] = 0
            self._agent_actions[int_id] = 0

        # Try to load pretrained weights
        if self.pretrained_path:
            try:
                import pickle
                with open(self.pretrained_path, 'rb') as f:
                    self._q_tables = pickle.load(f)
            except (FileNotFoundError, Exception):
                pass

        self._initialized = True

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update MARL agents."""
        if not self.enabled or not self._initialized:
            return {}

        total_reward = 0.0

        for int_id, intersection in self._network.intersections.items():
            # Get current state (discretized queue length)
            queue = intersection.get_total_queue()
            current_state = min(10, queue)

            # Select action (epsilon-greedy)
            if np.random.random() < self.epsilon and not self.inference_only:
                action = np.random.randint(2)
            else:
                action = np.argmax(self._q_tables[int_id][current_state])

            # Execute action
            if action == 1:  # Switch phase
                intersection.request_phase_change()

            # Calculate reward (negative delay)
            reward = -state.current_metrics.average_delay

            # Q-learning update (only if not inference-only)
            if not self.inference_only:
                old_state = self._agent_states[int_id]
                old_action = self._agent_actions[int_id]

                # Q(s,a) = Q(s,a) + lr * (r + gamma * max(Q(s',a')) - Q(s,a))
                old_q = self._q_tables[int_id][old_state, old_action]
                max_future_q = np.max(self._q_tables[int_id][current_state])
                new_q = old_q + self.learning_rate * (
                    reward + self.gamma * max_future_q - old_q
                )
                self._q_tables[int_id][old_state, old_action] = new_q

            # Store state/action for next update
            self._agent_states[int_id] = current_state
            self._agent_actions[int_id] = action
            total_reward += reward

        self._total_rewards += total_reward
        self._episode_rewards.append(total_reward)
        if len(self._episode_rewards) > 300:
            self._episode_rewards = self._episode_rewards[-300:]

        # Update visualization
        self._update_visualization(state)

        return {
            "total_reward": total_reward,
            "cumulative_reward": self._total_rewards,
        }

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for MARL overlay."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        # Show Q-value confidence as halos around intersections
        for int_id, intersection in self._network.intersections.items():
            current_state = self._agent_states.get(int_id, 0)
            q_values = self._q_tables[int_id][current_state]

            # Confidence = difference between best and worst action
            confidence = np.max(q_values) - np.min(q_values)
            normalized_conf = min(1.0, confidence / 10.0)

            self._visualization.intersection_values[int_id] = normalized_conf
            self._visualization.intersection_colors[int_id] = Colors.lerp(
                (100, 100, 100),
                Colors.MARL_COLOR,
                normalized_conf
            )

            # Add point for agent
            pos = intersection.position
            self._visualization.points.append(pos)
            self._visualization.point_colors.append(Colors.MARL_COLOR)
            self._visualization.point_sizes.append(8.0 + normalized_conf * 4)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get MARL visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get MARL-specific metrics."""
        avg_reward = 0.0
        if self._episode_rewards:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)

        return {
            "cumulative_reward": self._total_rewards,
            "avg_reward": avg_reward,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
        }

    def reset(self) -> None:
        """Reset MARL algorithm."""
        super().reset()

        # Reset Q-tables to zero
        for int_id in self._q_tables:
            self._q_tables[int_id].fill(0)

        self._total_rewards = 0.0
        self._episode_rewards.clear()
        self._agent_states.clear()
        self._agent_actions.clear()
