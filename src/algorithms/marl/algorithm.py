"""Multi-Agent Reinforcement Learning (MARL) algorithm for signal control."""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import os
from config.colors import Colors
from ..base import BaseAlgorithm, AlgorithmVisualization

from src.entities.network import RoadNetwork
from src.core.state import SimulationState


class MARLAlgorithm(BaseAlgorithm):
    """
    Multi-Agent Reinforcement Learning for traffic signal control.

    Each intersection is an agent with Q-values for signal decisions.
    By default, operates in inference_only mode where it computes Q-values
    and renders visualization halos but does NOT control signals (to avoid
    conflict with SOTL).

    Q-tables are initialized with traffic engineering heuristics.
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
        self.epsilon = config.get("epsilon", 0.0)  # No exploration in inference mode
        self.pretrained_path = config.get("pretrained_path")
        self.inference_only = config.get("inference_only", True)

        # Q-tables per intersection (tabular Q-learning)
        # State: discretized queue length (0-10)
        # Actions: 0 = keep phase, 1 = switch phase
        self._q_tables: Dict[int, np.ndarray] = {}

        # Agent states
        self._agent_states: Dict[int, int] = {}
        self._agent_actions: Dict[int, int] = {}
        self._agent_confidences: Dict[int, float] = {}

        # Metrics
        self._total_rewards = 0.0
        self._episode_rewards: List[float] = []

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize MARL agents."""
        self._network = network
        self._q_tables.clear()

        # Create Q-table for each intersection
        n_states = 11  # Queue length 0-10
        n_actions = 2  # Keep phase (0), Switch phase (1)

        for int_id in network.intersections:
            self._q_tables[int_id] = np.zeros((n_states, n_actions))
            self._agent_states[int_id] = 0
            self._agent_actions[int_id] = 0
            self._agent_confidences[int_id] = 0.0

        # Try to load pretrained weights
        loaded = False
        if self.pretrained_path:
            loaded = self._load_pretrained()

        # If no pretrained weights, initialize with heuristics
        if not loaded:
            self._initialize_heuristic_qtables()

        self._initialized = True

    def _load_pretrained(self) -> bool:
        """Try to load pretrained Q-tables from file."""
        try:
            if self.pretrained_path and os.path.exists(self.pretrained_path):
                import pickle
                with open(self.pretrained_path, 'rb') as f:
                    loaded_tables = pickle.load(f)
                    # Only load tables for intersections we have
                    for int_id in self._q_tables:
                        if int_id in loaded_tables:
                            self._q_tables[int_id] = loaded_tables[int_id]
                return True
        except Exception:
            pass
        return False

    def _initialize_heuristic_qtables(self) -> None:
        """
        Initialize Q-tables with traffic engineering heuristics.

        Based on traffic flow theory:
        - Low queue (0-3): Keep current phase (efficiency)
        - Medium queue (4-6): Slight preference to switch (responsiveness)
        - High queue (7+): Strong preference to switch (congestion relief)
        """
        for int_id in self._q_tables:
            q = self._q_tables[int_id]

            for state in range(11):
                if state <= 3:
                    # Low queue: favor keeping phase
                    q[state] = [0.8, 0.2]
                elif state <= 6:
                    # Medium queue: slight switch preference
                    q[state] = [0.45, 0.55]
                else:
                    # High queue: favor switching
                    q[state] = [0.15, 0.85]

    def update(self, state: SimulationState, dt: float) -> Dict[str, Any]:
        """Update MARL agents."""
        if not self.enabled or not self._initialized:
            return {}

        total_reward = 0.0

        for int_id, intersection in self._network.intersections.items():
            # Get current state (discretized queue length)
            queue = intersection.get_total_queue()
            current_state = min(10, queue)

            # Select action (epsilon-greedy, but epsilon=0 in inference mode)
            if np.random.random() < self.epsilon and not self.inference_only:
                action = np.random.randint(2)
            else:
                action = np.argmax(self._q_tables[int_id][current_state])

            # Calculate confidence (difference between Q-values)
            q_values = self._q_tables[int_id][current_state]
            confidence = abs(q_values[0] - q_values[1])
            self._agent_confidences[int_id] = min(1.0, confidence / 1.0)

            # ONLY execute action if NOT in inference_only mode
            # In inference_only mode, MARL is visualization-only
            if not self.inference_only:
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
            "inference_only": self.inference_only,
        }

    def _update_visualization(self, state: SimulationState) -> None:
        """Update visualization data for MARL overlay with Q-value confidence."""
        self._visualization = AlgorithmVisualization()

        if not self._network:
            return

        # Show Q-value confidence as halos around intersections
        for int_id, intersection in self._network.intersections.items():
            current_state = self._agent_states.get(int_id, 0)
            q_values = self._q_tables[int_id][current_state]

            # Confidence = difference between best and worst action
            confidence = abs(np.max(q_values) - np.min(q_values))
            normalized_conf = min(1.0, confidence / 1.0)

            # Best action determines color
            best_action = np.argmax(q_values)

            # Color encodes recommended action:
            # Keep phase (0) = MARL gold color
            # Switch phase (1) = Orange tint
            if best_action == 0:
                color = Colors.MARL_COLOR
            else:
                color = (200, 140, 80)  # Orange for switch recommendation

            self._visualization.intersection_values[int_id] = normalized_conf
            self._visualization.intersection_colors[int_id] = color

            # Add point for agent (for fallback rendering)
            pos = intersection.position
            self._visualization.points.append(pos)
            self._visualization.point_colors.append(color)
            self._visualization.point_sizes.append(6.0 + normalized_conf * 4)

    def get_visualization_data(self) -> AlgorithmVisualization:
        """Get MARL visualization data."""
        return self._visualization

    def get_metrics(self) -> Dict[str, float]:
        """Get MARL-specific metrics."""
        avg_reward = 0.0
        if self._episode_rewards:
            avg_reward = sum(self._episode_rewards) / len(self._episode_rewards)

        avg_confidence = 0.0
        if self._agent_confidences:
            avg_confidence = sum(self._agent_confidences.values()) / len(self._agent_confidences)

        return {
            "cumulative_reward": self._total_rewards,
            "avg_reward": avg_reward,
            "epsilon": self.epsilon,
            "learning_rate": self.learning_rate,
            "avg_confidence": avg_confidence,
            "inference_only": 1.0 if self.inference_only else 0.0,
        }

    def reset(self) -> None:
        """Reset MARL algorithm."""
        super().reset()

        # Re-initialize with heuristics (don't zero out)
        self._initialize_heuristic_qtables()

        self._total_rewards = 0.0
        self._episode_rewards.clear()
        self._agent_states.clear()
        self._agent_actions.clear()
        self._agent_confidences.clear()

    def save_qtables(self, path: str) -> None:
        """Save current Q-tables to file."""
        import pickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self._q_tables, f)
