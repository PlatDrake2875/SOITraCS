"""
Prioritized Experience Replay Buffer for Enhanced MARL.

Implements a replay buffer with prioritized sampling based on TD-error,
allowing more important experiences to be sampled more frequently.

References:
    [1] Schaul, T., et al. (2015). Prioritized Experience Replay. ICLR.
"""

from dataclasses import dataclass
from typing import List, Tuple, Any
import numpy as np

@dataclass
class Experience:
    """Single experience tuple for replay buffer."""

    state: int
    action: int
    reward: float
    next_state: int
    done: bool

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer.

    Experiences with higher TD-error are sampled more frequently,
    improving learning efficiency for rare but important transitions.

    Args:
        capacity: Maximum buffer size
        alpha: Priority exponent (0 = uniform, 1 = full prioritization)
    """

    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        """
        Initialize the prioritized replay buffer.

        Args:
            capacity: Maximum number of experiences to store
            alpha: Priority exponent controlling how much prioritization is used
        """
        self.capacity = capacity
        self.alpha = alpha

        self._buffer: List[Experience] = []
        self._priorities: np.ndarray = np.array([])
        self._position: int = 0

    def add(self, experience: Experience, td_error: float | None = None) -> None:
        """
        Add experience to the buffer.

        Args:
            experience: Experience tuple to add
            td_error: Optional TD-error for initial priority (uses max if not provided)
        """

        max_priority = self._priorities.max() if len(self._priorities) > 0 else 1.0

        if len(self._buffer) < self.capacity:
            self._buffer.append(experience)
            self._priorities = np.append(self._priorities, max_priority)
        else:
            self._buffer[self._position] = experience
            self._priorities[self._position] = max_priority

        self._position = (self._position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """
        Sample a batch with prioritized sampling.

        Args:
            batch_size: Number of experiences to sample
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)

        Returns:
            Tuple of (experiences, indices, importance_weights)
        """

        actual_batch_size = min(batch_size, len(self._buffer))

        priorities = self._priorities[: len(self._buffer)]
        probs = priorities**self.alpha
        probs /= probs.sum()

        indices = np.random.choice(
            len(self._buffer), actual_batch_size, p=probs, replace=False
        )

        weights = (len(self._buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()

        experiences = [self._buffer[i] for i in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """
        Update priorities based on new TD-errors.

        Args:
            indices: Indices of experiences to update
            td_errors: New TD-errors for each experience
        """
        for idx, td_error in zip(indices, td_errors):
            self._priorities[idx] = abs(td_error) + 1e-6

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self._buffer) >= batch_size
