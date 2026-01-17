"""Lightweight sparkline graph using native pygame."""

from typing import Tuple
import pygame
import numpy as np


class SparklineGraph:
    """Simple sparkline graph rendered with pygame.draw.lines()."""

    def __init__(
        self,
        width: int = 80,
        height: int = 20,
        color: Tuple[int, int, int] = (100, 200, 120),
        max_points: int = 60
    ) -> None:
        self.width = width
        self.height = height
        self.color = color
        self.max_points = max_points

    def render(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        data: np.ndarray
    ) -> None:
        """Render sparkline at position (x, y)."""
        if len(data) < 2:
            return

        # Take last max_points
        data = data[-self.max_points:]

        # Normalize to height
        min_val, max_val = float(data.min()), float(data.max())
        if max_val == min_val:
            max_val = min_val + 1  # Avoid division by zero

        # Build points
        points = []
        for i, val in enumerate(data):
            px = x + int((i / (len(data) - 1)) * self.width)
            py = y + self.height - int(((val - min_val) / (max_val - min_val)) * self.height)
            points.append((px, py))

        # Draw background rect (subtle)
        bg_rect = pygame.Rect(x, y, self.width, self.height)
        pygame.draw.rect(surface, (40, 42, 48), bg_rect, border_radius=2)

        # Draw line (single pygame call - fast)
        if len(points) >= 2:
            pygame.draw.lines(surface, self.color, False, points, 2)
