"""Base class for visualization layers."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, Any
import pygame

from config.settings import Settings
from src.core.state import SimulationState


class BaseLayer(ABC):
    """
    Abstract base class for visualization layers.

    Each layer renders a specific aspect of the simulation.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.visible = True
        self.opacity = 1.0

    @property
    @abstractmethod
    def name(self) -> str:
        """Layer name for identification."""
        pass

    @abstractmethod
    def render(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """
        Render the layer to the surface.

        Args:
            surface: Surface to render to
            state: Current simulation state
            offset: Camera offset (x, y)
            zoom: Current zoom level
        """
        pass

    def handle_click(self, world_pos: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """
        Handle a click at world coordinates.

        Returns:
            Dictionary with clicked element info, or None
        """
        return None

    def set_visible(self, visible: bool) -> None:
        """Set layer visibility."""
        self.visible = visible

    def set_opacity(self, opacity: float) -> None:
        """Set layer opacity (0.0 - 1.0)."""
        self.opacity = max(0.0, min(1.0, opacity))

    def transform_point(
        self,
        point: Tuple[float, float],
        offset: Tuple[float, float],
        zoom: float
    ) -> Tuple[int, int]:
        """Transform world point to screen coordinates."""
        return (
            int((point[0] + offset[0]) * zoom),
            int((point[1] + offset[1]) * zoom)
        )
