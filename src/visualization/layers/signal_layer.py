"""Signal layer for rendering traffic lights."""

from typing import Tuple, Optional, Dict, Any
import pygame
import math

from config.settings import Settings
from config.colors import Colors
from src.core.state import SimulationState
from src.entities.network import RoadNetwork
from src.entities.traffic_light import SignalState, Direction
from .base import BaseLayer


class SignalLayer(BaseLayer):
    """
    Layer for rendering traffic signals at intersections.

    Shows signals as compact light strips on intersection edges,
    grouped by phase (NS vs EW) for visual clarity.
    """

    def __init__(self, network: RoadNetwork, settings: Settings) -> None:
        super().__init__(settings)
        self.network = network

        # Signal rendering settings
        self.strip_length = 16  # Length of signal strip
        self.strip_width = 5   # Width of signal strip
        self.strip_offset = 22  # Distance from intersection center

    @property
    def name(self) -> str:
        return "signals"

    def render(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render all traffic signals."""
        if not self.visible:
            return

        for intersection in self.network.intersections.values():
            self._render_intersection_signals(surface, intersection, offset, zoom)

    def _render_intersection_signals(
        self,
        surface: pygame.Surface,
        intersection,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render signals for one intersection as compact light strips."""
        center = self.transform_point(intersection.position, offset, zoom)
        strip_offset = int(self.strip_offset * zoom)
        strip_length = int(self.strip_length * zoom)
        strip_width = max(2, int(self.strip_width * zoom))

        # Get signal states for NS and EW groups
        ns_directions = [Direction.NORTH, Direction.SOUTH]
        ew_directions = [Direction.EAST, Direction.WEST]

        # Determine phase colors
        ns_state = self._get_phase_state(intersection, ns_directions)
        ew_state = self._get_phase_state(intersection, ew_directions)

        ns_color = self._state_to_color(ns_state)
        ew_color = self._state_to_color(ew_state)

        # Draw NS signal strips (top and bottom of intersection)
        # North strip
        if intersection.get_incoming_road(Direction.NORTH):
            self._draw_signal_strip(
                surface, center,
                (0, -strip_offset),
                strip_length, strip_width,
                ns_color, horizontal=True, zoom=zoom
            )

        # South strip
        if intersection.get_incoming_road(Direction.SOUTH):
            self._draw_signal_strip(
                surface, center,
                (0, strip_offset),
                strip_length, strip_width,
                ns_color, horizontal=True, zoom=zoom
            )

        # Draw EW signal strips (left and right of intersection)
        # West strip
        if intersection.get_incoming_road(Direction.WEST):
            self._draw_signal_strip(
                surface, center,
                (-strip_offset, 0),
                strip_length, strip_width,
                ew_color, horizontal=False, zoom=zoom
            )

        # East strip
        if intersection.get_incoming_road(Direction.EAST):
            self._draw_signal_strip(
                surface, center,
                (strip_offset, 0),
                strip_length, strip_width,
                ew_color, horizontal=False, zoom=zoom
            )

        # Draw subtle phase timer arc
        self._draw_phase_timer(surface, intersection, center, zoom)

    def _get_phase_state(self, intersection, directions) -> SignalState:
        """Get the signal state for a group of directions."""
        for direction in directions:
            if intersection.get_incoming_road(direction):
                return intersection.get_signal_state(direction)
        return SignalState.RED

    def _state_to_color(self, state: SignalState) -> Tuple[int, int, int]:
        """Convert signal state to color."""
        if state == SignalState.GREEN:
            return Colors.SIGNAL_GREEN
        elif state == SignalState.YELLOW:
            return Colors.SIGNAL_YELLOW
        else:
            return Colors.SIGNAL_RED

    def _draw_signal_strip(
        self,
        surface: pygame.Surface,
        center: Tuple[int, int],
        offset: Tuple[int, int],
        length: int,
        width: int,
        color: Tuple[int, int, int],
        horizontal: bool,
        zoom: float
    ) -> None:
        """Draw a signal strip (compact rectangle with glow)."""
        pos_x = center[0] + offset[0]
        pos_y = center[1] + offset[1]

        if horizontal:
            rect = pygame.Rect(
                pos_x - length // 2,
                pos_y - width // 2,
                length,
                width
            )
        else:
            rect = pygame.Rect(
                pos_x - width // 2,
                pos_y - length // 2,
                width,
                length
            )

        # Draw glow (slightly larger, semi-transparent)
        glow_expand = max(2, int(3 * zoom))
        glow_rect = rect.inflate(glow_expand * 2, glow_expand * 2)
        glow_color = Colors.lerp(color, Colors.BACKGROUND, 0.6)
        pygame.draw.rect(surface, glow_color, glow_rect, border_radius=3)

        # Draw main signal strip
        pygame.draw.rect(surface, color, rect, border_radius=2)

    def _draw_phase_timer(
        self,
        surface: pygame.Surface,
        intersection,
        center: Tuple[int, int],
        zoom: float
    ) -> None:
        """Draw a subtle arc showing phase progress."""
        # Get current phase progress (0.0 to 1.0)
        if hasattr(intersection, 'get_phase_progress'):
            progress = intersection.get_phase_progress()
        else:
            # Fallback: estimate from timer if available
            progress = 0.0
            if hasattr(intersection, 'phase_timer') and hasattr(intersection, 'current_phase'):
                phase = intersection.current_phase
                if phase and hasattr(phase, 'duration') and phase.duration > 0:
                    progress = min(1.0, intersection.phase_timer / phase.duration)

        if progress <= 0:
            return

        # Draw arc
        arc_radius = int(18 * zoom)
        arc_width = max(1, int(2 * zoom))
        arc_rect = pygame.Rect(
            center[0] - arc_radius,
            center[1] - arc_radius,
            arc_radius * 2,
            arc_radius * 2
        )

        # Arc from top, clockwise based on progress
        start_angle = math.pi / 2  # Top
        end_angle = start_angle - (progress * 2 * math.pi)

        # Draw arc with subtle color
        arc_color = Colors.lerp(Colors.TEXT_SECONDARY, Colors.BACKGROUND, 0.5)
        pygame.draw.arc(surface, arc_color, arc_rect, end_angle, start_angle, arc_width)

    def handle_click(self, world_pos: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Check if a signal was clicked."""
        return None
