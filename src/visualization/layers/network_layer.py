"""Network layer for rendering roads and intersections."""

from typing import Tuple, Optional, Dict, Any, List
import pygame
import math

from config.settings import Settings
from config.colors import Colors
from src.core.state import SimulationState
from src.entities.network import RoadNetwork
from .base import BaseLayer


class NetworkLayer(BaseLayer):
    """
    Layer for rendering the road network.

    Draws roads, intersections, and road markings with a clean, polished style.
    """

    def __init__(self, network: RoadNetwork, settings: Settings) -> None:
        super().__init__(settings)
        self.network = network

        # Road rendering settings
        self.road_width = 22
        self.lane_width = 11
        self.intersection_radius = 28

        # Pre-render static elements (optimization)
        self._static_surface: Optional[pygame.Surface] = None
        self._needs_redraw = True

    @property
    def name(self) -> str:
        return "network"

    def render(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render the network."""
        if not self.visible:
            return

        # Render in order: roads, intersections (covers road ends)
        self._render_roads(surface, offset, zoom)
        self._render_intersections(surface, offset, zoom)

    def _render_roads(
        self,
        surface: pygame.Surface,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render all roads as clean rectangles."""
        # Group roads by connection to avoid double-drawing
        drawn_pairs = set()

        for road in self.network.roads.values():
            # Create a unique key for this road pair (bidirectional)
            pair_key = tuple(sorted([road.from_intersection, road.to_intersection]))
            if pair_key in drawn_pairs:
                continue
            drawn_pairs.add(pair_key)

            start = self.transform_point(road.start_pos, offset, zoom)
            end = self.transform_point(road.end_pos, offset, zoom)

            # Calculate road width based on lanes (both directions)
            width = int(self.lane_width * 2 * zoom)  # 2 lanes (each direction)

            # Draw road as a thick line with rounded caps
            self._draw_road_segment(surface, start, end, width, zoom)

    def _draw_road_segment(
        self,
        surface: pygame.Surface,
        start: Tuple[int, int],
        end: Tuple[int, int],
        width: int,
        zoom: float
    ) -> None:
        """Draw a road segment with rounded ends and lane markings."""
        # Calculate direction vector
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:
            return

        # Normalize direction
        dx /= length
        dy /= length

        # Perpendicular vector for width
        px, py = -dy, dx

        # Calculate rectangle corners
        half_width = width // 2
        corners = [
            (start[0] + px * half_width, start[1] + py * half_width),
            (start[0] - px * half_width, start[1] - py * half_width),
            (end[0] - px * half_width, end[1] - py * half_width),
            (end[0] + px * half_width, end[1] + py * half_width),
        ]
        corners = [(int(x), int(y)) for x, y in corners]

        # Draw road surface
        pygame.draw.polygon(surface, Colors.ROAD, corners)

        # Draw rounded end caps
        cap_radius = half_width
        pygame.draw.circle(surface, Colors.ROAD, start, cap_radius)
        pygame.draw.circle(surface, Colors.ROAD, end, cap_radius)

        # Draw center line (dashed) - lane divider between directions
        self._draw_dashed_line(surface, start, end, Colors.LANE_DIVIDER, zoom)

    def _draw_dashed_line(
        self,
        surface: pygame.Surface,
        start: Tuple[int, int],
        end: Tuple[int, int],
        color: Tuple[int, int, int],
        zoom: float
    ) -> None:
        """Draw a dashed center line."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        if length < 1:
            return

        # Normalize
        dx /= length
        dy /= length

        # Dash parameters (scaled by zoom)
        dash_length = int(12 * zoom)
        gap_length = int(8 * zoom)
        line_width = max(1, int(2 * zoom))

        # Draw dashes
        distance = 0
        while distance < length:
            dash_start = (
                int(start[0] + dx * distance),
                int(start[1] + dy * distance)
            )
            dash_end_dist = min(distance + dash_length, length)
            dash_end = (
                int(start[0] + dx * dash_end_dist),
                int(start[1] + dy * dash_end_dist)
            )

            pygame.draw.line(surface, color, dash_start, dash_end, line_width)

            distance += dash_length + gap_length

    def _render_intersections(
        self,
        surface: pygame.Surface,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render all intersections as clean circles."""
        for intersection in self.network.intersections.values():
            pos = self.transform_point(intersection.position, offset, zoom)
            radius = int(self.intersection_radius * zoom)

            # Draw intersection fill
            pygame.draw.circle(surface, Colors.INTERSECTION, pos, radius)

            # Draw subtle border
            border_width = max(1, int(2 * zoom))
            pygame.draw.circle(
                surface, Colors.INTERSECTION_BORDER, pos, radius, border_width
            )

    def handle_click(self, world_pos: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Check if an intersection or road was clicked."""
        # Check intersections first
        for int_id, intersection in self.network.intersections.items():
            dx = world_pos[0] - intersection.position[0]
            dy = world_pos[1] - intersection.position[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= self.intersection_radius:
                return {
                    "type": "intersection",
                    "id": int_id,
                    "position": intersection.position,
                }

        # Check roads
        for road_id, road in self.network.roads.items():
            if self._point_near_line(
                world_pos,
                road.start_pos,
                road.end_pos,
                self.road_width
            ):
                return {
                    "type": "road",
                    "id": road_id,
                    "from": road.from_intersection,
                    "to": road.to_intersection,
                }

        return None

    def _point_near_line(
        self,
        point: Tuple[float, float],
        line_start: Tuple[float, float],
        line_end: Tuple[float, float],
        threshold: float
    ) -> bool:
        """Check if a point is near a line segment."""
        px, py = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Line length squared
        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2) <= threshold

        # Project point onto line
        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))

        # Closest point on line
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)

        # Distance to closest point
        dist = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

        return dist <= threshold
