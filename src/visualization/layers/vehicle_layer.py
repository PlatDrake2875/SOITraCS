"""Vehicle layer for rendering vehicles with interpolation."""

from typing import Tuple, Optional, Dict, Any, List
import pygame
import math

from config.settings import Settings
from config.colors import Colors
from src.core.state import SimulationState
from src.entities.network import RoadNetwork
from src.entities.vehicle import Vehicle, VehicleState
from .base import BaseLayer

class VehicleLayer(BaseLayer):
    """
    Layer for rendering vehicles.

    Supports interpolation for smooth movement between ticks.
    Vehicles are colored by speed: red (stopped) to white (moving).
    """

    def __init__(self, network: RoadNetwork, settings: Settings) -> None:
        super().__init__(settings)
        self.network = network

        self.vehicle_length = 14
        self.vehicle_width = 7

        self.selected_vehicle_id: Optional[int] = None

    @property
    def name(self) -> str:
        return "vehicles"

    def render(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render all vehicles."""
        if not self.visible:
            return

        for vehicle in state.vehicles.values():
            self._render_vehicle(surface, vehicle, offset, zoom)

    def _render_vehicle(
        self,
        surface: pygame.Surface,
        vehicle: Vehicle,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render a single vehicle with rounded corners and speed-based coloring."""

        world_pos = vehicle.get_world_position(self.network)
        if world_pos is None:
            return

        pos = self.transform_point(world_pos, offset, zoom)

        road = self.network.get_road(vehicle.current_road_id)
        if road:
            dx = road.end_pos[0] - road.start_pos[0]
            dy = road.end_pos[1] - road.start_pos[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0:
                dx /= length
                dy /= length
            else:
                dx, dy = 1, 0
        else:
            dx, dy = 1, 0

        vlen = int(self.vehicle_length * zoom * 0.5)
        vwid = int(self.vehicle_width * zoom * 0.5)

        px, py = -dy, dx

        if vehicle.id == self.selected_vehicle_id:
            color = Colors.VEHICLE_SELECTED
        else:
            speed_ratio = vehicle.speed / max(1, vehicle.max_speed)
            color = Colors.speed_color(speed_ratio)

        shadow_offset = max(1, int(2 * zoom))
        shadow_corners = self._get_vehicle_corners(
            (pos[0] + shadow_offset, pos[1] + shadow_offset),
            dx, dy, px, py, vlen, vwid
        )
        self._draw_rounded_polygon(surface, shadow_corners, Colors.VEHICLE_SHADOW, zoom)

        corners = self._get_vehicle_corners(pos, dx, dy, px, py, vlen, vwid)
        self._draw_rounded_polygon(surface, corners, color, zoom)

        self._draw_windshield(surface, pos, dx, dy, px, py, vlen, vwid, color, zoom)

    def _get_vehicle_corners(
        self,
        pos: Tuple[int, int],
        dx: float, dy: float,
        px: float, py: float,
        vlen: int, vwid: int
    ) -> List[Tuple[int, int]]:
        """Calculate the four corners of a vehicle."""
        corners = [
            (pos[0] + dx * vlen + px * vwid, pos[1] + dy * vlen + py * vwid),
            (pos[0] + dx * vlen - px * vwid, pos[1] + dy * vlen - py * vwid),
            (pos[0] - dx * vlen - px * vwid, pos[1] - dy * vlen - py * vwid),
            (pos[0] - dx * vlen + px * vwid, pos[1] - dy * vlen + py * vwid),
        ]
        return [(int(x), int(y)) for x, y in corners]

    def _draw_rounded_polygon(
        self,
        surface: pygame.Surface,
        corners: List[Tuple[int, int]],
        color: Tuple[int, int, int],
        zoom: float
    ) -> None:
        """Draw a polygon with rounded corners (approximation)."""

        pygame.draw.polygon(surface, color, corners)

        outline_color = Colors.lerp(color, Colors.BACKGROUND, 0.3)
        pygame.draw.polygon(surface, outline_color, corners, 1)

    def _draw_windshield(
        self,
        surface: pygame.Surface,
        pos: Tuple[int, int],
        dx: float, dy: float,
        px: float, py: float,
        vlen: int, vwid: int,
        body_color: Tuple[int, int, int],
        zoom: float
    ) -> None:
        """Draw a subtle windshield area on the vehicle."""

        windshield_ratio = 0.35
        ws_len = int(vlen * windshield_ratio)
        ws_wid = int(vwid * 0.7)

        ws_center_x = pos[0] + dx * (vlen * 0.4)
        ws_center_y = pos[1] + dy * (vlen * 0.4)

        ws_corners = [
            (ws_center_x + dx * ws_len + px * ws_wid, ws_center_y + dy * ws_len + py * ws_wid),
            (ws_center_x + dx * ws_len - px * ws_wid, ws_center_y + dy * ws_len - py * ws_wid),
            (ws_center_x - dx * ws_len - px * ws_wid, ws_center_y - dy * ws_len - py * ws_wid),
            (ws_center_x - dx * ws_len + px * ws_wid, ws_center_y - dy * ws_len + py * ws_wid),
        ]
        ws_corners = [(int(x), int(y)) for x, y in ws_corners]

        windshield_color = Colors.lerp(body_color, Colors.BACKGROUND, 0.25)
        pygame.draw.polygon(surface, windshield_color, ws_corners)

    def handle_click(self, world_pos: Tuple[float, float]) -> Optional[Dict[str, Any]]:
        """Check if a vehicle was clicked."""
        return None

    def select_vehicle(self, vehicle_id: Optional[int]) -> None:
        """Select a vehicle for highlighting."""
        self.selected_vehicle_id = vehicle_id
