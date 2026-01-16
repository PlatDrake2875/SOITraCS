"""Main render coordinator for the simulation."""

from typing import Dict, List, Optional, Tuple
import pygame

from config.colors import Colors
from config.settings import Settings
from src.core.state import SimulationState
from src.entities.network import RoadNetwork

from .layers.base import BaseLayer
from .layers.network_layer import NetworkLayer
from .layers.vehicle_layer import VehicleLayer
from .layers.signal_layer import SignalLayer
from .layers.overlay_layer import OverlayLayer
from .dashboard.panel import DashboardPanel


class Renderer:
    """
    Main render coordinator.

    Manages all render layers and coordinates drawing to the screen.
    """

    def __init__(
        self,
        screen: pygame.Surface,
        settings: Settings,
    ) -> None:
        self.screen = screen
        self.settings = settings

        # Render regions
        self.sim_rect = pygame.Rect(
            0, 0,
            settings.display.simulation_width,
            settings.display.simulation_height
        )
        self.dashboard_rect = pygame.Rect(
            settings.display.simulation_width, 0,
            settings.display.dashboard_width,
            settings.display.window_height
        )

        # Create simulation surface
        self.sim_surface = pygame.Surface(
            (self.sim_rect.width, self.sim_rect.height)
        )

        # Layers (in draw order)
        self._layers: List[BaseLayer] = []
        self._layer_visibility: Dict[str, bool] = {}

        # Dashboard
        self.dashboard: Optional[DashboardPanel] = None

        # Camera/viewport
        self.camera_offset = (0, 0)
        self.zoom = 1.0

        # Fonts
        pygame.font.init()
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)

    def initialize(self, network: RoadNetwork, state: SimulationState) -> None:
        """Initialize all layers with the network."""
        self._layers.clear()

        # Create layers in draw order
        self._layers.append(NetworkLayer(network, self.settings))
        self._layers.append(SignalLayer(network, self.settings))
        self._layers.append(VehicleLayer(network, self.settings))
        self._layers.append(OverlayLayer(network, state, self.settings))

        # Set all layers visible by default
        for layer in self._layers:
            self._layer_visibility[layer.name] = True

        # Create dashboard
        self.dashboard = DashboardPanel(
            self.dashboard_rect,
            self.settings,
            state
        )

        # Center camera on network
        bounds = network.get_bounds()
        center_x = (bounds[0] + bounds[2]) / 2
        center_y = (bounds[1] + bounds[3]) / 2
        self.camera_offset = (
            self.sim_rect.width / 2 - center_x,
            self.sim_rect.height / 2 - center_y
        )

    def render(self, state: SimulationState, dt: float) -> None:
        """Render the complete frame."""
        # Clear screen
        self.screen.fill(Colors.BACKGROUND)

        # Render simulation area
        self._render_simulation(state, dt)

        # Render dashboard
        if self.dashboard:
            self.dashboard.render(self.screen, state)

        # Draw divider line
        pygame.draw.line(
            self.screen,
            Colors.PANEL_BORDER,
            (self.sim_rect.right, 0),
            (self.sim_rect.right, self.sim_rect.bottom),
            2
        )

    def _render_simulation(self, state: SimulationState, dt: float) -> None:
        """Render the simulation view."""
        # Clear simulation surface
        self.sim_surface.fill(Colors.BACKGROUND)

        # Apply camera transform
        # (In future: implement zoom and pan)

        # Render each visible layer
        for layer in self._layers:
            if self._layer_visibility.get(layer.name, True):
                layer.render(self.sim_surface, state, self.camera_offset, self.zoom)

        # Blit to main screen
        self.screen.blit(self.sim_surface, self.sim_rect)

    def toggle_layer(self, layer_name: str) -> bool:
        """Toggle layer visibility. Returns new state."""
        current = self._layer_visibility.get(layer_name, True)
        self._layer_visibility[layer_name] = not current
        return self._layer_visibility[layer_name]

    def set_layer_visibility(self, layer_name: str, visible: bool) -> None:
        """Set layer visibility."""
        self._layer_visibility[layer_name] = visible

    def get_layer_visibility(self) -> Dict[str, bool]:
        """Get all layer visibility states."""
        return dict(self._layer_visibility)

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Dict]:
        """Handle mouse click. Returns clicked element info or None."""
        # Check if click is in dashboard
        if self.dashboard and self.dashboard_rect.collidepoint(pos):
            return self.dashboard.handle_click(pos)

        # Check if click is in simulation area
        if self.sim_rect.collidepoint(pos):
            # Transform to world coordinates
            world_x = pos[0] - self.camera_offset[0]
            world_y = pos[1] - self.camera_offset[1]

            # Check layers in reverse order (top to bottom)
            for layer in reversed(self._layers):
                result = layer.handle_click((world_x, world_y))
                if result:
                    return result

        return None

    def world_to_screen(self, world_pos: Tuple[float, float]) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        return (
            int(world_pos[0] + self.camera_offset[0]),
            int(world_pos[1] + self.camera_offset[1])
        )

    def screen_to_world(self, screen_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        return (
            screen_pos[0] - self.camera_offset[0],
            screen_pos[1] - self.camera_offset[1]
        )

    def set_zoom(self, zoom: float) -> None:
        """Set zoom level."""
        self.zoom = max(0.25, min(4.0, zoom))

    def pan(self, dx: int, dy: int) -> None:
        """Pan the camera."""
        self.camera_offset = (
            self.camera_offset[0] + dx,
            self.camera_offset[1] + dy
        )
