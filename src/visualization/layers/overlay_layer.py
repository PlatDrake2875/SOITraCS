"""Overlay layer for algorithm-specific visualizations."""

from typing import Tuple, Optional, Dict, Any, List
import pygame

from config.settings import Settings
from config.colors import Colors
from src.core.state import SimulationState
from src.entities.network import RoadNetwork
from src.algorithms.base import AlgorithmVisualization
from .base import BaseLayer


class OverlayLayer(BaseLayer):
    """
    Layer for rendering algorithm-specific overlays.

    Composites visualization data from all active algorithms with
    subtle, non-intrusive effects (reduced opacity, thin lines).
    """

    # Default opacity values - subtle overlays
    DEFAULT_OPACITY = 0.35
    LINE_OPACITY = 0.4
    POINT_OPACITY = 0.5

    def __init__(
        self,
        network: RoadNetwork,
        state: SimulationState,
        settings: Settings
    ) -> None:
        super().__init__(settings)
        self.network = network
        self.state = state

        # Per-algorithm visibility (defaults to False for subtle appearance)
        self._algorithm_visibility: Dict[str, bool] = {}
        self._algorithm_opacity: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return "overlays"

    def render(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render all algorithm overlays."""
        if not self.visible:
            return

        # Get visualization data from each active algorithm
        for algo_name, algo in state.algorithms.items():
            if not algo.enabled:
                continue

            if not self._algorithm_visibility.get(algo_name, False):
                continue

            vis_data = algo.get_visualization_data()
            opacity = self._algorithm_opacity.get(algo_name, self.DEFAULT_OPACITY)

            self._render_algorithm_overlay(surface, algo_name, vis_data, offset, zoom, opacity)

    def _render_algorithm_overlay(
        self,
        surface: pygame.Surface,
        algo_name: str,
        vis_data: AlgorithmVisualization,
        offset: Tuple[float, float],
        zoom: float,
        opacity: float
    ) -> None:
        """Render overlay data from one algorithm with style appropriate to algorithm."""
        if not vis_data.show_overlay:
            return

        # Use algorithm-specific rendering for cleaner appearance
        if algo_name == "sotl":
            self._render_sotl_overlay(surface, vis_data, offset, zoom, opacity)
        elif algo_name == "aco":
            self._render_aco_overlay(surface, vis_data, offset, zoom, opacity)
        else:
            # Default rendering for other algorithms
            self._render_road_overlays(surface, vis_data, offset, zoom, opacity)
            self._render_lines(surface, vis_data, offset, zoom, opacity)
            self._render_points(surface, vis_data, offset, zoom)

    def _render_sotl_overlay(
        self,
        surface: pygame.Surface,
        vis_data: AlgorithmVisualization,
        offset: Tuple[float, float],
        zoom: float,
        opacity: float
    ) -> None:
        """Render SOTL overlay as subtle glow on active signals only."""
        # Only render subtle glow on intersections with high activity
        overlay_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        for int_id, color in vis_data.intersection_colors.items():
            intersection = self.network.get_intersection(int_id)
            if not intersection:
                continue

            pos = self.transform_point(intersection.position, offset, zoom)
            value = vis_data.intersection_values.get(int_id, 0.0)

            # Only show for significant values
            if value < 0.2:
                continue

            # Subtle glow ring instead of filled circle
            alpha = int(80 * opacity * value)
            glow_radius = int(20 * zoom)
            glow_width = max(2, int(3 * zoom))

            pygame.draw.circle(
                overlay_surface,
                (*Colors.SOTL_COLOR, alpha),
                pos,
                glow_radius,
                glow_width
            )

        surface.blit(overlay_surface, (0, 0))

    def _render_aco_overlay(
        self,
        surface: pygame.Surface,
        vis_data: AlgorithmVisualization,
        offset: Tuple[float, float],
        zoom: float,
        opacity: float
    ) -> None:
        """Render ACO overlay as thin semi-transparent pheromone trails."""
        overlay_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        # Render road pheromone levels as thin colored lines
        for road_id, color in vis_data.road_colors.items():
            road = self.network.get_road(road_id)
            if not road:
                continue

            start = self.transform_point(road.start_pos, offset, zoom)
            end = self.transform_point(road.end_pos, offset, zoom)

            value = vis_data.road_values.get(road_id, 0.0)

            # Only show significant pheromone levels
            if value < 0.1:
                continue

            # Thin line with transparency based on pheromone strength
            alpha = int(100 * opacity * value)
            line_width = max(2, int(3 * zoom * (0.5 + value * 0.5)))

            pygame.draw.line(
                overlay_surface,
                (*Colors.ACO_COLOR, alpha),
                start, end,
                line_width
            )

        # Render pheromone trail lines
        for i, (start, end) in enumerate(vis_data.lines):
            if i >= len(vis_data.line_colors):
                break

            color = vis_data.line_colors[i]
            start_screen = self.transform_point(start, offset, zoom)
            end_screen = self.transform_point(end, offset, zoom)

            # Thin semi-transparent lines
            alpha = int(60 * self.LINE_OPACITY)
            pygame.draw.line(
                overlay_surface,
                (*color[:3], alpha) if len(color) >= 3 else (*color, alpha),
                start_screen,
                end_screen,
                max(1, int(2 * zoom))
            )

        surface.blit(overlay_surface, (0, 0))

    def _render_road_overlays(
        self,
        surface: pygame.Surface,
        vis_data: AlgorithmVisualization,
        offset: Tuple[float, float],
        zoom: float,
        opacity: float
    ) -> None:
        """Render colored overlays on roads (subtle, thin lines)."""
        overlay_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        for road_id, color in vis_data.road_colors.items():
            road = self.network.get_road(road_id)
            if not road:
                continue

            start = self.transform_point(road.start_pos, offset, zoom)
            end = self.transform_point(road.end_pos, offset, zoom)

            value = vis_data.road_values.get(road_id, 0.5)

            # Skip low values for cleaner appearance
            if value < 0.15:
                continue

            # Subtle semi-transparent line
            alpha = int(100 * opacity * value)

            pygame.draw.line(
                overlay_surface,
                (*color, alpha),
                start, end,
                max(2, int(4 * zoom))
            )

        surface.blit(overlay_surface, (0, 0))

    def _render_lines(
        self,
        surface: pygame.Surface,
        vis_data: AlgorithmVisualization,
        offset: Tuple[float, float],
        zoom: float,
        opacity: float
    ) -> None:
        """Render line overlays (trails, connections) - thin and subtle."""
        overlay_surface = pygame.Surface(surface.get_size(), pygame.SRCALPHA)

        for i, (start, end) in enumerate(vis_data.lines):
            if i >= len(vis_data.line_colors):
                break

            color = vis_data.line_colors[i]
            start_screen = self.transform_point(start, offset, zoom)
            end_screen = self.transform_point(end, offset, zoom)

            # Add alpha if not present
            if len(color) == 3:
                alpha = int(120 * opacity)
                draw_color = (*color, alpha)
            else:
                draw_color = (*color[:3], int(color[3] * opacity))

            pygame.draw.line(
                overlay_surface,
                draw_color,
                start_screen,
                end_screen,
                max(1, int(2 * zoom))
            )

        surface.blit(overlay_surface, (0, 0))

    def _render_points(
        self,
        surface: pygame.Surface,
        vis_data: AlgorithmVisualization,
        offset: Tuple[float, float],
        zoom: float
    ) -> None:
        """Render point overlays (particles, agents) - small and subtle."""
        for i, pos in enumerate(vis_data.points):
            if i >= len(vis_data.point_colors):
                break

            color = vis_data.point_colors[i]
            size = vis_data.point_sizes[i] if i < len(vis_data.point_sizes) else 3.0
            screen_pos = self.transform_point(pos, offset, zoom)

            # Smaller points for less clutter
            pygame.draw.circle(
                surface,
                color,
                screen_pos,
                max(2, int(size * zoom * 0.7))
            )

    def toggle_algorithm_overlay(self, algo_name: str) -> bool:
        """Toggle algorithm overlay visibility."""
        current = self._algorithm_visibility.get(algo_name, False)
        self._algorithm_visibility[algo_name] = not current
        return self._algorithm_visibility[algo_name]

    def set_algorithm_opacity(self, algo_name: str, opacity: float) -> None:
        """Set algorithm overlay opacity."""
        self._algorithm_opacity[algo_name] = max(0.0, min(1.0, opacity))

    def set_algorithm_visibility(self, algo_name: str, visible: bool) -> None:
        """Set algorithm overlay visibility directly."""
        self._algorithm_visibility[algo_name] = visible
