"""Dashboard panel for metrics and controls."""

from typing import Dict, List, Optional, Tuple, Any, Callable
import pygame
import numpy as np

from config.settings import Settings
from config.colors import Colors
from src.core.state import SimulationState
from .sparkline import SparklineGraph


class DashboardPanel:
    """
    Dashboard panel showing metrics, algorithm toggles, and controls.

    Modern dark theme with clear visual hierarchy.

    Layout:
    - Global metrics (top)
    - Algorithm toggles (middle)
    - Speed controls (bottom)
    """

    def __init__(
        self,
        rect: pygame.Rect,
        settings: Settings,
        state: SimulationState
    ) -> None:
        self.rect = rect
        self.settings = settings
        self.state = state

        # Fonts
        pygame.font.init()
        self.font_title = pygame.font.Font(None, 32)
        self.font_label = pygame.font.Font(None, 20)
        self.font_value = pygame.font.Font(None, 30)
        self.font_small = pygame.font.Font(None, 18)

        # UI state
        self._toggle_states: Dict[str, bool] = {}
        self._button_rects: Dict[str, pygame.Rect] = {}

        # Callbacks
        self._on_algorithm_toggle: Optional[Callable[[str], None]] = None
        self._on_speed_change: Optional[Callable[[float], None]] = None
        self._on_pause_toggle: Optional[Callable[[], None]] = None
        self._on_scenario_change: Optional[Callable[[str], None]] = None

        # Active scenario tracking
        self._active_scenario: Optional[str] = None

        # Sparkline graphs
        self._sparklines = {
            "total_vehicles": SparklineGraph(color=(100, 140, 200)),
            "average_speed": SparklineGraph(color=(100, 200, 120)),
            "average_delay": SparklineGraph(color=(200, 100, 100)),
            "average_queue_length": SparklineGraph(color=(200, 180, 100)),
        }
        self._graph_update_counter = 0
        self._cached_graph_data: Dict[str, np.ndarray] = {}

        # Spacing
        self.padding = 16
        self.section_gap = 16
        self.item_height = 32

    def set_callbacks(
        self,
        on_algorithm_toggle: Callable[[str], None] = None,
        on_speed_change: Callable[[float], None] = None,
        on_pause_toggle: Callable[[], None] = None,
        on_scenario_change: Callable[[str], None] = None
    ) -> None:
        """Set callback functions for UI interactions."""
        self._on_algorithm_toggle = on_algorithm_toggle
        self._on_speed_change = on_speed_change
        self._on_pause_toggle = on_pause_toggle
        self._on_scenario_change = on_scenario_change

    def render(self, surface: pygame.Surface, state: SimulationState) -> None:
        """Render the dashboard."""
        # Background
        pygame.draw.rect(surface, Colors.PANEL_BG, self.rect)

        # Border on left side
        border_rect = pygame.Rect(self.rect.left, self.rect.top, 2, self.rect.height)
        pygame.draw.rect(surface, Colors.PANEL_BORDER, border_rect)

        y = self.rect.top + self.padding

        # Title
        y = self._render_title(surface, y)
        y = self._render_divider(surface, y)

        # Global metrics
        y = self._render_metrics(surface, state, y)
        y = self._render_divider(surface, y)

        # Algorithm toggles
        y = self._render_algorithm_toggles(surface, state, y)
        y = self._render_divider(surface, y)

        # Scenario buttons
        y = self._render_scenario_buttons(surface, state, y)
        y = self._render_divider(surface, y)

        # Speed controls
        y = self._render_speed_controls(surface, state, y)

    def _render_divider(self, surface: pygame.Surface, y: int) -> int:
        """Render a subtle horizontal divider."""
        y += self.section_gap // 2
        divider_rect = pygame.Rect(
            self.rect.left + self.padding,
            y,
            self.rect.width - 2 * self.padding,
            1
        )
        pygame.draw.rect(surface, Colors.PANEL_DIVIDER, divider_rect)
        return y + self.section_gap // 2 + 4

    def _render_title(self, surface: pygame.Surface, y: int) -> int:
        """Render dashboard title."""
        title = self.font_title.render("SOITCS", True, Colors.TEXT_ACCENT)
        x = self.rect.left + self.padding
        surface.blit(title, (x, y))
        return y + title.get_height() + 4

    def _render_metrics(self, surface: pygame.Surface, state: SimulationState, y: int) -> int:
        """Render global metrics section with sparklines."""
        x = self.rect.left + self.padding
        metrics = state.current_metrics

        # Update graph data every 10 frames (not every frame)
        self._graph_update_counter += 1
        if self._graph_update_counter >= 10:
            self._graph_update_counter = 0
            for key in self._sparklines:
                self._cached_graph_data[key] = state.get_metrics_array(key, 60)

        # Section header
        header = self.font_label.render("METRICS", True, Colors.TEXT_SECONDARY)
        surface.blit(header, (x, y))
        y += header.get_height() + 10

        # Metric items with sparklines
        metric_items = [
            ("Vehicles", str(metrics.total_vehicles), Colors.TEXT_VALUE, "total_vehicles"),
            ("Avg Speed", f"{metrics.average_speed:.1f}", Colors.TEXT_VALUE, "average_speed"),
            ("Avg Delay", f"{metrics.average_delay:.1f}s", Colors.TEXT_VALUE, "average_delay"),
            ("Queue", f"{metrics.average_queue_length:.1f}", Colors.TEXT_VALUE, "average_queue_length"),
        ]

        for label, value, value_color, metric_key in metric_items:
            y = self._render_metric_row(surface, x, y, label, value, value_color, metric_key)

        return y

    def _render_metric_row(
        self,
        surface: pygame.Surface,
        x: int,
        y: int,
        label: str,
        value: str,
        value_color: Tuple[int, int, int],
        metric_key: str = ""
    ) -> int:
        """Render a single metric row with sparkline."""
        # Label
        label_surf = self.font_small.render(label, True, Colors.TEXT_SECONDARY)
        surface.blit(label_surf, (x, y + 4))

        # Value (fixed position to leave room for sparkline)
        value_surf = self.font_value.render(value, True, value_color)
        value_x = x + 70
        surface.blit(value_surf, (value_x, y))

        # Sparkline (right side)
        if metric_key and metric_key in self._sparklines:
            graph_x = self.rect.right - self.padding - 80
            self._sparklines[metric_key].render(
                surface, graph_x, y + 2,
                self._cached_graph_data.get(metric_key, np.array([]))
            )

        return y + self.item_height

    def _render_algorithm_toggles(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        y: int
    ) -> int:
        """Render algorithm toggle buttons."""
        x = self.rect.left + self.padding

        # Section header
        header = self.font_label.render("ALGORITHMS", True, Colors.TEXT_SECONDARY)
        surface.blit(header, (x, y))
        y += header.get_height() + 10

        # Algorithm toggles - with colored text instead of indicator boxes
        algorithms = [
            ("cellular_automata", "CA", Colors.CA_COLOR),
            ("sotl", "SOTL", Colors.SOTL_COLOR),
            ("aco", "ACO", Colors.ACO_COLOR),
            ("pso", "PSO", Colors.PSO_COLOR),
            ("som", "SOM", Colors.SOM_COLOR),
            ("marl", "MARL", Colors.MARL_COLOR),
        ]

        # Render in two columns
        col_width = (self.rect.width - 3 * self.padding) // 2
        start_y = y
        col = 0

        for i, (algo_name, display_name, algo_color) in enumerate(algorithms):
            if i > 0 and i % 3 == 0:
                col = 1
                y = start_y

            algo = state.get_algorithm(algo_name)
            enabled = algo.enabled if algo else False

            btn_x = x + col * (col_width + self.padding)
            btn_rect = pygame.Rect(btn_x, y, col_width, self.item_height - 6)

            self._button_rects[algo_name] = btn_rect
            self._toggle_states[algo_name] = enabled

            # Draw button with colored border when enabled
            if enabled:
                # Filled background
                pygame.draw.rect(surface, Colors.TOGGLE_ON, btn_rect, border_radius=4)
                # Colored top accent line
                accent_rect = pygame.Rect(btn_x, y, col_width, 3)
                pygame.draw.rect(surface, algo_color, accent_rect, border_radius=2)
                text_color = Colors.TEXT_PRIMARY
            else:
                pygame.draw.rect(surface, Colors.TOGGLE_OFF, btn_rect, border_radius=4)
                text_color = Colors.TEXT_SECONDARY

            # Draw label (colored when enabled)
            label_color = algo_color if enabled else text_color
            label = self.font_small.render(display_name, True, label_color)
            label_x = btn_x + (col_width - label.get_width()) // 2
            label_y = y + (self.item_height - 6 - label.get_height()) // 2 - 4
            surface.blit(label, (label_x, label_y))

            # Show key metric when enabled
            if enabled and algo:
                metric_text = self._get_algorithm_metric_text(algo_name, algo)
                if metric_text:
                    metric_surf = self.font_small.render(metric_text, True, Colors.TEXT_SECONDARY)
                    metric_x = btn_x + (col_width - metric_surf.get_width()) // 2
                    surface.blit(metric_surf, (metric_x, y + self.item_height - 18))

            y += self.item_height

        return max(y, start_y + 3 * self.item_height)

    def _get_algorithm_metric_text(self, name: str, algo: Any) -> str:
        """Get key metric text for an algorithm."""
        try:
            metrics = algo.get_metrics()
            if name == "cellular_automata":
                return f"Flow: {metrics.get('flow_rate', 0):.0f}/min"
            elif name == "sotl":
                return f"Chg: {metrics.get('total_phase_changes', 0):.0f}"
            elif name == "aco":
                return f"Rte: {metrics.get('total_reroutes', 0):.0f}"
            elif name == "pso":
                return f"Fit: {metrics.get('best_fitness', 0):.2f}"
            elif name == "som":
                pattern = getattr(algo, 'current_pattern', 'unknown')
                return pattern.upper()[:8]
            elif name == "marl":
                return f"Rwd: {metrics.get('avg_reward', 0):.1f}"
        except Exception:
            pass
        return ""

    def _render_scenario_buttons(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        y: int
    ) -> int:
        """Render scenario selection buttons."""
        x = self.rect.left + self.padding

        # Section header
        header = self.font_label.render("SCENARIOS", True, Colors.TEXT_SECONDARY)
        surface.blit(header, (x, y))
        y += header.get_height() + 10

        # Scenario buttons
        scenarios = [("normal", "Normal"), ("rush_hour", "Rush"), ("incident", "Incident")]
        btn_width = (self.rect.width - 2 * self.padding - 2 * 6) // 3

        for i, (scenario_id, label) in enumerate(scenarios):
            btn_x = x + i * (btn_width + 6)
            btn_rect = pygame.Rect(btn_x, y, btn_width, self.item_height - 4)

            self._button_rects[f"scenario_{scenario_id}"] = btn_rect

            # Highlight active scenario
            is_active = (self._active_scenario == scenario_id) or \
                        (self._active_scenario is None and scenario_id == "normal")
            bg_color = Colors.TOGGLE_ON if is_active else Colors.BUTTON_NORMAL
            pygame.draw.rect(surface, bg_color, btn_rect, border_radius=4)

            label_surf = self.font_small.render(label, True, Colors.TEXT_PRIMARY)
            label_x = btn_x + (btn_width - label_surf.get_width()) // 2
            label_y = y + (self.item_height - 4 - label_surf.get_height()) // 2
            surface.blit(label_surf, (label_x, label_y))

        return y + self.item_height

    def _render_speed_controls(
        self,
        surface: pygame.Surface,
        state: SimulationState,
        y: int
    ) -> int:
        """Render speed control buttons."""
        x = self.rect.left + self.padding

        # Section header
        header = self.font_label.render("SPEED", True, Colors.TEXT_SECONDARY)
        surface.blit(header, (x, y))
        y += header.get_height() + 10

        # Speed buttons
        speeds = [("0.5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("5x", 5.0)]
        btn_width = (self.rect.width - 2 * self.padding - 3 * 6) // 4

        for i, (label, speed) in enumerate(speeds):
            btn_x = x + i * (btn_width + 6)
            btn_rect = pygame.Rect(btn_x, y, btn_width, self.item_height - 4)

            self._button_rects[f"speed_{speed}"] = btn_rect

            bg_color = Colors.BUTTON_NORMAL
            pygame.draw.rect(surface, bg_color, btn_rect, border_radius=4)

            label_surf = self.font_small.render(label, True, Colors.TEXT_PRIMARY)
            label_x = btn_x + (btn_width - label_surf.get_width()) // 2
            label_y = y + (self.item_height - 4 - label_surf.get_height()) // 2
            surface.blit(label_surf, (label_x, label_y))

        y += self.item_height + 4

        # Pause button
        pause_rect = pygame.Rect(x, y, self.rect.width - 2 * self.padding, self.item_height - 4)
        self._button_rects["pause"] = pause_rect

        pygame.draw.rect(surface, Colors.BUTTON_NORMAL, pause_rect, border_radius=4)
        pause_label = self.font_small.render("PAUSE (Space)", True, Colors.TEXT_PRIMARY)
        label_x = x + (pause_rect.width - pause_label.get_width()) // 2
        label_y = y + (self.item_height - 4 - pause_label.get_height()) // 2
        surface.blit(pause_label, (label_x, label_y))

        return y + self.item_height

    def handle_click(self, pos: Tuple[int, int]) -> Optional[Dict[str, Any]]:
        """Handle click on dashboard. Returns action info or None."""
        for algo_name, rect in self._button_rects.items():
            if rect.collidepoint(pos):
                if algo_name.startswith("speed_"):
                    speed = float(algo_name.split("_")[1])
                    if self._on_speed_change:
                        self._on_speed_change(speed)
                    return {"action": "speed_change", "speed": speed}

                elif algo_name == "pause":
                    if self._on_pause_toggle:
                        self._on_pause_toggle()
                    return {"action": "pause_toggle"}

                elif algo_name.startswith("scenario_"):
                    scenario = algo_name.split("_", 1)[1]
                    self._active_scenario = None if scenario == "normal" else scenario
                    if self._on_scenario_change:
                        self._on_scenario_change(scenario)
                    return {"action": "scenario_change", "scenario": scenario}

                else:
                    # Algorithm toggle
                    if self._on_algorithm_toggle:
                        self._on_algorithm_toggle(algo_name)
                    return {"action": "algorithm_toggle", "algorithm": algo_name}

        return None
