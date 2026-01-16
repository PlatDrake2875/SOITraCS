#!/usr/bin/env python3
"""
SOITCS - Self-Organizing Intelligent Traffic Control Systems

A Pygame-based visualization demonstrating self-organizing traffic control
algorithms working together for a Master's thesis project.

Run with: uv run python main.py
"""

import sys
import argparse
from pathlib import Path
import time

import pygame

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import Settings, get_settings
from config.colors import Colors
from src.core.simulation import Simulation
from src.core.state import SimulationState
from src.core.clock import SimulationClock
from src.core.event_bus import get_event_bus, reset_event_bus
from src.entities.network import RoadNetwork
from src.visualization.renderer import Renderer


class Application:
    """Main application class."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.settings = get_settings()
        self.running = False

        # Override settings from args
        if args.network:
            self.network_path = Path(args.network)
        else:
            self.network_path = None

        # Pygame setup
        pygame.init()
        pygame.display.set_caption(self.settings.display.title)

        self.screen = pygame.display.set_mode((
            self.settings.display.window_width,
            self.settings.display.window_height
        ))

        self.clock = pygame.time.Clock()

        # Core components
        self.simulation: Simulation | None = None
        self.renderer: Renderer | None = None

        # Timing
        self.last_time = time.time()
        self.fps_display_timer = 0

    def initialize(self) -> None:
        """Initialize simulation and renderer."""
        # Reset event bus
        reset_event_bus()

        # Create simulation
        self.simulation = Simulation(settings=self.settings)

        # Initialize with network
        if self.network_path and self.network_path.exists():
            self.simulation.initialize(self.network_path)
        else:
            self.simulation.initialize()  # Use default grid

        # Create renderer
        self.renderer = Renderer(self.screen, self.settings)
        self.renderer.initialize(
            self.simulation.state.network,
            self.simulation.state
        )

        # Set up dashboard callbacks
        if self.renderer.dashboard:
            self.renderer.dashboard.set_callbacks(
                on_algorithm_toggle=self._on_algorithm_toggle,
                on_speed_change=self._on_speed_change,
                on_pause_toggle=self._on_pause_toggle
            )

    def run(self) -> None:
        """Main application loop."""
        self.initialize()
        self.running = True

        while self.running:
            # Calculate delta time
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time

            # Handle events
            self._handle_events()

            # Update simulation
            if self.simulation:
                self.simulation.update(dt)

            # Render
            if self.renderer and self.simulation:
                self.renderer.render(self.simulation.state, dt)

            # Draw FPS counter
            self._draw_fps()

            # Update display
            pygame.display.flip()

            # Cap frame rate
            self.clock.tick(self.settings.display.fps_target)

        pygame.quit()

    def _handle_events(self) -> None:
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mouse_click(event)

            elif event.type == pygame.MOUSEWHEEL:
                self._handle_mouse_wheel(event)

    def _handle_keydown(self, event: pygame.event.Event) -> None:
        """Handle key press events."""
        if event.key == pygame.K_ESCAPE:
            self.running = False

        elif event.key == pygame.K_SPACE:
            if self.simulation:
                self.simulation.toggle_pause()

        elif event.key == pygame.K_r:
            if self.simulation:
                self.simulation.reset()

        elif event.key == pygame.K_1:
            self._on_speed_change(0.5)

        elif event.key == pygame.K_2:
            self._on_speed_change(1.0)

        elif event.key == pygame.K_3:
            self._on_speed_change(2.0)

        elif event.key == pygame.K_4:
            self._on_speed_change(5.0)

        # Algorithm toggles
        elif event.key == pygame.K_c:
            self._on_algorithm_toggle("cellular_automata")

        elif event.key == pygame.K_s:
            self._on_algorithm_toggle("sotl")

        elif event.key == pygame.K_a:
            self._on_algorithm_toggle("aco")

        elif event.key == pygame.K_p:
            self._on_algorithm_toggle("pso")

        elif event.key == pygame.K_o:
            self._on_algorithm_toggle("som")

        elif event.key == pygame.K_m:
            self._on_algorithm_toggle("marl")

    def _handle_mouse_click(self, event: pygame.event.Event) -> None:
        """Handle mouse click events."""
        if event.button == 1:  # Left click
            if self.renderer:
                result = self.renderer.handle_click(event.pos)
                if result:
                    self._handle_click_result(result)

        elif event.button == 3:  # Right click
            # Inject incident at clicked road
            if self.renderer and self.simulation:
                world_pos = self.renderer.screen_to_world(event.pos)
                # Check if clicked on a road
                for road_id, road in self.simulation.state.network.roads.items():
                    # Simple distance check (could be improved)
                    if self._point_near_road(world_pos, road):
                        self.simulation.inject_incident(road_id, 100)
                        print(f"Incident injected on road {road_id}")
                        break

    def _point_near_road(self, point, road, threshold: float = 20) -> bool:
        """Check if point is near a road."""
        import math
        px, py = point
        x1, y1 = road.start_pos
        x2, y2 = road.end_pos

        line_len_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        if line_len_sq == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2) <= threshold

        t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_len_sq))
        closest_x = x1 + t * (x2 - x1)
        closest_y = y1 + t * (y2 - y1)
        dist = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)

        return dist <= threshold

    def _handle_mouse_wheel(self, event: pygame.event.Event) -> None:
        """Handle mouse wheel events (zoom)."""
        if self.renderer:
            if event.y > 0:
                self.renderer.set_zoom(self.renderer.zoom * 1.1)
            else:
                self.renderer.set_zoom(self.renderer.zoom / 1.1)

    def _handle_click_result(self, result: dict) -> None:
        """Handle click result from renderer."""
        if result.get("type") == "intersection":
            print(f"Clicked intersection {result['id']}")

        elif result.get("type") == "road":
            print(f"Clicked road {result['id']}")

        elif result.get("action") == "algorithm_toggle":
            print(f"Toggled algorithm {result['algorithm']}")

        elif result.get("action") == "speed_change":
            print(f"Speed changed to {result['speed']}x")

    def _on_algorithm_toggle(self, algo_name: str) -> None:
        """Handle algorithm toggle."""
        if self.simulation:
            new_state = self.simulation.toggle_algorithm(algo_name)
            if new_state is not None:
                status = "enabled" if new_state else "disabled"
                print(f"Algorithm {algo_name} {status}")

    def _on_speed_change(self, speed: float) -> None:
        """Handle speed change."""
        if self.simulation:
            self.simulation.set_speed(speed)
            print(f"Speed set to {speed}x")

    def _on_pause_toggle(self) -> None:
        """Handle pause toggle."""
        if self.simulation:
            paused = self.simulation.toggle_pause()
            status = "paused" if paused else "resumed"
            print(f"Simulation {status}")

    def _draw_fps(self) -> None:
        """Draw FPS counter."""
        fps = int(self.clock.get_fps())
        font = pygame.font.Font(None, 24)

        # Simulation info
        if self.simulation:
            tick = self.simulation.clock.tick
            sim_time = self.simulation.clock.format_time()
            speed = self.simulation.clock.speed
            paused = " [PAUSED]" if self.simulation.clock.is_paused else ""

            info_text = f"FPS: {fps} | Tick: {tick} | Time: {sim_time} | Speed: {speed}x{paused}"
        else:
            info_text = f"FPS: {fps}"

        text_surface = font.render(info_text, True, Colors.TEXT_ACCENT)
        self.screen.blit(text_surface, (10, 10))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SOITCS - Self-Organizing Intelligent Traffic Control Systems"
    )

    parser.add_argument(
        "-n", "--network",
        type=str,
        help="Path to network YAML configuration file"
    )

    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to algorithm configuration YAML file"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Window width"
    )

    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Window height"
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    try:
        app = Application(args)
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
