"""Color scheme for SOITCS visualization."""

from dataclasses import dataclass
from typing import Tuple

# Type alias for RGB colors
RGB = Tuple[int, int, int]
RGBA = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Colors:
    """Color palette for the simulation - polished visual style."""

    # Background and infrastructure - softer, less harsh
    BACKGROUND: RGB = (25, 28, 32)
    ROAD: RGB = (55, 58, 65)
    ROAD_SURFACE: RGB = (48, 51, 58)  # Slightly darker road fill
    ROAD_MARKING: RGB = (180, 180, 185)
    LANE_DIVIDER: RGB = (90, 93, 100)  # Subtle lane dividers
    INTERSECTION: RGB = (45, 48, 55)
    INTERSECTION_BORDER: RGB = (65, 68, 75)

    # Traffic signals - more vibrant
    SIGNAL_RED: RGB = (235, 65, 65)
    SIGNAL_YELLOW: RGB = (245, 195, 60)
    SIGNAL_GREEN: RGB = (65, 215, 95)
    SIGNAL_OFF: RGB = (60, 60, 65)  # Inactive signal

    # Vehicles - brighter for visibility
    VEHICLE_DEFAULT: RGB = (220, 225, 235)
    VEHICLE_SELECTED: RGB = (255, 220, 80)
    VEHICLE_STOPPED: RGB = (235, 90, 90)  # Red when stopped
    VEHICLE_MOVING: RGB = (240, 245, 255)  # White when moving fast
    VEHICLE_SHADOW: RGB = (20, 22, 25)  # Subtle shadow

    # Algorithm-specific colors - more pastel/subtle for overlays
    CA_COLOR: RGB = (200, 100, 100)       # Soft red - Traffic flow
    ACO_COLOR: RGB = (100, 200, 120)      # Soft green - Pheromone trails
    PSO_COLOR: RGB = (100, 140, 200)      # Soft blue - Signal optimization
    MARL_COLOR: RGB = (200, 180, 100)     # Soft gold - Learning agents
    SOM_COLOR: RGB = (160, 120, 200)      # Soft purple - Pattern clusters
    SOTL_COLOR: RGB = (200, 160, 100)     # Soft orange - Self-organizing lights

    # Dashboard - modern dark theme with better contrast
    PANEL_BG: RGB = (32, 35, 40)
    PANEL_HEADER: RGB = (38, 42, 48)
    PANEL_BORDER: RGB = (55, 58, 65)
    PANEL_DIVIDER: RGB = (50, 53, 60)
    TEXT_PRIMARY: RGB = (235, 238, 245)
    TEXT_SECONDARY: RGB = (140, 145, 155)
    TEXT_ACCENT: RGB = (110, 190, 235)
    TEXT_VALUE: RGB = (255, 255, 255)  # Bright white for values

    # Graph colors
    GRAPH_LINE_1: RGB = (100, 180, 220)
    GRAPH_LINE_2: RGB = (220, 100, 100)
    GRAPH_LINE_3: RGB = (100, 200, 100)
    GRAPH_GRID: RGB = (50, 53, 60)

    # Interactive elements - more refined
    BUTTON_NORMAL: RGB = (50, 55, 65)
    BUTTON_HOVER: RGB = (65, 70, 80)
    BUTTON_ACTIVE: RGB = (80, 130, 170)
    TOGGLE_ON: RGB = (65, 150, 85)
    TOGGLE_OFF: RGB = (55, 58, 65)

    # Overlays - lower contrast for subtlety
    CONGESTION_LOW: RGB = (80, 180, 100)
    CONGESTION_MEDIUM: RGB = (200, 170, 70)
    CONGESTION_HIGH: RGB = (200, 90, 70)

    # Pheromone trail (use with alpha)
    PHEROMONE_WEAK: RGBA = (100, 200, 120, 20)
    PHEROMONE_MEDIUM: RGBA = (100, 200, 120, 60)
    PHEROMONE_STRONG: RGBA = (100, 200, 120, 120)

    @classmethod
    def with_alpha(cls, color: RGB, alpha: int) -> RGBA:
        """Add alpha channel to RGB color."""
        return (color[0], color[1], color[2], alpha)

    @classmethod
    def lerp(cls, color1: RGB, color2: RGB, t: float) -> RGB:
        """Linear interpolation between two colors."""
        t = max(0.0, min(1.0, t))
        return (
            int(color1[0] + (color2[0] - color1[0]) * t),
            int(color1[1] + (color2[1] - color1[1]) * t),
            int(color1[2] + (color2[2] - color1[2]) * t),
        )

    @classmethod
    def congestion_color(cls, level: float) -> RGB:
        """Get color for congestion level (0.0 = free, 1.0 = jammed)."""
        if level < 0.5:
            return cls.lerp(cls.CONGESTION_LOW, cls.CONGESTION_MEDIUM, level * 2)
        else:
            return cls.lerp(cls.CONGESTION_MEDIUM, cls.CONGESTION_HIGH, (level - 0.5) * 2)

    @classmethod
    def speed_color(cls, speed_ratio: float) -> RGB:
        """Get color based on speed (0.0 = stopped/red, 1.0 = moving/white)."""
        return cls.lerp(cls.VEHICLE_STOPPED, cls.VEHICLE_MOVING, speed_ratio)
