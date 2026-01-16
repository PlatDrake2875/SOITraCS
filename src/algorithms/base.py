"""Base algorithm interface that all algorithms implement."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.entities.network import RoadNetwork
    from src.core.state import SimulationState


@dataclass
class AlgorithmVisualization:
    """
    Container for algorithm-specific visualization data.

    Each algorithm provides data for its visual overlay.
    """

    # Layer enable flags
    show_overlay: bool = True
    opacity: float = 0.7

    # Road-based overlays (road_id -> value)
    road_values: Dict[int, float] = field(default_factory=dict)
    road_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    # Intersection-based overlays
    intersection_values: Dict[int, float] = field(default_factory=dict)
    intersection_colors: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)

    # Point-based overlays (particles, agents, etc.)
    points: List[Tuple[float, float]] = field(default_factory=list)
    point_colors: List[Tuple[int, int, int]] = field(default_factory=list)
    point_sizes: List[float] = field(default_factory=list)

    # Line-based overlays (trails, connections)
    lines: List[Tuple[Tuple[float, float], Tuple[float, float]]] = field(default_factory=list)
    line_colors: List[Tuple[int, int, int, int]] = field(default_factory=list)

    # Heatmap data (for SOM, congestion)
    heatmap: Optional[Any] = None  # numpy array
    heatmap_bounds: Optional[Tuple[float, float, float, float]] = None

    # Text annotations
    annotations: List[Tuple[float, float, str]] = field(default_factory=list)


class BaseAlgorithm(ABC):
    """
    Abstract base class for all traffic control algorithms.

    Each algorithm (CA, SOTL, ACO, PSO, SOM, MARL) implements this interface.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize algorithm with configuration.

        Args:
            config: Algorithm-specific configuration from YAML
        """
        self.config = config
        self.enabled = config.get("enabled", True)
        self._initialized = False
        self._network: Optional["RoadNetwork"] = None
        self._visualization = AlgorithmVisualization()

    @property
    @abstractmethod
    def name(self) -> str:
        """Algorithm name for identification."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable algorithm name for UI."""
        pass

    @property
    @abstractmethod
    def color(self) -> Tuple[int, int, int]:
        """Algorithm's signature color for visualization."""
        pass

    @abstractmethod
    def initialize(self, network: "RoadNetwork", state: "SimulationState") -> None:
        """
        Initialize algorithm with network topology.

        Called once when simulation starts.

        Args:
            network: The road network
            state: The simulation state
        """
        pass

    @abstractmethod
    def update(self, state: "SimulationState", dt: float) -> Dict[str, Any]:
        """
        Process one simulation step.

        Args:
            state: Current simulation state
            dt: Time delta in simulation seconds

        Returns:
            Dictionary of algorithm decisions/outputs
        """
        pass

    @abstractmethod
    def get_visualization_data(self) -> AlgorithmVisualization:
        """
        Get data for visualization overlay.

        Returns:
            AlgorithmVisualization with current visual state
        """
        pass

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """
        Get algorithm-specific metrics for dashboard.

        Returns:
            Dictionary of metric name -> value
        """
        pass

    def toggle(self, enabled: bool) -> None:
        """
        Enable or disable the algorithm.

        Args:
            enabled: New enabled state
        """
        self.enabled = enabled

    def reset(self) -> None:
        """Reset algorithm to initial state."""
        self._initialized = False
        self._visualization = AlgorithmVisualization()

    def on_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Handle events from the event bus.

        Override in subclass to respond to specific events.

        Args:
            event_type: Type of event
            data: Event data
        """
        pass

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key
            default: Default value if not found

        Returns:
            Configuration value
        """
        return self.config.get(key, default)

    def set_config_value(self, key: str, value: Any) -> None:
        """
        Update a configuration value at runtime.

        Args:
            key: Configuration key
            value: New value
        """
        self.config[key] = value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(enabled={self.enabled})"
