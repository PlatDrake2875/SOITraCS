"""Global settings and constants for SOITCS simulation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DisplaySettings:
    """Display and window settings."""
    window_width: int = 1600
    window_height: int = 900
    simulation_width: int = 1200
    simulation_height: int = 900
    dashboard_width: int = 400
    fps_target: int = 30
    title: str = "SOITCS - Self-Organizing Traffic Control"


@dataclass
class SimulationSettings:
    """Core simulation parameters."""
    cell_size: int = 10  # Pixels per cell
    vehicle_length: int = 2  # Cells
    min_gap: int = 1  # Minimum gap between vehicles (cells)
    spawn_rate: float = 0.1  # Default spawn rate (vehicles/tick)
    max_vehicles: int = 500  # Performance cap
    reroute_interval: int = 20  # Ticks between vehicle reroute attempts


@dataclass
class AlgorithmSettings:
    """Algorithm-specific settings loaded from YAML."""
    cellular_automata: dict = field(default_factory=lambda: {
        "enabled": True,
        "max_velocity": 5,
        "slow_probability": 0.3,
        "lane_change_threshold": 2,
    })
    sotl: dict = field(default_factory=lambda: {
        "enabled": True,
        "min_green_time": 10,
        "max_green_time": 60,
        "theta": 5,
        "mu": 3,
        "omega": 15,
    })
    aco: dict = field(default_factory=lambda: {
        "enabled": True,
        "alpha": 1.0,
        "beta": 2.0,
        "rho": 0.1,
        "Q": 100,
        "reroute_frequency": 10,
    })
    pso: dict = field(default_factory=lambda: {
        "enabled": False,
        "swarm_size": 30,
        "c1": 2.0,
        "c2": 2.0,
        "w": 0.7,
        "optimization_interval": 50,
    })
    som: dict = field(default_factory=lambda: {
        "enabled": False,
        "grid_size": [10, 10],
        "learning_rate": 0.5,
        "sigma": 3.0,
        "pretrained_path": "data/pretrained_models/som_traffic_patterns.pkl",
    })
    marl: dict = field(default_factory=lambda: {
        "enabled": False,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon": 0.1,
        "pretrained_path": "data/pretrained_models/marl_signal_control.pt",
        "inference_only": True,
    })


@dataclass
class Settings:
    """Main settings container."""
    display: DisplaySettings = field(default_factory=DisplaySettings)
    simulation: SimulationSettings = field(default_factory=SimulationSettings)
    algorithms: AlgorithmSettings = field(default_factory=AlgorithmSettings)

    @classmethod
    def load(cls, config_path: Path | None = None) -> "Settings":
        """Load settings from YAML file, falling back to defaults."""
        settings = cls()

        if config_path and config_path.exists():
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            # Update algorithm settings from YAML
            if "algorithms" in data:
                for algo_name, algo_config in data["algorithms"].items():
                    if hasattr(settings.algorithms, algo_name):
                        current = getattr(settings.algorithms, algo_name)
                        current.update(algo_config)

            # Update display settings
            if "display" in data:
                for key, value in data["display"].items():
                    if hasattr(settings.display, key):
                        setattr(settings.display, key, value)

            # Update simulation settings
            if "simulation" in data:
                for key, value in data["simulation"].items():
                    if hasattr(settings.simulation, key):
                        setattr(settings.simulation, key, value)

        return settings

    def save(self, config_path: Path) -> None:
        """Save current settings to YAML file."""
        data = {
            "display": {
                "window_width": self.display.window_width,
                "window_height": self.display.window_height,
                "fps_target": self.display.fps_target,
            },
            "simulation": {
                "cell_size": self.simulation.cell_size,
                "spawn_rate": self.simulation.spawn_rate,
                "max_vehicles": self.simulation.max_vehicles,
            },
            "algorithms": {
                "cellular_automata": self.algorithms.cellular_automata,
                "sotl": self.algorithms.sotl,
                "aco": self.algorithms.aco,
                "pso": self.algorithms.pso,
                "som": self.algorithms.som,
                "marl": self.algorithms.marl,
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)


# Global settings instance (lazy loaded)
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        config_path = Path(__file__).parent / "algorithms.yaml"
        _settings = Settings.load(config_path)
    return _settings
