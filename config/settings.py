"""Global settings and constants for SOITraCS simulation."""

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
    title: str = "SOITraCS - Self-Organizing Traffic Control"

@dataclass
class SimulationSettings:
    """Core simulation parameters."""
    cell_size: int = 10
    vehicle_length: int = 2
    min_gap: int = 1
    spawn_rate: float = 0.1
    max_vehicles: int = 500
    reroute_interval: int = 20

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
        "inference_only": False,
    })
    marl_enhanced: dict = field(default_factory=lambda: {
        "enabled": True,
        "learning_rate": 0.01,
        "gamma": 0.95,
        "epsilon": 1.0,
        "epsilon_decay": 0.9995,
        "epsilon_min": 0.01,
        "double_q": True,
        "experience_replay": True,
        "prioritized_replay": True,
        "buffer_size": 10000,
        "batch_size": 32,
        "target_update_interval": 100,
    })
    pressure: dict = field(default_factory=lambda: {
        "enabled": True,
        "overload_threshold": 20,
        "both_overload_threshold": 15,
        "stall_staleness": 40,
        "all_red_clear_ticks": 3,
        "hysteresis_delta": 3,
        "starvation_ticks": 120,
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

            algo_data = data.get("algorithms", {})

            for algo_name in ["cellular_automata", "sotl", "aco", "pso", "som", "marl", "marl_enhanced", "pressure"]:
                if algo_name in data and algo_name not in algo_data:
                    algo_data[algo_name] = data[algo_name]

            for algo_name, algo_config in algo_data.items():
                if hasattr(settings.algorithms, algo_name):
                    current = getattr(settings.algorithms, algo_name)
                    current.update(algo_config)

            if "display" in data:
                for key, value in data["display"].items():
                    if hasattr(settings.display, key):
                        setattr(settings.display, key, value)

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
                "marl_enhanced": self.algorithms.marl_enhanced,
                "pressure": self.algorithms.pressure,
            }
        }

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

_settings: Settings | None = None

def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        config_path = Path(__file__).parent / "algorithms.yaml"
        _settings = Settings.load(config_path)
    return _settings
