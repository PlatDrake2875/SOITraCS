# SOITCS - Self-Organizing Intelligent Traffic Control System

A Pygame-based visualization demonstrating self-organizing traffic control algorithms working together. Built for a Master's thesis project exploring emergent behavior in traffic systems.

## Features

### Traffic Simulation
- **Cellular Automata (CA)** - NaSch model for realistic traffic flow with random slowdowns
- **Self-Organizing Traffic Lights (SOTL)** - Adaptive signal control based on queue length and platoon detection
- **Ant Colony Optimization (ACO)** - Pheromone-based dynamic vehicle routing
- **Particle Swarm Optimization (PSO)** - Signal timing optimization
- **Self-Organizing Maps (SOM)** - Traffic pattern recognition
- **Multi-Agent Reinforcement Learning (MARL)** - Learning-based signal control

### Visualization
- Clean road rendering with dashed lane dividers and rounded ends
- Compact signal strips on intersection edges (grouped by phase)
- Speed-based vehicle coloring (red = stopped, white = moving)
- Subtle algorithm overlays with adjustable opacity
- Modern dark-themed dashboard with real-time metrics

### Configuration
- YAML-based network topology (customizable intersections and roads)
- YAML-tunable algorithm parameters
- Scenario presets (rush hour, incidents)
- Toggle algorithms on/off at runtime

## Project Structure

```
SOITCS/
├── main.py                      # Application entry point
├── pyproject.toml               # Project dependencies (uv/pip)
├── config/
│   ├── settings.py              # Global constants
│   ├── colors.py                # Color palette
│   ├── algorithms.yaml          # Algorithm parameters
│   └── scenarios/
│       └── grid_4x4.yaml        # 4x4 grid network definition
├── src/
│   ├── core/
│   │   ├── simulation.py        # Main simulation loop
│   │   ├── event_bus.py         # Pub/sub for algorithm communication
│   │   ├── state.py             # SimulationState container
│   │   └── clock.py             # Timing management
│   ├── entities/
│   │   ├── vehicle.py           # Vehicle entity
│   │   ├── intersection.py      # Intersection with signals
│   │   ├── traffic_light.py     # Traffic light phases
│   │   ├── road.py              # Road segments
│   │   └── network.py           # Road network graph
│   ├── algorithms/
│   │   ├── base.py              # BaseAlgorithm interface
│   │   ├── registry.py          # Algorithm registration
│   │   ├── cellular_automata/   # CA traffic flow
│   │   ├── sotl/                # Self-organizing lights
│   │   ├── aco/                 # Ant colony routing
│   │   ├── pso/                 # Signal optimization
│   │   ├── som/                 # Pattern recognition
│   │   └── marl/                # Reinforcement learning
│   ├── visualization/
│   │   ├── renderer.py          # Main render coordinator
│   │   ├── layers/
│   │   │   ├── base.py          # BaseLayer interface
│   │   │   ├── network_layer.py # Roads and intersections
│   │   │   ├── signal_layer.py  # Traffic signals
│   │   │   ├── vehicle_layer.py # Vehicles
│   │   │   └── overlay_layer.py # Algorithm visualizations
│   │   └── dashboard/
│   │       └── panel.py         # Metrics and controls
│   └── utils/
│       ├── threading_utils.py   # Background computation
│       └── recorder.py          # Export for thesis figures
├── plans/
│   ├── IMPLEMENTATION_PLAN.md   # Development roadmap
│   └── SOITCS_Critical_Analysis.md
└── tests/
```

## Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)
```bash
# Clone the repository
git clone https://github.com/PlatDrake2875/SOITCS.git
cd SOITCS

# Install dependencies and run
uv run python main.py
```

### Using pip
```bash
# Clone the repository
git clone https://github.com/PlatDrake2875/SOITCS.git
cd SOITCS

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install pygame numpy pyyaml matplotlib

# Run
python main.py
```

## Usage

### Running the Simulation
```bash
uv run python main.py
```

### Controls
| Key | Action |
|-----|--------|
| **Space** | Pause/Resume simulation |
| **1-5** | Set simulation speed (0.5x, 1x, 2x, 5x) |
| **Click** | Select intersection/vehicle for details |

### Dashboard
The right panel displays:
- **Metrics**: Vehicle count, average speed, delay, queue length
- **Algorithms**: Toggle buttons for each algorithm (CA, SOTL, ACO, PSO, SOM, MARL)
- **Speed**: Simulation speed controls and pause button

## Configuration

### Algorithm Parameters (`config/algorithms.yaml`)
```yaml
cellular_automata:
  enabled: true
  max_velocity: 5
  slow_probability: 0.3

sotl:
  enabled: false
  min_green_time: 10
  theta: 5  # Queue threshold

aco:
  enabled: false
  alpha: 1.0
  rho: 0.1  # Evaporation rate
```

### Network Topology (`config/scenarios/grid_4x4.yaml`)
```yaml
intersections:
  - id: 0
    position: [100, 100]
    signal_phases:
      - duration: 30
        green_directions: [north, south]

roads:
  - id: 0
    from: 0
    to: 1
    lanes: 1
    speed_limit: 5
```

## Algorithm Overview

| Algorithm | Purpose | Visualization |
|-----------|---------|---------------|
| **CA** | Traffic flow dynamics | Vehicle movement |
| **SOTL** | Adaptive signal control | Signal strips on intersections |
| **ACO** | Dynamic routing | Green pheromone trails |
| **PSO** | Signal optimization | Blue particles (when enabled) |
| **SOM** | Pattern recognition | Purple cluster overlay |
| **MARL** | Learning agents | Yellow decision indicators |

## Performance

- Target: **30 FPS** minimum
- Recommended: **100-500 vehicles** for educational visualization
- Heavy algorithms (PSO, MARL) run in background threads

## License

MIT License - See LICENSE file for details.

## Acknowledgments

Built as part of a Master's thesis on self-organizing traffic control systems.
