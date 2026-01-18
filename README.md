# SOITraCS - Self-Organizing Intelligent Traffic Control System

A Pygame-based simulation framework for studying self-organizing traffic control algorithms. The system integrates six algorithms within an event-driven architecture, enabling both interactive visualization and automated experiments.

## Features

### Traffic Simulation
- **Cellular Automata (CA)** - Nagel-Schreckenberg model for microscopic traffic flow
- **Self-Organizing Traffic Lights (SOTL)** - Adaptive signal control based on queue thresholds
- **Ant Colony Optimization (ACO)** - Pheromone-based dynamic vehicle routing
- **Particle Swarm Optimization (PSO)** - Signal timing parameter optimization
- **Self-Organizing Maps (SOM)** - Traffic pattern recognition and classification
- **Multi-Agent Reinforcement Learning (MARL)** - Q-learning agents for adaptive control

### Visualization
- Real-time traffic flow rendering with speed-based vehicle coloring
- Signal state visualization on intersection edges
- Algorithm overlays (pheromone trails, SOM clusters, MARL decisions)
- Dashboard with live metrics and algorithm toggles

### Experimentation
- Automated ablation study framework
- Configurable experiment protocols
- Statistical analysis with significance testing
- Result export (CSV, LaTeX tables, figures)

## Project Structure

```
SOITraCS/
├── main.py                      # Application entry point
├── pyproject.toml               # Project dependencies
├── config/
│   ├── settings.py              # Global settings
│   ├── algorithms.yaml          # Algorithm parameters
│   ├── scenarios/
│   │   └── grid_4x4.yaml        # Network topology
│   └── experiments/
│       ├── ablation_study.yaml  # Ablation experiment config
│       └── standard_comparison.yaml
├── src/
│   ├── core/
│   │   ├── simulation.py        # Main simulation loop
│   │   ├── event_bus.py         # Pub/sub communication
│   │   └── state.py             # Simulation state container
│   ├── entities/
│   │   ├── vehicle.py           # Vehicle entity
│   │   ├── intersection.py      # Intersection with signals
│   │   ├── road.py              # Road segments
│   │   └── network.py           # Road network graph
│   ├── algorithms/
│   │   ├── base.py              # BaseAlgorithm interface
│   │   ├── cellular_automata/   # CA traffic flow
│   │   ├── sotl/                # Self-organizing lights
│   │   ├── aco/                 # Ant colony routing
│   │   ├── pso/                 # Signal optimization
│   │   ├── som/                 # Pattern recognition
│   │   └── marl/                # Reinforcement learning
│   ├── visualization/
│   │   ├── renderer.py          # Render coordinator
│   │   ├── layers/              # Rendering layers
│   │   └── dashboard/           # Metrics panel
│   └── evaluation/
│       ├── experiment.py        # Experiment runner
│       ├── analysis.py          # Statistical analysis
│       └── self_organization.py # Emergence metrics
├── scripts/
│   ├── run_ablation.py          # Run ablation study
│   ├── analyze_ablation.py      # Analyze results
│   ├── run_experiments.py       # General experiments
│   └── train_marl.py            # Train MARL agents
├── docs/
│   ├── final_report.tex         # IEEE-format paper
│   ├── final_report.pdf
│   ├── final_presentation.tex   # Beamer slides
│   ├── final_presentation.pdf
│   └── references.bib           # Bibliography
├── results/                     # Experiment outputs
│   └── ablation_final/          # Ablation study results
└── tests/
```

## Installation

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Using uv (recommended)
```bash
git clone https://github.com/PlatDrake2875/SOITraCS.git
cd SOITraCS

# Install dependencies and run visualization
uv run python main.py
```

### Using pip
```bash
git clone https://github.com/PlatDrake2875/SOITraCS.git
cd SOITraCS

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install pygame numpy pyyaml matplotlib scipy

python main.py
```

## Usage

### Interactive Visualization
```bash
uv run python main.py
```

**Controls:**
| Key | Action |
|-----|--------|
| Space | Pause/Resume |
| 1-5 | Set speed (0.5x to 5x) |
| Click | Inject incident at intersection |

**Dashboard:** Toggle algorithms on/off, view real-time metrics (delay, throughput, speed).

### Running Experiments

**Ablation Study:**
```bash
# Run ablation experiments (340 runs)
uv run python scripts/run_ablation.py --output results/ablation

# Analyze results
uv run python scripts/analyze_ablation.py results/ablation
```

**Train MARL Agents:**
```bash
uv run python scripts/train_marl.py --episodes 1000
```

## Configuration

### Algorithm Parameters (`config/algorithms.yaml`)
```yaml
cellular_automata:
  enabled: true
  max_velocity: 5          # cells/tick
  slow_probability: 0.3    # NaSch random slowdown

sotl:
  enabled: true
  theta: 5                 # Queue threshold
  min_green_time: 10
  max_green_time: 60

aco:
  enabled: true
  alpha: 2.0               # Pheromone weight
  beta: 1.5                # Distance weight
  rho: 0.02                # Evaporation rate
```

### Experiment Configuration (`config/experiments/ablation_study.yaml`)
```yaml
experiment:
  duration: 5400           # ticks per run
  warmup: 400              # excluded from metrics
  seeds: [0, 1, ..., 19]   # 20 repetitions

configurations:
  - name: baseline
    algorithms: [ca]
  - name: ca_sotl
    algorithms: [ca, sotl]
  # ... 17 configurations total
```

## Results Summary

From ablation study (340 runs):

| Configuration | Delay Reduction | Significance |
|---------------|-----------------|--------------|
| + SOTL | -29.4% | p < 0.001 |
| + ACO | -29.9% | p < 0.001 |
| Full Stack | -44.5% | p < 0.001 |
| - SOM (from full) | +11.3% | p < 0.001 |

Key findings:
- SOTL and ACO provide complementary benefits
- SOM acts as integration enabler (0% alone, 11.3% contribution via coordination)
- Combined algorithms achieve synergistic improvement

## Documentation

- **Final Report:** `docs/final_report.pdf` - IEEE conference format paper
- **Presentation:** `docs/final_presentation.pdf` - Beamer slides

## Algorithm Overview

| Algorithm | Role | Key Parameters |
|-----------|------|----------------|
| CA | Vehicle dynamics | v_max=5, p_slow=0.3 |
| SOTL | Signal control | θ=5, t∈[10,60] |
| ACO | Dynamic routing | α=2.0, β=1.5, ρ=0.02 |
| PSO | Timing optimization | swarm=30, w=0.7 |
| SOM | Pattern recognition | 10×10 grid |
| MARL | Adaptive control | γ=0.95, ε=0.1 |

## License

MIT License - See LICENSE file for details.
