# SOITCS Implementation Plan

## Overview
Build a Pygame-based visualization demonstrating self-organizing traffic control algorithms working together for a Master's thesis project.

**Timeline**: 6 months
**Focus**: Algorithm demonstration + integration patterns
**Key Feature**: YAML-configurable algorithm parameters for easy tuning

## Architecture Summary

### Core Design Principles
1. **Plugin-based algorithms** - Each algorithm implements `BaseAlgorithm` interface
2. **Event-driven communication** - Algorithms communicate via `EventBus` (pub/sub)
3. **Layered visualization** - Composable render layers with toggle controls
4. **Separation of concerns** - Simulation logic separate from rendering
5. **YAML-driven configuration** - All algorithm parameters tunable without code changes

### Color Scheme
| Algorithm | Color | Purpose |
|-----------|-------|---------|
| CA | Soft Red (200,100,100) | Traffic flow visualization |
| ACO | Soft Green (100,200,120) | Pheromone trail routing |
| PSO | Soft Blue (100,140,200) | Signal optimization particles |
| MARL | Soft Gold (200,180,100) | Learning agent decisions |
| SOM | Soft Purple (160,120,200) | Pattern cluster overlay |
| SOTL | Soft Orange (200,160,100) | Self-organizing light rules |

---

## Project Structure

```
SOS_proj/
├── main.py                      # Entry point
├── requirements.txt
├── config/
│   ├── settings.py              # Global constants
│   ├── colors.py                # Color scheme (polished palette)
│   ├── algorithms.yaml          # Algorithm parameters
│   └── scenarios/               # Preset scenarios (yaml)
│       └── grid_4x4.yaml        # 4x4 grid with 250px spacing
├── src/
│   ├── core/
│   │   ├── simulation.py        # Main loop controller
│   │   ├── event_bus.py         # Pub/sub messaging
│   │   ├── state.py             # SimulationState container
│   │   └── clock.py             # Timing management
│   ├── entities/
│   │   ├── vehicle.py
│   │   ├── intersection.py
│   │   ├── traffic_light.py
│   │   └── network.py           # Road network graph
│   ├── algorithms/
│   │   ├── base.py              # Abstract BaseAlgorithm
│   │   ├── registry.py          # Algorithm registration
│   │   ├── cellular_automata/   # CA traffic flow
│   │   ├── sotl/                # Self-organizing lights
│   │   ├── aco/                 # Ant colony routing
│   │   ├── som/                 # Pattern recognition
│   │   ├── pso/                 # Signal optimization
│   │   └── marl/                # Learning agents (optional)
│   ├── visualization/
│   │   ├── renderer.py          # Main render coordinator
│   │   ├── layers/              # Composable visual layers
│   │   │   ├── base.py          # BaseLayer interface
│   │   │   ├── network_layer.py # Roads & intersections (polished)
│   │   │   ├── signal_layer.py  # Traffic signals (compact strips)
│   │   │   ├── vehicle_layer.py # Vehicles (speed-colored)
│   │   │   └── overlay_layer.py # Algorithm overlays (subtle)
│   │   ├── dashboard/           # Metrics panel, toggles
│   │   │   └── panel.py         # Modern dark theme
│   │   └── effects/             # Pheromone trails, heatmaps
│   └── utils/
│       ├── threading_utils.py   # Background computation
│       └── recorder.py          # Export for thesis
└── tests/
```

---

## Key Interfaces

### BaseAlgorithm (all algorithms implement this)
```python
class BaseAlgorithm(ABC):
    def initialize(self, network, state) -> None
    def update(self, state, dt) -> Dict[str, Any]  # Returns decisions
    def get_visualization_data(self) -> AlgorithmVisualization
    def get_metrics(self) -> Dict[str, float]
    def toggle(self, enabled: bool) -> None
    def reset(self) -> None
```

### EventBus (algorithm communication)
- `EventType.TICK` - Simulation step
- `EventType.CONGESTION_DETECTED` - Triggers ACO rerouting
- `EventType.SIGNAL_PHASE_CHANGED` - SOTL decisions
- `EventType.PATTERN_RECOGNIZED` - SOM classification

---

## Implementation Approach: Incremental MVP

Start with CA + SOTL working end-to-end, then add algorithms incrementally.

### Phase 1: Minimal Core + Network ✅ COMPLETE
- [x] Project structure and requirements.txt
- [x] **YAML-based network configuration** (custom topology support)
- [x] Network loader (intersections, roads from YAML)
- [x] Basic Pygame window displaying configurable network
- [x] Minimal SimulationState

### Phase 2: Traffic Flow (CA) - First Working Demo ✅ COMPLETE
- [x] Cellular Automata engine (NaSch rules)
- [x] Vehicle spawning/movement/despawning
- [x] VehicleLayer rendering with interpolation
- [x] Fixed-timing traffic signals
- [x] **Basic metrics panel** (throughput, delay, queue length)

### Phase 3: Self-Organizing Lights (SOTL) + Core Demo Features ✅ COMPLETE
- [x] SOTL controller with phase/platoon/gap rules
- [x] Queue visualization at intersections
- [x] **Comparison mode** (split-screen: fixed vs SOTL)
- [x] **Speed controls** (1x, 2x, 5x, pause)
- [x] **Algorithm toggle UI**
- [x] EventBus for algorithm communication

**MILESTONE: First complete demo (CA + SOTL with comparison) ✅**

### Phase 3.5: Visual Polish ✅ COMPLETE
- [x] **Color palette refinement** - Softer background, pastel algorithm colors
- [x] **Network layer cleanup** - Removed mid-road arrows, dashed lane dividers, rounded road ends
- [x] **Signal redesign** - Compact light strips on intersection edges, removed scattered circles
- [x] **Vehicle polish** - Speed-based coloring (red→white), shadows, windshield detail
- [x] **Overlay cleanup** - Reduced opacity (0.35), subtle SOTL glows, thin ACO trails
- [x] **Dashboard modernization** - Dark theme, divider lines, colored text toggles
- [x] **Grid spacing** - Increased from 200px to 250px for better clarity
- [x] **Default config** - Only CA enabled, overlays hidden by default

### Phase 4: Dynamic Routing (ACO) + Interactive Controls ✅ COMPLETE
- [x] Pheromone matrix and route computation
- [x] Evaporation and deposit mechanics
- [x] **Pheromone trail visualization** (semi-transparent green paths)
- [x] Congestion detection with hysteresis (0.7 threshold, 0.5 clear)
- [x] Vehicle rerouting using ACO pheromone weights at intersections
- [x] **Interactive incident injection** (click to add incident, visual feedback)
- [x] ACO responds to INCIDENT_INJECTED events (70% pheromone reduction)
- [x] Route continuity validation after rerouting

**MILESTONE: ACO routing with congestion handling complete ✅**

### Phase 5: Full Dashboard + Pattern Recognition (SOM) ✅ PARTIAL
- [x] **Full metrics dashboard** with live sparkline graphs (native pygame, not matplotlib)
- [x] Scenario presets (rush hour, incident) with F1/F2/F3 keyboard shortcuts
- [x] Dashboard shows per-algorithm metrics on toggle buttons
- [x] Scenario manager with incident ownership tracking
- [ ] SOM classifier with pre-trained weights
- [ ] U-matrix visualization (cluster heatmap)

**MILESTONE: Dashboard with sparklines and scenarios complete ✅**

### Phase 6: Signal Optimization (PSO)
- [ ] PSO optimizer with background threading
- [ ] Particle swarm visualization
- [ ] Convergence animation
- [ ] PSO informs SOTL timing

### Phase 7: Polish (MARL + Export)
- [ ] Simplified MARL (pre-trained, inference-only)
- [ ] Q-value confidence halos
- [ ] **Recording/screenshot export** for thesis figures
- [ ] Algorithm explanation tooltips
- [ ] Demo script mode

---

## Critical Files to Create

1. **src/core/simulation.py** - Main loop with fixed timestep (30 FPS target) ✅
2. **src/algorithms/base.py** - BaseAlgorithm abstract class ✅
3. **src/core/event_bus.py** - Thread-safe pub/sub system ✅
4. **src/visualization/renderer.py** - Layer compositor ✅
5. **src/visualization/layers/overlay_layer.py** - Algorithm-specific overlays ✅

---

## Dashboard Layout

```
+------------------+------------------+
|                  |     SOITCS       |
|                  +------------------+
|                  |  METRICS         |
|   SIMULATION     |  Vehicles    45  |
|      VIEW        |  Avg Speed  3.2  |
|                  |  Avg Delay 2.1s  |
|   (1000x950)     |  Queue      1.5  |
|                  +------------------+
|                  |  ALGORITHMS      |
|                  |  [CA]    [SOTL]  |
|                  |  [ACO]   [PSO]   |
|                  |  [SOM]   [MARL]  |
|                  +------------------+
|                  |  SPEED           |
|                  | [0.5x][1x][2x][5x]
|                  | [  PAUSE (Space) ]
+------------------+------------------+
```

---

## Visual Design (Polished)

### Roads & Intersections
- Roads: Dark gray (#373A41) with rounded ends
- Lane dividers: Dashed lines (#5A5D64)
- Intersections: Circles with subtle borders

### Traffic Signals
- Compact rectangular strips on intersection edges
- Grouped by phase (NS=top/bottom, EW=left/right)
- Subtle glow effect around active signals
- Phase progress arc (optional)

### Vehicles
- Speed-based coloring: Red (stopped) → White (moving)
- Subtle drop shadows
- Windshield detail for direction indication
- No headlight dots (too detailed for scale)

### Algorithm Overlays
- Default: Hidden (toggle to show)
- Opacity: 0.35 (subtle, non-intrusive)
- SOTL: Glow rings on active intersections
- ACO: Thin semi-transparent pheromone trails

---

## Verification Plan

1. **Phase 1**: Window displays static road network ✅
2. **Phase 2**: Vehicles move and stop at red lights ✅
3. **Phase 3**: SOTL shows measurable improvement vs fixed timing ✅
4. **Phase 3.5**: Visual polish - clean roads, compact signals, visible vehicles ✅
5. **Phase 4**: Pheromone trails visible, vehicles reroute around congestion ✅
6. **Phase 5**: Dashboard shows live sparklines, scenarios work (rush hour, incident) ✅
7. **Phase 6**: PSO particles converge, signal timing improves
8. **Phase 7**: Full demo script runs without crashes, exports figures

---

## Performance Targets
- **30 FPS** minimum for smooth visualization ✅
- Heavy computation (ACO, PSO, MARL) in background threads
- Target **100-500 vehicles** for educational visualization

## Dependencies
```
pygame>=2.5.0
numpy>=1.24.0
pyyaml>=6.0
matplotlib>=3.7.0  # Optional: for SOM U-matrix heatmaps (sparklines use native pygame)
```

---

## Algorithm Configuration (YAML Tunable Parameters)

```yaml
# config/algorithms.yaml
cellular_automata:
  enabled: true              # Only CA enabled by default
  max_velocity: 5
  slow_probability: 0.3
  lane_change_threshold: 2

sotl:
  enabled: false             # Disabled by default for cleaner visuals
  min_green_time: 10
  max_green_time: 60
  theta: 5
  mu: 3
  omega: 15

aco:
  enabled: false             # Disabled by default for cleaner visuals
  alpha: 1.0
  beta: 2.0
  rho: 0.1
  Q: 100
  reroute_frequency: 10

pso:
  enabled: false
  swarm_size: 30
  c1: 2.0
  c2: 2.0
  w: 0.7
  optimization_interval: 50

som:
  enabled: false
  grid_size: [10, 10]
  learning_rate: 0.5
  sigma: 3.0
  pretrained_path: "data/pretrained_models/som_traffic_patterns.pkl"

marl:
  enabled: false
  learning_rate: 0.001
  gamma: 0.99
  epsilon: 0.1
  pretrained_path: "data/pretrained_models/marl_signal_control.pt"
  inference_only: true
```

---

## ML Training Data Strategy

### For SOM (Pattern Recognition)
**Option 1: Generate from simulation**
1. Run simulation for 10,000+ ticks with varying demand
2. Collect feature vectors every 10 ticks:
   - Average speed per road segment
   - Queue lengths at intersections
   - Total vehicle count
   - Time of day (simulated)
3. Train SOM offline using minisom or sklearn
4. Save cluster labels: "free_flow", "moderate", "rush_hour", "incident"

**Option 2: Use public traffic datasets**
- [UTD19](https://utd19.ethz.ch/) - Zurich traffic loop detector data
- [PeMS](https://pems.dot.ca.gov/) - California traffic data
- [UK Traffic Counts](https://roadtraffic.dft.gov.uk/) - Historical flow data

### For MARL (Signal Control)
**Option 1: Pre-train in simulation**
1. Create simplified training environment (single intersection)
2. Train DQN/PPO for 100k episodes offline
3. Transfer weights to full network (fine-tune or freeze)

**Option 2: Use existing implementations**
- [SUMO-RL](https://github.com/LucasAlegworthy/sumo-rl) - Pre-trained traffic signal agents
- [Flow](https://github.com/flow-project/flow) - RL for traffic (has pre-trained models)

**Recommendation**: Start with SOM (simpler), skip MARL for MVP, add it in Phase 7.

---

## Example Network Configuration (YAML)

```yaml
# config/scenarios/grid_4x4.yaml
name: "4x4 Grid Network"
description: "Simple grid for demonstrating emergence"

intersections:
  - id: 0
    position: [100, 100]      # Increased spacing (250px between)
    signal_phases:
      - duration: 30
        green_directions: [north, south]
      - duration: 30
        green_directions: [east, west]
  - id: 1
    position: [350, 100]
    # ...

roads:
  - id: 0
    from: 0
    to: 1
    lanes: 1
    length: 25               # Adjusted for new spacing
    speed_limit: 5
  # ...

spawn_points:
  - intersection: 0
    rate: 0.08
    destinations: [15, 3, 12]

scenarios:
  rush_hour:
    spawn_rate_multiplier: 3.0
    duration: 300
  incident:
    blocked_road: 8
    duration: 100
```

---

## First Implementation Steps

1. ~~Create project structure with `main.py`, `requirements.txt`~~ ✅
2. ~~Implement YAML network loader~~ ✅
3. ~~Create basic Pygame window that renders the loaded network~~ ✅
4. ~~Add vehicles with CA movement rules~~ ✅
5. ~~Implement fixed-timing signals~~ ✅
6. ~~Add SOTL and comparison mode~~ ✅
7. ~~Visual polish pass~~ ✅

**Next steps:**
- Implement SOM pattern recognition with pre-trained weights (Phase 5)
- Add U-matrix visualization for SOM clusters (Phase 5)
- Begin PSO signal optimization (Phase 6)

---

## 6-Month Timeline

| Month | Phase | Deliverable | Status |
|-------|-------|-------------|--------|
| **1** | Phase 1-2 | YAML network loader, CA traffic flowing, fixed signals | ✅ Complete |
| **2** | Phase 3 | SOTL working, comparison mode, basic UI controls | ✅ Complete |
| **2.5** | Phase 3.5 | Visual polish - clean rendering, modern dashboard | ✅ Complete |
| **3** | Phase 4 | ACO routing with pheromone visualization, incident injection | ✅ Complete |
| **4** | Phase 5 | Dashboard sparklines, scenarios, per-algorithm metrics | ✅ Partial (SOM pending) |
| **5** | Phase 6 | PSO signal optimization, threading for performance | Pending |
| **6** | Phase 7 | MARL (if time), polish, recording, thesis figures | Pending |

**Buffer**: Month 6 has slack for debugging, thesis writing, and demo preparation.

---

## Critical Risks & Mitigations

### 1. Pygame Performance Bottleneck
**Risk**: 100+ vehicles with overlays may drop below 30 FPS
**Mitigation**:
- Use `pygame.sprite.Group` with dirty rect updates
- Pre-render static elements (roads, intersections)
- Move heavy computation (ACO, PSO) to threads with `concurrent.futures`
- ✅ Reduced overlay opacity and complexity

### 2. Algorithm Interference
**Risk**: ACO rerouting conflicts with SOTL signal timing (feedback loops)
**Mitigation**:
- Use EventBus with priority levels - SOTL gets final say on signals
- Add coordination protocol: ACO respects current signal state when routing
- Implement "cooldown" periods after major decisions

### 3. Visualization Chaos
**Risk**: 6 algorithm overlays become unreadable mess
**Mitigation**:
- ✅ Default: only CA visible, others toggled on demand
- ✅ Opacity controls per layer (reduced to 0.35)
- "Focus mode" that highlights one algorithm at a time

### 4. YAML Config Complexity
**Risk**: Too many parameters overwhelms users
**Mitigation**:
- Provide "presets" (conservative, balanced, aggressive)
- Only expose commonly-tuned parameters in main config
- Advanced parameters in separate `advanced.yaml`

### 5. SOM/MARL Training Time
**Risk**: Training takes days, delays integration
**Mitigation**:
- Generate training data from Phase 3 simulation runs (already built)
- Use small SOM (5x5) initially, scale up later
- For MARL: use inference-only mode with pre-trained weights from SUMO-RL

### 6. Scope Creep
**Risk**: Adding features delays core deliverable
**Mitigation**:
- **Hard cutoff**: If Phase 4 not complete by Month 3, drop PSO/MARL
- Each phase has explicit "done" criteria
- Thesis defense only needs Phases 1-4 (CA, SOTL, ACO, comparison mode)

---

## Minimum Viable Thesis (If Time Runs Short)

If behind schedule, this subset is sufficient for thesis defense:

1. **CA traffic simulation** - Shows understanding of cellular automata ✅
2. **SOTL self-organizing lights** - Demonstrates emergence ✅
3. **ACO routing** - Shows swarm intelligence ✅
4. **Comparison mode** - Proves algorithms improve performance ✅
5. **Basic metrics** - Quantifies improvement ✅
6. **Polished visuals** - Professional presentation quality ✅

Skip: SOM, PSO, MARL, live graphs, recording, fancy dashboard
