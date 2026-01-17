"""Centralized constants for the traffic simulation."""

# Incident handling
PERMANENT_INCIDENT_DURATION = -1  # Sentinel for never-expiring incidents

# Vehicle rerouting
REROUTE_POSITION_THRESHOLD = 3  # Max cells from road start to allow reroute

# ACO algorithm parameters
ACO_DEPOSIT_MULTIPLIER = 0.01  # Pheromone deposit scaling factor
ACO_CONGESTION_PENALTY = 0.5   # Pheromone multiplier on congested roads
ACO_INCIDENT_PENALTY = 0.3     # Pheromone multiplier on incident roads

# Congestion detection thresholds
CONGESTION_THRESHOLD = 0.7       # Density to trigger congestion event
CONGESTION_CLEAR_THRESHOLD = 0.5  # Density to clear congestion (hysteresis)

# PSO fitness evaluation thresholds
PSO_HIGH_QUEUE_THRESHOLD = 5
PSO_VERY_HIGH_QUEUE_THRESHOLD = 8
PSO_SLOW_MIN_GREEN_THRESHOLD = 25
PSO_LONG_MAX_GREEN_THRESHOLD = 50
PSO_LOW_QUEUE_THRESHOLD = 2
PSO_QUICK_MIN_GREEN = 15

# PSO fitness penalties/rewards
PSO_HIGH_QUEUE_PENALTY = 0.1
PSO_LONG_PHASE_PENALTY = 0.05
PSO_BASELINE_PENALTY = 0.02
PSO_INVALID_CONFIG_PENALTY = 10.0
PSO_QUICK_SWITCH_BONUS = 0.5

# SOTL-PSO integration
SOTL_PSO_SMOOTHING_FACTOR = 0.3  # Weight for new PSO suggestions
SOTL_SUGGESTION_COOLDOWN = 30    # Min ticks between applying suggestions per intersection
