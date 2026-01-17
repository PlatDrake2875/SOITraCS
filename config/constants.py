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
