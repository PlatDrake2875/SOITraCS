"""Vehicle entity with cellular automaton movement."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto
import random

if TYPE_CHECKING:
    from .network import RoadNetwork, SpawnPoint
    from .road import Road, Lane
    from src.core.state import SimulationState


class VehicleState(Enum):
    """Vehicle states."""
    SPAWNING = auto()
    MOVING = auto()
    WAITING = auto()
    AT_INTERSECTION = auto()
    ARRIVED = auto()


@dataclass
class Vehicle:
    """
    A vehicle in the traffic simulation.

    Uses Nagel-Schreckenberg cellular automaton rules for movement.
    """

    id: int
    route: List[int]  # List of road IDs to follow

    # Current position
    current_road_id: int = -1
    current_lane: int = 0
    position: int = 0  # Cell position on current road

    # Velocity (cells per tick)
    speed: int = 0
    max_speed: int = 5

    # State
    state: VehicleState = VehicleState.SPAWNING

    # Metrics
    total_distance: int = field(default=0, init=False)
    total_time: int = field(default=0, init=False)
    wait_time: int = field(default=0, init=False)
    _route_index: int = field(default=0, init=False)

    # For interpolation (rendering)
    prev_position: int = field(default=0, init=False)
    interpolation_t: float = field(default=0.0, init=False)

    # Origin/destination for metrics
    origin: int = field(default=-1)
    destination: int = field(default=-1)

    @property
    def delay(self) -> float:
        """Calculate delay (actual time - free flow time)."""
        if self.total_distance <= 0:
            return 0.0
        free_flow_time = self.total_distance / self.max_speed
        return max(0, self.total_time - free_flow_time)

    @property
    def current_road_in_route(self) -> int | None:
        """Get current road ID from route."""
        if 0 <= self._route_index < len(self.route):
            return self.route[self._route_index]
        return None

    def update(self, dt: float, sim_state: "SimulationState") -> None:
        """
        Update vehicle position using NaSch CA rules.

        Steps:
        1. Acceleration: v = min(v + 1, v_max)
        2. Braking: v = min(v, gap)
        3. Randomization: v = max(v - 1, 0) with probability p
        4. Movement: x = x + v
        """
        if self.state == VehicleState.ARRIVED:
            return

        network = sim_state.network
        if not network:
            return

        road = network.get_road(self.current_road_id)
        if not road:
            self.state = VehicleState.ARRIVED
            return

        self.total_time += 1
        self.prev_position = self.position

        # Get current lane
        lane = road.get_lane(self.current_lane)
        if not lane:
            return

        # Check for intersection at end of road
        at_intersection = self.position >= lane.length - 3

        if at_intersection:
            self._handle_intersection(network, road, lane, sim_state)
        else:
            self._apply_nasch_rules(road, lane, sim_state)

    def _apply_nasch_rules(
        self,
        road: "Road",
        lane: "Lane",
        sim_state: "SimulationState",
        slow_prob: float = 0.3
    ) -> None:
        """Apply Nagel-Schreckenberg cellular automaton rules."""
        # Get algorithm config for slow probability
        ca_algo = sim_state.get_algorithm("cellular_automata")
        if ca_algo:
            slow_prob = ca_algo.config.get("slow_probability", 0.3)

        # Step 1: Acceleration
        max_v = min(self.max_speed, road.get_effective_speed_limit())
        new_speed = min(self.speed + 1, max_v)

        # Step 2: Braking (check gap ahead)
        is_clear, gap = lane.is_clear_ahead(self.position, new_speed)
        new_speed = min(new_speed, gap)

        # Step 3: Randomization
        if new_speed > 0 and random.random() < slow_prob:
            new_speed = max(new_speed - 1, 0)

        self.speed = new_speed

        # Step 4: Movement
        if self.speed > 0:
            new_position = self.position + self.speed
            if lane.move_vehicle(self.id, new_position):
                self.position = min(new_position, lane.length - 1)
                self.total_distance += self.speed
                self.state = VehicleState.MOVING
        else:
            self.wait_time += 1
            self.state = VehicleState.WAITING

    def _handle_intersection(
        self,
        network: "RoadNetwork",
        road: "Road",
        lane: "Lane",
        sim_state: "SimulationState"
    ) -> None:
        """Handle vehicle at intersection."""
        intersection = network.get_intersection(road.to_intersection)
        if not intersection:
            # End of network - vehicle arrived
            lane.remove_vehicle(self.id)
            self.state = VehicleState.ARRIVED
            return

        # Determine approach direction
        from_direction = network.get_approach_direction(road.id, intersection.id)

        # Check if we can proceed
        if intersection.can_proceed(from_direction):
            # Move to next road
            self._advance_to_next_road(network, lane, intersection)
        else:
            # Wait at intersection
            self.speed = 0
            self.wait_time += 1
            self.state = VehicleState.AT_INTERSECTION

    def _advance_to_next_road(
        self,
        network: "RoadNetwork",
        current_lane: "Lane",
        intersection: "Intersection"
    ) -> None:
        """Move vehicle to the next road in its route."""
        from .traffic_light import Direction

        # Remove from current lane
        current_lane.remove_vehicle(self.id)

        # Advance route index
        self._route_index += 1

        if self._route_index >= len(self.route):
            # Reached destination
            self.state = VehicleState.ARRIVED
            return

        next_road_id = self.route[self._route_index]
        next_road = network.get_road(next_road_id)

        if not next_road:
            self.state = VehicleState.ARRIVED
            return

        # Place on new road
        next_lane = next_road.get_lane(0)  # Default to first lane
        if next_lane and next_lane.place_vehicle(self.id, 0):
            self.current_road_id = next_road_id
            self.current_lane = 0
            self.position = 0
            self.state = VehicleState.MOVING
        else:
            # Can't enter - wait at intersection
            current_lane.place_vehicle(self.id, current_lane.length - 1)
            self.state = VehicleState.AT_INTERSECTION

    def get_interpolated_position(self) -> float:
        """Get interpolated position for smooth rendering."""
        return self.prev_position + (self.position - self.prev_position) * self.interpolation_t

    def get_world_position(self, network: "RoadNetwork") -> Tuple[float, float] | None:
        """Get world coordinates for rendering."""
        road = network.get_road(self.current_road_id)
        if not road:
            return None

        # Interpolate along road
        t = self.get_interpolated_position() / road.length if road.length > 0 else 0
        t = max(0, min(1, t))

        x = road.start_pos[0] + (road.end_pos[0] - road.start_pos[0]) * t
        y = road.start_pos[1] + (road.end_pos[1] - road.start_pos[1]) * t

        # Offset for lane
        if road.num_lanes > 1:
            # Calculate perpendicular offset
            dx = road.end_pos[0] - road.start_pos[0]
            dy = road.end_pos[1] - road.start_pos[1]
            length = (dx**2 + dy**2) ** 0.5
            if length > 0:
                lane_offset = (self.current_lane - (road.num_lanes - 1) / 2) * 8
                x += -dy / length * lane_offset
                y += dx / length * lane_offset

        return (x, y)

    def reset(self) -> None:
        """Reset vehicle state."""
        self.state = VehicleState.SPAWNING
        self.speed = 0
        self.total_distance = 0
        self.total_time = 0
        self.wait_time = 0
        self._route_index = 0

    @classmethod
    def spawn(
        cls,
        vehicle_id: int,
        spawn_point: "SpawnPoint",
        network: "RoadNetwork"
    ) -> Optional["Vehicle"]:
        """Spawn a new vehicle at a spawn point."""
        # Generate route
        route = network.find_route(
            spawn_point.intersection_id,
            random.choice(spawn_point.destinations) if spawn_point.destinations else -1
        )

        if not route:
            return None

        # Get first road
        first_road = network.get_road(route[0])
        if not first_road:
            return None

        # Check if can enter
        first_lane = first_road.get_lane(0)
        if not first_lane or not first_lane.is_empty(0):
            return None

        # Create vehicle
        vehicle = cls(
            id=vehicle_id,
            route=route,
            current_road_id=route[0],
            current_lane=0,
            position=0,
            origin=spawn_point.intersection_id,
            destination=spawn_point.destinations[0] if spawn_point.destinations else -1,
        )

        # Place on lane
        first_lane.place_vehicle(vehicle_id, 0)
        vehicle.state = VehicleState.MOVING

        return vehicle
