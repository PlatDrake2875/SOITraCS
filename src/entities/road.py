"""Road and lane entities for the traffic network."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, TYPE_CHECKING
from enum import Enum, auto
import numpy as np

if TYPE_CHECKING:
    from .vehicle import Vehicle


class LaneType(Enum):
    """Types of lanes."""
    REGULAR = auto()
    TURNING = auto()
    BUS = auto()


@dataclass
class Lane:
    """
    A single lane within a road.

    Uses cellular automaton representation where each cell can hold one vehicle.
    """

    road_id: int
    lane_index: int
    length: int  # Number of cells
    lane_type: LaneType = LaneType.REGULAR

    # Cellular representation: cell[i] = vehicle_id or -1 if empty
    cells: np.ndarray = field(default=None, init=False)

    # Vehicle references for quick lookup
    vehicles: Dict[int, int] = field(default_factory=dict)  # vehicle_id -> position

    def __post_init__(self) -> None:
        """Initialize cell array."""
        self.cells = np.full(self.length, -1, dtype=np.int32)

    def is_empty(self, position: int) -> bool:
        """Check if a cell is empty."""
        if 0 <= position < self.length:
            return self.cells[position] == -1
        return False

    def is_clear_ahead(self, position: int, distance: int) -> Tuple[bool, int]:
        """
        Check if cells ahead are clear.

        Returns:
            Tuple of (is_clear, distance_to_obstacle)
        """
        for i in range(1, distance + 1):
            check_pos = position + i
            if check_pos >= self.length:
                return True, i  # End of road is clear
            if self.cells[check_pos] != -1:
                return False, i - 1
        return True, distance

    def place_vehicle(self, vehicle_id: int, position: int) -> bool:
        """Place a vehicle at a position."""
        if 0 <= position < self.length and self.cells[position] == -1:
            self.cells[position] = vehicle_id
            self.vehicles[vehicle_id] = position
            return True
        return False

    def remove_vehicle(self, vehicle_id: int) -> bool:
        """Remove a vehicle from the lane."""
        if vehicle_id in self.vehicles:
            position = self.vehicles[vehicle_id]
            self.cells[position] = -1
            del self.vehicles[vehicle_id]
            return True
        return False

    def move_vehicle(self, vehicle_id: int, new_position: int) -> bool:
        """Move a vehicle to a new position."""
        if vehicle_id not in self.vehicles:
            return False

        old_position = self.vehicles[vehicle_id]

        # Check if new position is valid
        if new_position >= self.length:
            # Vehicle exits lane
            self.cells[old_position] = -1
            del self.vehicles[vehicle_id]
            return True

        if new_position < 0 or self.cells[new_position] != -1:
            return False

        # Move vehicle
        self.cells[old_position] = -1
        self.cells[new_position] = vehicle_id
        self.vehicles[vehicle_id] = new_position
        return True

    def get_vehicle_position(self, vehicle_id: int) -> int | None:
        """Get position of a vehicle in this lane."""
        return self.vehicles.get(vehicle_id)

    def count_vehicles(self) -> int:
        """Count vehicles in lane."""
        return len(self.vehicles)

    def get_density(self) -> float:
        """Get lane density (0.0 - 1.0)."""
        return len(self.vehicles) / self.length if self.length > 0 else 0.0

    def get_queue_length(self) -> int:
        """Get number of stopped vehicles from end of lane."""
        queue = 0
        for i in range(self.length - 1, -1, -1):
            if self.cells[i] != -1:
                queue += 1
            elif queue > 0:
                break  # Gap in queue
        return queue

    def reset(self) -> None:
        """Reset lane to empty state."""
        self.cells.fill(-1)
        self.vehicles.clear()


@dataclass
class Road:
    """
    A road segment connecting two intersections.

    Contains one or more lanes in a single direction.
    """

    id: int
    from_intersection: int
    to_intersection: int
    length: int  # Cells
    speed_limit: int = 5  # Max velocity in cells/tick
    num_lanes: int = 1

    # Lanes
    lanes: List[Lane] = field(default_factory=list)

    # Road properties
    _incident: bool = field(default=False, init=False)
    _incident_timer: int = field(default=0, init=False)

    # Pheromone level (for ACO)
    pheromone: float = field(default=1.0, init=False)

    # Geometry (for rendering)
    start_pos: Tuple[float, float] = field(default=(0, 0))
    end_pos: Tuple[float, float] = field(default=(0, 0))

    def __post_init__(self) -> None:
        """Initialize lanes."""
        if not self.lanes:
            for i in range(self.num_lanes):
                self.lanes.append(Lane(
                    road_id=self.id,
                    lane_index=i,
                    length=self.length,
                ))

    @property
    def has_incident(self) -> bool:
        """Check if road has an active incident."""
        return self._incident

    def set_incident(self, duration: int) -> None:
        """Set an incident on this road."""
        self._incident = True
        self._incident_timer = duration

    def clear_incident(self) -> None:
        """Clear incident on this road."""
        self._incident = False
        self._incident_timer = 0

    def update(self, dt: float = 1.0) -> None:
        """Update road state (incidents, etc.)."""
        if self._incident:
            self._incident_timer -= 1
            if self._incident_timer <= 0:
                self.clear_incident()

    def get_lane(self, lane_index: int = 0) -> Lane | None:
        """Get a lane by index."""
        if 0 <= lane_index < len(self.lanes):
            return self.lanes[lane_index]
        return None

    def get_effective_speed_limit(self) -> int:
        """Get current speed limit (reduced during incidents)."""
        if self._incident:
            return max(1, self.speed_limit // 2)
        return self.speed_limit

    def count_vehicles(self) -> int:
        """Count total vehicles on road."""
        return sum(lane.count_vehicles() for lane in self.lanes)

    def get_density(self) -> float:
        """Get average lane density."""
        if not self.lanes:
            return 0.0
        return sum(lane.get_density() for lane in self.lanes) / len(self.lanes)

    def get_queue_length(self) -> int:
        """Get maximum queue length across lanes."""
        if not self.lanes:
            return 0
        return max(lane.get_queue_length() for lane in self.lanes)

    def get_travel_time(self) -> float:
        """Estimate travel time based on current conditions."""
        density = self.get_density()
        effective_speed = self.get_effective_speed_limit() * (1 - density * 0.5)
        if effective_speed <= 0:
            return float('inf')
        return self.length / effective_speed

    def deposit_pheromone(self, amount: float) -> None:
        """Deposit pheromone (for ACO)."""
        self.pheromone += amount

    def evaporate_pheromone(self, rate: float) -> None:
        """Evaporate pheromone (for ACO)."""
        self.pheromone = max(0.1, self.pheromone * (1 - rate))

    def reset(self) -> None:
        """Reset road to initial state."""
        for lane in self.lanes:
            lane.reset()
        self._incident = False
        self._incident_timer = 0
        self.pheromone = 1.0
