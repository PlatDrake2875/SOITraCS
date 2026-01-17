"""Road network graph with YAML loading."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
import yaml
import math

from .intersection import Intersection
from .road import Road
from .traffic_light import Direction, SignalPhase


@dataclass
class SpawnPoint:
    """Vehicle spawn point configuration."""

    intersection_id: int
    rate: float = 0.1  # Vehicles per tick probability
    destinations: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "SpawnPoint":
        """Create from dictionary."""
        return cls(
            intersection_id=data.get("intersection", data.get("intersection_id", 0)),
            rate=data.get("rate", 0.1),
            destinations=data.get("destinations", []),
        )


@dataclass
class RoadNetwork:
    """
    Complete road network with intersections and roads.

    Supports loading from YAML configuration files.
    """

    name: str = "Default Network"
    description: str = ""

    # Core entities
    intersections: Dict[int, Intersection] = field(default_factory=dict)
    roads: Dict[int, Road] = field(default_factory=dict)
    spawn_points: List[SpawnPoint] = field(default_factory=list)

    # Scenario definitions (raw dict, parsed by ScenarioManager)
    scenarios: Dict[str, Any] = field(default_factory=dict)

    # Adjacency for routing
    _adjacency: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)  # intersection -> [(neighbor, road_id)]
    _road_directions: Dict[Tuple[int, int], Direction] = field(default_factory=dict)  # (road_id, intersection_id) -> approach direction
    _reverse_roads: Dict[int, Optional[int]] = field(default_factory=dict)  # road_id -> reverse_road_id (or None)

    def add_intersection(self, intersection: Intersection) -> None:
        """Add an intersection to the network."""
        self.intersections[intersection.id] = intersection
        if intersection.id not in self._adjacency:
            self._adjacency[intersection.id] = []

    def add_road(self, road: Road) -> None:
        """Add a road to the network."""
        self.roads[road.id] = road

        # Update adjacency
        if road.from_intersection not in self._adjacency:
            self._adjacency[road.from_intersection] = []
        self._adjacency[road.from_intersection].append((road.to_intersection, road.id))

        # Update reverse road lookup
        self._update_reverse_lookup(road)

        # Update intersection connections
        from_int = self.intersections.get(road.from_intersection)
        to_int = self.intersections.get(road.to_intersection)

        if from_int and to_int:
            direction = self._calculate_direction(from_int.position, to_int.position)
            from_int.connect_outgoing(direction, road.id)
            to_int.connect_incoming(Direction.opposite(direction), road.id)
            self._road_directions[(road.id, road.to_intersection)] = Direction.opposite(direction)

    def _update_reverse_lookup(self, road: Road) -> None:
        """Update reverse road lookup for efficient bidirectional road queries."""
        # Find if there's an existing road going the opposite direction
        for other_id, other_road in self.roads.items():
            if other_id == road.id:
                continue
            if (other_road.from_intersection == road.to_intersection and
                other_road.to_intersection == road.from_intersection):
                # Found reverse road - link both directions
                self._reverse_roads[road.id] = other_id
                self._reverse_roads[other_id] = road.id
                return
        # No reverse road found
        self._reverse_roads[road.id] = None

    def get_reverse_road(self, road_id: int) -> Optional[Road]:
        """Get the reverse road (opposite direction) if it exists."""
        reverse_id = self._reverse_roads.get(road_id)
        if reverse_id is not None:
            return self.roads.get(reverse_id)
        return None

    def _calculate_direction(
        self,
        from_pos: Tuple[float, float],
        to_pos: Tuple[float, float]
    ) -> Direction:
        """Calculate direction from one position to another."""
        dx = to_pos[0] - from_pos[0]
        dy = to_pos[1] - from_pos[1]

        if abs(dx) > abs(dy):
            return Direction.EAST if dx > 0 else Direction.WEST
        else:
            return Direction.SOUTH if dy > 0 else Direction.NORTH

    def get_intersection(self, intersection_id: int) -> Intersection | None:
        """Get intersection by ID."""
        return self.intersections.get(intersection_id)

    def get_road(self, road_id: int) -> Road | None:
        """Get road by ID."""
        return self.roads.get(road_id)

    def get_approach_direction(self, road_id: int, intersection_id: int) -> Direction:
        """Get the approach direction for a road entering an intersection."""
        return self._road_directions.get((road_id, intersection_id), Direction.NORTH)

    def get_outgoing_roads(self, intersection_id: int) -> List[Road]:
        """Get all roads leaving an intersection."""
        roads = []
        for neighbor_id, road_id in self._adjacency.get(intersection_id, []):
            road = self.roads.get(road_id)
            if road:
                roads.append(road)
        return roads

    def get_incoming_roads(self, intersection_id: int) -> List[Road]:
        """Get all roads entering an intersection."""
        roads = []
        for road in self.roads.values():
            if road.to_intersection == intersection_id:
                roads.append(road)
        return roads

    def find_route(
        self,
        start: int,
        end: int,
        weights: Dict[int, float] | None = None
    ) -> List[int]:
        """
        Find route (list of road IDs) from start to end intersection.

        Uses Dijkstra's algorithm with optional weight overrides (for ACO).
        """
        if start == end or end == -1:
            # Random destination - pick any path
            outgoing = self.get_outgoing_roads(start)
            if outgoing:
                return [outgoing[0].id]
            return []

        # Dijkstra's algorithm
        distances: Dict[int, float] = {start: 0}
        previous: Dict[int, Tuple[int, int]] = {}  # intersection -> (prev_intersection, road_id)
        unvisited: Set[int] = set(self.intersections.keys())

        while unvisited:
            # Find closest unvisited
            current = None
            current_dist = float('inf')
            for node in unvisited:
                if node in distances and distances[node] < current_dist:
                    current = node
                    current_dist = distances[node]

            if current is None or current == end:
                break

            unvisited.remove(current)

            # Update neighbors
            for neighbor, road_id in self._adjacency.get(current, []):
                if neighbor not in unvisited:
                    continue

                road = self.roads.get(road_id)
                if not road:
                    continue

                # Use custom weight if provided (for ACO)
                if weights and road_id in weights:
                    edge_weight = weights[road_id]
                else:
                    edge_weight = road.get_travel_time()

                new_dist = current_dist + edge_weight
                if new_dist < distances.get(neighbor, float('inf')):
                    distances[neighbor] = new_dist
                    previous[neighbor] = (current, road_id)

        # Reconstruct path
        if end not in previous:
            return []

        route = []
        current = end
        while current in previous:
            prev, road_id = previous[current]
            route.append(road_id)
            current = prev

        route.reverse()
        return route

    def update(self, dt: float = 1.0) -> None:
        """Update all network elements."""
        for intersection in self.intersections.values():
            intersection.update(dt)
        for road in self.roads.values():
            road.update(dt)

    def reset(self) -> None:
        """Reset network to initial state."""
        for intersection in self.intersections.values():
            intersection.reset()
        for road in self.roads.values():
            road.reset()

    @classmethod
    def from_yaml(cls, path: Path) -> "RoadNetwork":
        """Load network from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)

        network = cls(
            name=data.get("name", "Unnamed Network"),
            description=data.get("description", ""),
        )

        # Load intersections
        for int_data in data.get("intersections", []):
            intersection = Intersection.from_dict(int_data)
            network.add_intersection(intersection)

        # Load roads
        for road_data in data.get("roads", []):
            road = cls._road_from_dict(road_data, network)
            network.add_road(road)

        # Load spawn points
        for spawn_data in data.get("spawn_points", []):
            spawn_point = SpawnPoint.from_dict(spawn_data)
            network.spawn_points.append(spawn_point)

        # Load scenarios (raw dict, ScenarioManager will parse)
        network.scenarios = data.get("scenarios", {})

        # If no spawn points, create defaults at edge intersections
        if not network.spawn_points:
            network._create_default_spawn_points()

        return network

    @classmethod
    def _road_from_dict(cls, data: dict, network: "RoadNetwork") -> Road:
        """Create road from dictionary."""
        from_int = network.intersections.get(data["from"])
        to_int = network.intersections.get(data["to"])

        start_pos = from_int.position if from_int else (0, 0)
        end_pos = to_int.position if to_int else (0, 0)

        return Road(
            id=data["id"],
            from_intersection=data["from"],
            to_intersection=data["to"],
            length=data.get("length", 20),
            speed_limit=data.get("speed_limit", 5),
            num_lanes=data.get("lanes", 1),
            start_pos=start_pos,
            end_pos=end_pos,
        )

    def _create_default_spawn_points(self) -> None:
        """Create spawn points at edge intersections."""
        # Find edge intersections (fewer than 4 connections)
        for int_id, intersection in self.intersections.items():
            connections = len(self._adjacency.get(int_id, []))
            incoming = len(self.get_incoming_roads(int_id))

            if connections + incoming < 4:
                # Edge intersection - create spawn point
                destinations = [
                    other_id for other_id in self.intersections
                    if other_id != int_id
                ]
                self.spawn_points.append(SpawnPoint(
                    intersection_id=int_id,
                    rate=0.05,
                    destinations=destinations[:3],  # Limit destinations
                ))

    @classmethod
    def create_default_grid(cls, rows: int = 4, cols: int = 4, spacing: int = 150) -> "RoadNetwork":
        """Create a default grid network for testing."""
        network = cls(name="Default Grid", description=f"{rows}x{cols} grid network")

        # Create intersections
        offset_x = 100
        offset_y = 100

        for row in range(rows):
            for col in range(cols):
                int_id = row * cols + col
                position = (offset_x + col * spacing, offset_y + row * spacing)

                intersection = Intersection(id=int_id, position=position)

                # Set signal phases
                phases = [
                    SignalPhase(green_directions={Direction.NORTH, Direction.SOUTH}, duration=30),
                    SignalPhase(green_directions={Direction.EAST, Direction.WEST}, duration=30),
                ]
                intersection.set_signal_phases(phases)

                network.add_intersection(intersection)

        # Create roads (bidirectional)
        road_id = 0
        road_length = spacing // 10  # Convert to cells

        for row in range(rows):
            for col in range(cols):
                int_id = row * cols + col

                # Horizontal roads
                if col < cols - 1:
                    right_id = int_id + 1
                    # East
                    network.add_road(Road(
                        id=road_id,
                        from_intersection=int_id,
                        to_intersection=right_id,
                        length=road_length,
                        speed_limit=5,
                        num_lanes=1,
                        start_pos=network.intersections[int_id].position,
                        end_pos=network.intersections[right_id].position,
                    ))
                    road_id += 1

                    # West
                    network.add_road(Road(
                        id=road_id,
                        from_intersection=right_id,
                        to_intersection=int_id,
                        length=road_length,
                        speed_limit=5,
                        num_lanes=1,
                        start_pos=network.intersections[right_id].position,
                        end_pos=network.intersections[int_id].position,
                    ))
                    road_id += 1

                # Vertical roads
                if row < rows - 1:
                    below_id = int_id + cols
                    # South
                    network.add_road(Road(
                        id=road_id,
                        from_intersection=int_id,
                        to_intersection=below_id,
                        length=road_length,
                        speed_limit=5,
                        num_lanes=1,
                        start_pos=network.intersections[int_id].position,
                        end_pos=network.intersections[below_id].position,
                    ))
                    road_id += 1

                    # North
                    network.add_road(Road(
                        id=road_id,
                        from_intersection=below_id,
                        to_intersection=int_id,
                        length=road_length,
                        speed_limit=5,
                        num_lanes=1,
                        start_pos=network.intersections[below_id].position,
                        end_pos=network.intersections[int_id].position,
                    ))
                    road_id += 1

        # Create spawn points at corners
        corner_ids = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
        for corner_id in corner_ids:
            destinations = [cid for cid in corner_ids if cid != corner_id]
            network.spawn_points.append(SpawnPoint(
                intersection_id=corner_id,
                rate=0.08,
                destinations=destinations,
            ))

        return network

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box of network (min_x, min_y, max_x, max_y)."""
        if not self.intersections:
            return (0, 0, 800, 600)

        positions = [i.position for i in self.intersections.values()]
        min_x = min(p[0] for p in positions) - 50
        min_y = min(p[1] for p in positions) - 50
        max_x = max(p[0] for p in positions) + 50
        max_y = max(p[1] for p in positions) + 50

        return (min_x, min_y, max_x, max_y)
