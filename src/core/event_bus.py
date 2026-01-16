"""Thread-safe pub/sub event system for algorithm communication."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Set
from threading import Lock
from collections import defaultdict
import time


class EventType(Enum):
    """Event types for algorithm communication."""

    # Simulation lifecycle
    TICK = auto()                      # Every simulation step
    SIMULATION_START = auto()
    SIMULATION_PAUSE = auto()
    SIMULATION_RESUME = auto()
    SIMULATION_RESET = auto()

    # Traffic events
    VEHICLE_SPAWNED = auto()
    VEHICLE_DESPAWNED = auto()
    VEHICLE_ARRIVED = auto()
    CONGESTION_DETECTED = auto()
    CONGESTION_CLEARED = auto()

    # Signal events
    SIGNAL_PHASE_CHANGED = auto()
    SIGNAL_TIMING_UPDATED = auto()

    # Algorithm events
    PATTERN_RECOGNIZED = auto()        # SOM classification
    ROUTE_UPDATED = auto()             # ACO rerouting
    OPTIMIZATION_COMPLETE = auto()     # PSO convergence
    POLICY_UPDATED = auto()            # MARL decision

    # User interaction
    INCIDENT_INJECTED = auto()
    ALGORITHM_TOGGLED = auto()
    SPEED_CHANGED = auto()


@dataclass
class Event:
    """Event container with metadata."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    source: str = ""  # Algorithm or component that fired the event
    priority: int = 0  # Higher priority events processed first

    def __lt__(self, other: "Event") -> bool:
        """Compare events by priority for priority queue."""
        return self.priority > other.priority  # Higher priority first


# Type alias for event handlers
EventHandler = Callable[[Event], None]


class EventBus:
    """Thread-safe publish/subscribe event system."""

    def __init__(self) -> None:
        self._handlers: Dict[EventType, List[EventHandler]] = defaultdict(list)
        self._pending_events: List[Event] = []
        self._lock = Lock()
        self._paused = False
        self._event_history: List[Event] = []
        self._history_limit = 1000

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Subscribe a handler to an event type."""
        with self._lock:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Unsubscribe a handler from an event type."""
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """Publish an event (queued for next flush)."""
        with self._lock:
            self._pending_events.append(event)

    def publish_immediate(self, event: Event) -> None:
        """Publish and immediately dispatch an event."""
        self._dispatch_event(event)

    def flush(self) -> int:
        """Process all pending events. Returns number of events processed."""
        if self._paused:
            return 0

        with self._lock:
            events = sorted(self._pending_events)
            self._pending_events = []

        count = 0
        for event in events:
            self._dispatch_event(event)
            count += 1

        return count

    def _dispatch_event(self, event: Event) -> None:
        """Dispatch a single event to all handlers."""
        # Store in history
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > self._history_limit:
                self._event_history = self._event_history[-self._history_limit:]

            handlers = list(self._handlers[event.type])

        # Call handlers outside lock to prevent deadlocks
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler for {event.type}: {e}")

    def pause(self) -> None:
        """Pause event processing."""
        self._paused = True

    def resume(self) -> None:
        """Resume event processing."""
        self._paused = False

    def clear_pending(self) -> None:
        """Clear all pending events."""
        with self._lock:
            self._pending_events.clear()

    def get_history(
        self,
        event_type: EventType | None = None,
        limit: int = 100
    ) -> List[Event]:
        """Get recent event history, optionally filtered by type."""
        with self._lock:
            if event_type:
                events = [e for e in self._event_history if e.type == event_type]
            else:
                events = list(self._event_history)
        return events[-limit:]

    @property
    def pending_count(self) -> int:
        """Number of pending events."""
        with self._lock:
            return len(self._pending_events)


# Global event bus instance
_event_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus instance."""
    global _event_bus
    if _event_bus is None:
        _event_bus = EventBus()
    return _event_bus


def reset_event_bus() -> None:
    """Reset the global event bus (for testing)."""
    global _event_bus
    _event_bus = EventBus()
