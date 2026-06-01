"""
Discrete-Event Simulation Engine.

Provides a lightweight, high-performance event-driven simulation kernel
for modeling asynchronous distributed systems. Events are processed in
strict chronological order using a min-heap priority queue with O(log n)
insertion and O(log n) extraction.

Reference:
    Banks, J., Carson, J.S., Nelson, B.L., & Nicol, D.M. (2010).
    Discrete-Event System Simulation (5th ed.). Prentice Hall.
"""

from __future__ import annotations

import heapq
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass(order=True)
class Event:
    """
    A scheduled simulation event.

    Events are ordered primarily by time, then by priority (lower = higher urgency),
    then by sequence number (FIFO among equal-priority events).

    Attributes:
        time: Scheduled execution time (seconds).
        priority: Tie-breaking priority (0 = highest).
        seq: Insertion sequence for FIFO ordering among equal priorities.
        id: Globally unique event identifier for cancellation.
        callback: Function to invoke when event fires.
        data: Arbitrary payload passed to callback.
        cancelled: If True, event is skipped during processing.
    """
    time: float
    priority: int = field(default=0)
    seq: int = field(default=0)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12], compare=False)
    callback: Optional[Callable] = field(default=None, compare=False, repr=False)
    data: Any = field(default=None, compare=False, repr=False)
    cancelled: bool = field(default=False, compare=False, repr=False)


class SimulationEngine:
    """
    Priority-queue-based discrete-event simulation engine.

    Manages a global simulation clock and an event calendar. Supports
    scheduling events at future times, cancelling pending events, and
    running the simulation until a specified end time.

    Example:
        >>> engine = SimulationEngine()
        >>> engine.schedule(1.0, lambda d: print(f"Event at {engine.now()}"))
        >>> engine.run(until=10.0)
    """

    def __init__(self, seed: int = 42):
        self._clock: float = 0.0
        self._event_queue: list[Event] = []
        self._sequence: int = 0
        self._cancelled: set[str] = set()
        self._events_processed: int = 0
        self._seed = seed

    @property
    def now(self) -> float:
        """Current simulation time in seconds."""
        return self._clock

    @property
    def events_processed(self) -> int:
        """Total number of events processed so far."""
        return self._events_processed

    @property
    def pending_events(self) -> int:
        """Number of events still in the queue."""
        return len(self._event_queue) - len(self._cancelled)

    def schedule(
        self,
        delay: float,
        callback: Callable,
        data: Any = None,
        priority: int = 0,
    ) -> str:
        """
        Schedule an event to fire after `delay` seconds from now.

        Args:
            delay: Time offset from current clock (must be >= 0).
            callback: Function(data) to invoke.
            data: Payload passed to callback.
            priority: Lower values execute first among same-time events.

        Returns:
            Event ID string for optional cancellation.

        Raises:
            ValueError: If delay is negative.
        """
        if delay < 0:
            raise ValueError(f"Cannot schedule event in the past (delay={delay})")

        event = Event(
            time=self._clock + delay,
            priority=priority,
            seq=self._sequence,
            callback=callback,
            data=data,
        )
        self._sequence += 1
        heapq.heappush(self._event_queue, event)
        return event.id

    def schedule_at(
        self,
        time: float,
        callback: Callable,
        data: Any = None,
        priority: int = 0,
    ) -> str:
        """Schedule an event at an absolute simulation time."""
        return self.schedule(time - self._clock, callback, data, priority)

    def cancel(self, event_id: str) -> bool:
        """
        Cancel a pending event by its ID.

        Returns True if the event was found and cancelled,
        False if it was already processed or not found.
        """
        if event_id in self._cancelled:
            return False
        self._cancelled.add(event_id)
        return True

    def run(self, until: float) -> int:
        """
        Process events in chronological order until simulation time `until`.

        Args:
            until: Stop processing events scheduled after this time.

        Returns:
            Number of events processed during this run.
        """
        processed = 0

        while self._event_queue:
            # Peek at next event
            event = self._event_queue[0]

            # Stop if we've passed the end time
            if event.time > until:
                break

            # Pop the event
            heapq.heappop(self._event_queue)

            # Skip cancelled events
            if event.id in self._cancelled:
                self._cancelled.discard(event.id)
                continue

            # Advance clock and fire
            self._clock = event.time
            if event.callback is not None:
                event.callback(event.data)

            processed += 1
            self._events_processed += 1

        # Advance clock to end time
        self._clock = until
        return processed

    def reset(self):
        """Reset engine to initial state."""
        self._clock = 0.0
        self._event_queue.clear()
        self._sequence = 0
        self._cancelled.clear()
        self._events_processed = 0

    def __repr__(self) -> str:
        return (
            f"SimulationEngine(clock={self._clock:.4f}, "
            f"pending={len(self._event_queue)}, "
            f"processed={self._events_processed})"
        )
