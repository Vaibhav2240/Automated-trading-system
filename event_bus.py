import queue
from typing import Dict, List, Callable, Type
from core.events import Event

class EventBus:
    def __init__(self):
        # High-performance, thread-safe FIFO (First-In, First-Out) queue
        self._queue: queue.Queue = queue.Queue()

        # Maps an Event type to a list of callback functions (subscribers)
        self._listeners: Dict[Type[Event], List[Callable]] = {}

    def subscribe(self, event_type: Type[Event], callback: Callable[[Event], None]) -> None:
        """Registers a module component to listen for a specific event type."""
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        self._listeners[event_type].append(callback)

    def publish(self, event: Event) -> None:
        """Places an event onto the queue to be processed by the engine loop."""
        self._queue.put(event)

    def process_next(self) -> bool:
        """
        Pops the oldest event from the queue and broadcasts it to subscribers.
        Returns True if an event was processed, False if the queue is empty.
        """
        try:
            # Non-blocking fetch; throws queue.Empty if nothing is there
            event = self._queue.get_nowait()
        except queue.Empty:
            return False

        # Find who is listening for this specific event type
        event_type = type(event)
        if event_type in self._listeners:
            for callback in self._listeners[event_type]:
                try:
                    # Execute the callback function safely
                    callback(event)
                except Exception as e:
                    # Critical Safeguard: One buggy module must not crash the entire core system
                    print(f"[CRITICAL ERROR] Failed to process {event_type.__name__}: {str(e)}")

        # Mark the task as completely done in the queue
        self._queue.task_done()
        return True

    @property
    def is_empty(self) -> bool:
        """Helper to check if any unprocessed events remain."""
        return self._queue.empty()