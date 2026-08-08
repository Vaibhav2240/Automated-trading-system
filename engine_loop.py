import time
import signal
from typing import Any
from core.event_bus import EventBus

class TradingEngine:
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.is_running = False

        # Register system termination signals for a graceful shutdown sequence
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)

    def start(self) -> None:
        """Launches the primary execution loop of the trading infrastructure."""
        print("[SYSTEM INFO] Initializing TradeX Core Execution Loop...")
        self.is_running = True

        # The infinite processing cycle
        while self.is_running:
            # Process everything currently pending in the event pipeline
            events_processed = False
            while self.bus.process_next():
                events_processed = True

            # Micro-throttle mechanism: Prevents 100% CPU utilization when queue is dry
            if not events_processed:
                time.sleep(0.001)  # 1 millisecond sleep interval

        self._execute_clean_shutdown()

    def _handle_shutdown_signal(self, signum: int, frame: Any) -> None:
        """Interceptors for OS kill signals (e.g., Ctrl+C or kill command)."""
        print(f"\n[SYSTEM WARNING] Shutdown signal received ({signum}). Initiating exit protocol...")
        self.is_running = False

    def _execute_clean_shutdown(self) -> None:
        """Guarantees the system enters a safe state prior to closing the application."""
        print("[SAFETY INFO] Commencing structural isolation sequence...")

        # 1. Process any remaining messages left on the queue
        print("[SAFETY INFO] Flushing remaining pipeline events...")
        while self.bus.process_next():
            pass

        # 2. Hard Stop placeholder
        # Later, we will explicitly tap into the safety/circuit_breakers module
        # to cancel any unexecuted outstanding limit orders on Alpaca here.

        print("[SYSTEM INFO] TradeX Engine safely deactivated. Status: OFFLINE.")