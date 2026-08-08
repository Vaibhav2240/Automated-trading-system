import sys
import threading
import time
from core.event_bus import EventBus
from brain.regime_detector import HMMRegimeDetector
from allocation.portfolio_state import PortfolioState
from allocation.position_sizing import PositionSizer
from safety.circuit_breakers import RiskCircuitBreakers
from broker.execution_router import IBKRExecutionRouter
from broker.data_streamer import IBKRDataStreamer
from core.events import MarketEvent, SignalEvent, OrderRequestEvent, OrderEvent, FillEvent

class TradeXEngine:
    def __init__(self):
        print("[SYSTEM START] Initializing TradeX Core Infrastructure...")

        # 1. Initialize the central Event Bus
        self.bus = EventBus()

        # 2. Initialize internal state layers
        self.portfolio = PortfolioState(initial_balance=100000.0) # Start with $100k paper equity
        self.sizer = PositionSizer(portfolio=self.portfolio, max_risk_per_trade=0.02, leverage=100)
        self.safety = RiskCircuitBreakers(max_lots_per_trade=5.0, cooldown_seconds=60.0)

        # 3. Initialize HMM Brain
        self.brain = HMMRegimeDetector()

        # 4. Initialize Connection Routing Layers
        self.router = IBKRExecutionRouter(event_bus=self.bus)
        self.streamer = IBKRDataStreamer(event_bus=self.bus)

        # 5. Register Centralized Event Subscriptions (The Pipeline Wiring)
        self._wire_pipeline()

    def _wire_pipeline(self):
        """Connects the components by registering event listeners on the bus."""
        # Market Ticks feed directly into the HMM Brain
        self.bus.subscribe(MarketEvent, self._handle_market_tick)

        # Brain signals feed into the Position Sizer
        self.bus.subscribe(SignalEvent, self._handle_signal)

        # Position sizing requests feed into Safety Circuit Breakers
        self.bus.subscribe(OrderRequestEvent, self._handle_order_request)

        # Approved orders feed into the physical Execution Router
        self.bus.subscribe(OrderEvent, self._handle_approved_order)

        # Trade confirmations route into the Portfolio Accounting state
        self.bus.subscribe(FillEvent, self.portfolio.update_from_fill)

    # --- Event Handler Bridges ---
    def _handle_market_tick(self, event: MarketEvent):
        # Let the brain process the mid-price tick and generate a signal if required
        signal = self.brain.process_market_data(event)
        if signal:
            self.bus.publish(signal)

    def _handle_signal(self, event: SignalEvent):
        # Sizer grabs the latest mid-price from the HMM features ledger
        mid_price = self.brain.features.get_latest_price(event.symbol)
        order_request = self.sizer.size_trade(event, mid_price)
        if order_request:
            self.bus.publish(order_request)

    def _handle_order_request(self, event: OrderRequestEvent):
        approved_order = self.safety.inspect_order(event)
        if approved_order:
            self.bus.publish(approved_order)

    def _handle_approved_order(self, event: OrderEvent):
        # Send the order to IBKR
        self.router.execute_order(event)

    def run(self):
        """Spins up background threads and starts the system engine."""
        # Create a dedicated background thread for the network data streaming socket
        stream_thread = threading.Thread(target=self.streamer.start_stream, daemon=True)
        stream_thread.start()

        print("[SYSTEM INFO] TradeX Engine is fully active and listening for live ticks...")

        try:
            # Keep the main process alive to monitor performance logs
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[SYSTEM SHUTDOWN] Safely severing broker pipes and closing sockets...")
            self.streamer.stop_stream()
            self.router.disconnect()
            print("[SHUTDOWN COMPLETE] Engine offline.")
            sys.exit(0)

if __name__ == "__main__":
    engine = TradeXEngine()
    engine.run()