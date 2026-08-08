import threading
from typing import Dict
from core.events import FillEvent

class PortfolioState:
    def __init__(self, initial_balance: float = 10000.0):
        """
        Tracks the live state of the MT5 Forex/Metals portfolio.
        """
        self._lock = threading.Lock()

        # MT5 Standard Account Metrics
        self.balance: float = initial_balance      # Realized cash (closed positions only)
        self.equity: float = initial_balance       # Balance + Floating PnL
        self.used_margin: float = 0.0              # Capital locked up by active lots
        self.free_margin: float = initial_balance  # Usable capital for new trades

        # Active holdings mapping: symbol -> lots (Positive for Long, Negative for Short)
        self.positions: Dict[str, float] = {}

    def update_from_fill(self, event: FillEvent) -> None:
        """Updates local inventory when an order is executed (Approximation until MT5 sync)."""
        with self._lock:
            # Deduct standard commissions if applicable
            self.balance -= event.commission

            # Update lot inventory
            # In MT5, buying adds to position, selling subtracts.
            # We allow negative positions (Shorting) in Forex.
            current_lots = self.positions.get(event.symbol, 0.0)

            if event.side == "BUY":
                self.positions[event.symbol] = round(current_lots + event.quantity, 2)
            elif event.side == "SELL":
                self.positions[event.symbol] = round(current_lots - event.quantity, 2)

            # Clean up if flat
            if self.positions[event.symbol] == 0.0:
                del self.positions[event.symbol]

    def sync_with_mt5(self, balance: float, equity: float, margin: float, free_margin: float, positions: Dict[str, float]) -> None:
        """Hard-syncs the local state with MetaTrader 5's server state."""
        with self._lock:
            self.balance = balance
            self.equity = equity
            self.used_margin = margin
            self.free_margin = free_margin
            self.positions = positions
            print(f"[ALLOCATION INFO] MT5 Sync Complete. Free Margin: ${self.free_margin:.2f} | Equity: ${self.equity:.2f}")

    def get_position_size(self, symbol: str) -> float:
        """Safely fetches current lots held (can be negative if shorting)."""
        with self._lock:
            return self.positions.get(symbol, 0.0)

    def get_buying_power(self) -> float:
        """Returns Free Margin available for new allocations."""
        with self._lock:
            return self.free_margin