import math
from typing import Optional
from core.events import SignalEvent, OrderRequestEvent
from allocation.portfolio_state import PortfolioState

class PositionSizer:
    def __init__(self, portfolio: PortfolioState, max_risk_per_trade: float = 0.02, leverage: int = 100):
        self.portfolio = portfolio
        self.max_risk_per_trade = max_risk_per_trade
        self.leverage = leverage

    def size_trade(self, signal: SignalEvent, current_market_price: float) -> Optional[OrderRequestEvent]:
        if current_market_price <= 0:
            return None

        if signal.direction == 0:
            return self._generate_exit_order(signal.symbol, current_market_price)

        buying_power = self.portfolio.get_buying_power()
        total_equity = self.portfolio.total_equity

        adjusted_risk_pct = self.max_risk_per_trade * signal.strength
        target_dollar_allocation = total_equity * adjusted_risk_pct

        contract_size = 100 if "XAU" in signal.symbol else 5000
        margin_per_lot = (current_market_price * contract_size) / self.leverage
        target_lots = round(target_dollar_allocation / margin_per_lot, 2)

        if target_lots < 0.01:
            print(f"[ALLOCATION INFO] Target risk too small to purchase 0.01 lots of {signal.symbol}.")
            return None

        side = "BUY" if signal.direction == 1 else "SELL"

        return OrderRequestEvent(
            timestamp=signal.timestamp,
            symbol=signal.symbol,
            order_type="LIMIT",  # <-- CHANGED FROM "MARKET" TO "LIMIT"
            side=side,
            quantity=target_lots,
            current_price=current_market_price
        )

    def _generate_exit_order(self, symbol: str, current_price: float) -> Optional[OrderRequestEvent]:
        current_quantity = self.portfolio.get_position_size(symbol)
        if current_quantity == 0.0:
            return None

        return OrderRequestEvent(
            timestamp=0.0,
            symbol=symbol,
            order_type="LIMIT",  # <-- CHANGED FROM "MARKET" TO "LIMIT"
            side="SELL" if current_quantity > 0 else "BUY",
            quantity=abs(current_quantity),
            current_price=current_price
        )