import numpy as np
from collections import deque
from typing import Dict, Optional

class FeatureEngineer:
    def __init__(self, window_size: int = 14):
        """
        Initializes the feature engineering pipeline with fixed-size rolling windows.

        Args:
            window_size: The number of periods required to calculate rolling volatility.
        """
        self.window_size = window_size

        # We use deque (Double-Ended Queue) for O(1) performance.
        # When the queue is full, appending a new item automatically drops the oldest item.
        self.prices: deque = deque(maxlen=window_size + 1)
        self.log_returns: deque = deque(maxlen=window_size)

    def process_tick(self, close_price: float) -> Optional[Dict[str, float]]:
        """
        Ingests a new close price, updates the rolling windows, and calculates math features.

        Args:
            close_price: The latest market price.

        Returns:
            A dictionary containing 'log_return' and 'volatility', or None if the window is warming up.
        """
        self.prices.append(close_price)

        # We need at least 2 prices to calculate a single return
        if len(self.prices) >= 2:
            # Calculate continuous log return
            current_return = np.log(self.prices[-1] / self.prices[-2])
            self.log_returns.append(current_return)

        # We need a fully populated window to calculate statistically valid volatility
        if len(self.log_returns) == self.window_size:
            # Calculate standard deviation of log returns (ddof=1 for sample standard deviation)
            volatility = np.std(self.log_returns, ddof=1)

            return {
                "log_return": self.log_returns[-1],
                "volatility": volatility
            }

        # Return None if the system is still buffering the initial baseline data
        return None

    def reset(self) -> None:
        """Clears the rolling windows (used during daily system restarts or circuit breaker resets)."""
        self.prices.clear()
        self.log_returns.clear()