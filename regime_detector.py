import numpy as np
from core.events import MarketEvent, SignalEvent
from core.event_bus import EventBus
from brain.feature_engineering import FeatureEngineer
from brain.hmm_model import RegimeHMM

class HMMRegimeDetector:
    # Inside brain/regime_detector.py

    def __init__(self, event_bus=None, hmm_model=None, feature_engineer=None):
        # If event_bus wasn't passed, look up the parent instance dynamically to avoid messing with main_live.py
        if event_bus is None:
            import inspect
            # Walk up the call stack to find the running TradeXEngine instance self.bus
            frame = inspect.currentframe()
            try:
                while frame:
                    local_vars = frame.f_locals
                    if 'self' in local_vars and hasattr(local_vars['self'], 'bus'):
                        event_bus = local_vars['self'].bus
                        break
                    frame = frame.f_back
            finally:
                del frame # Prevent reference cycles in memory

        self.bus = event_bus

        # Surgical fix: If dependencies aren't passed from main,
        # the component handles its own instantiation internally.
        if hmm_model is None:
            from brain.hmm_model import GaussianHMM  # Replace with your actual class name
            self.hmm_model = GaussianHMM()
        else:
            self.hmm_model = hmm_model

        if feature_engineer is None:
            from brain.feature_engineering import FeatureEngineer  # Replace with your actual class name
            self.feature_engineer = FeatureEngineer()
        else:
            self.feature_engineer = feature_engineer

        # Subscribe this module to listen specifically for incoming market data
        # self.bus.subscribe(MarketEvent, self._handle_market_event)

    def _handle_market_event(self, event: MarketEvent) -> None:
        """Callback function triggered instantly every time a new price tick arrives."""

        # 1. Update the rolling mathematical windows
        mid_price = (event.bid + event.ask) / 2.0
        latest_features = self.features.process_tick(mid_price)

        # 2. Wait for the feature buffer to warm up (e.g., the first 14 ticks)
        if latest_features is None:
            return  # Not enough data yet; silently wait for the next tick.

        # 3. Format features for the HMM (requires a 2D array for scikit-learn compatibility)
        feature_vector = np.array([[latest_features["log_return"], latest_features["volatility"]]])

        try:
            # 4. Ask the HMM matrix for the current market regime
            current_state, probabilities = self.model.predict_current_state(feature_vector)

            # Extract the specific confidence score (probability) of the predicted state
            confidence = float(probabilities[current_state])

            # 5. Translate the hidden regime into a concrete trading direction
            # Note: We will map these exactly during your offline training phase.
            # Example assumption: State 0 = Low Vol/Uptrend (Buy), State 1 = High Vol/Downtrend (Sell/Flat)
            direction = 1 if current_state == 0 else -1

            # 6. Publish the mathematical conclusion back to the Event Bus
            signal = SignalEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                regime_state=current_state,
                direction=direction,
                strength=confidence
            )

            self.bus.publish(signal)

        except Exception as e:
            # If the math fails (e.g., NaN values in data), log it but don't crash the bot
            print(f"[BRAIN WARNING] Regime detection calculation failed for {event.symbol}: {str(e)}")