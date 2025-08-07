"""
Volatility arbitrage strategy
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from .base_strategy import BaseStrategy, Signal
import sys

sys.path.append('..')
from models.garman_kohlhagen import GarmanKohlhagen
from models.gvv_model import GVVModel


class VolatilityArbitrageStrategy(BaseStrategy):
    """
    Trade mispricing between implied and realized volatility
    """

    def __init__(self, lookback_window: int = 20, zscore_threshold: float = 2.0):
        super().__init__("VolatilityArbitrage")
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold
        self.gk = GarmanKohlhagen()
        self.gvv = GVVModel()

    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Signal]:
        """
        Generate volatility arbitrage signals
        """
        signals = []

        for pair, pair_data in data.items():
            if date not in pair_data.index:
                continue

            # Get current data
            current = pair_data.loc[date]

            # Calculate realized volatility
            returns = np.log(pair_data['spot'] / pair_data['spot'].shift(1))
            realized_vol = returns.rolling(self.lookback_window).std() * np.sqrt(252)

            # Check each tenor
            for tenor in ['1M', '3M', '6M']:
                if f'atm_vol_{tenor}' not in pair_data.columns:
                    continue

                # Get implied vol
                implied_vol = current[f'atm_vol_{tenor}']
                current_realized = realized_vol.loc[date]

                if pd.isna(implied_vol) or pd.isna(current_realized):
                    continue

                # Calculate vol premium
                vol_premium = implied_vol - current_realized

                # Calculate z-score
                historical_premium = pair_data[f'atm_vol_{tenor}'] - realized_vol
                premium_mean = historical_premium.rolling(self.lookback_window * 3).mean()
                premium_std = historical_premium.rolling(self.lookback_window * 3).std()

                if pd.isna(premium_mean.loc[date]) or pd.isna(premium_std.loc[date]) or premium_std.loc[date] == 0:
                    continue

                zscore = (vol_premium - premium_mean.loc[date]) / premium_std.loc[date]

                # Generate signal
                if abs(zscore) > self.zscore_threshold:
                    # If IV is too high relative to RV, sell volatility
                    if zscore > self.zscore_threshold:
                        direction = -1  # Sell vol (sell straddle)
                    else:
                        direction = 1  # Buy vol (buy straddle)

                    signal = Signal(
                        pair=pair,
                        tenor=tenor,
                        direction=direction,
                        confidence=min(abs(zscore) / 4, 1.0),  # Normalize confidence
                        expected_edge=abs(vol_premium) * 0.01,  # Convert to decimal
                        strategy_name=self.name
                    )
                    signals.append(signal)

        return signals

    def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """
        Calculate position size using Kelly criterion with cap
        """
        # Base size as percentage of capital
        base_size = capital * 0.02  # 2% per trade

        # Adjust by confidence
        size = base_size * signal.confidence

        # Adjust by expected edge
        kelly_fraction = min(signal.expected_edge / 0.1, 0.25)  # Cap at 25%
        size = size * kelly_fraction

        # Cap maximum position size
        max_size = capital * 0.05  # Max 5% per position
        return min(size, max_size)
