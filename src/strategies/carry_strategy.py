"""
Carry to Volatility Ratio Strategy
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from .base_strategy import BaseStrategy, Signal


class CarryToVolStrategy(BaseStrategy):
    """
    Trade based on carry to volatility ratio
    """

    def __init__(self, min_ratio: float = 0.5):
        super().__init__("CarryToVol")
        self.min_ratio = min_ratio

    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Signal]:
        """
        Generate carry/vol signals
        """
        signals = []

        for pair, pair_data in data.items():
            if date not in pair_data.index:
                continue

            current = pair_data.loc[date]

            # Calculate carry for different tenors
            for tenor in ['3M', '6M', '1Y']:
                if f'fwd_points_{tenor}' not in pair_data.columns or f'atm_vol_{tenor}' not in pair_data.columns:
                    continue

                # Calculate annualized carry
                fwd_points = current[f'fwd_points_{tenor}']
                spot = current['spot']
                days = {'3M': 90, '6M': 180, '1Y': 365}[tenor]
                annualized_carry = (fwd_points / spot) * (365 / days)

                # Get volatility
                vol = current[f'atm_vol_{tenor}']

                if pd.isna(vol) or vol == 0 or pd.isna(annualized_carry):
                    continue

                # Calculate carry/vol ratio
                carry_vol_ratio = annualized_carry / vol

                # Historical percentile
                historical_ratio = (pair_data[f'fwd_points_{tenor}'] / pair_data['spot']) / pair_data[
                    f'atm_vol_{tenor}']
                percentile = (historical_ratio < carry_vol_ratio).rolling(252).mean().loc[date]

                if pd.isna(percentile):
                    continue

                # Generate signal
                if abs(carry_vol_ratio) > self.min_ratio:
                    # High positive carry/vol: bullish
                    # High negative carry/vol: bearish
                    direction = 1 if carry_vol_ratio > 0 else -1

                    # Stronger signal if in extreme percentiles
                    if percentile > 0.8 or percentile < 0.2:
                        confidence = 0.8
                    else:
                        confidence = 0.5

                    signal = Signal(
                        pair=pair,
                        tenor=tenor,
                        direction=direction,
                        confidence=confidence,
                        expected_edge=abs(carry_vol_ratio) * 0.01,
                        strategy_name=self.name
                    )
                    signals.append(signal)

        return signals

    def calculate_position_size(self, signal: Signal, capital: float) -> float:
        """Calculate position size"""
        base_size = capital * 0.015  # 1.5% per trade
        size = base_size * signal.confidence
        max_size = capital * 0.03  # Max 3% per position
        return min(size, max_size)