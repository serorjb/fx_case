"""
Enhanced volatility arbitrage strategy with automated delta hedging
"""
import numpy as np
import pandas as pd
from typing import Dict, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.strategies.base_strategy import BaseStrategy, Signal, Position
from src.models.gk_model import GarmanKohlhagen
from src.models.gvv_model import GVVModel
from hedging import DeltaHedger

class VolatilityArbitrageStrategy(BaseStrategy):
    """
    Trade mispricing between implied and realized volatility with delta hedging
    """

    def __init__(self, lookback_window: int = 20, zscore_threshold: float = 2.0,
                 enable_hedging: bool = True, hedge_threshold: float = 0.01):
        """
        Initialize strategy

        Parameters:
        -----------
        lookback_window : int
            Window for calculating realized volatility
        zscore_threshold : float
            Z-score threshold for signal generation
        enable_hedging : bool
            Whether to enable automatic delta hedging
        hedge_threshold : float
            Minimum delta exposure to trigger hedge
        """
        super().__init__("VolatilityArbitrage")
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold
        self.enable_hedging = enable_hedging

        # Initialize models
        self.gk = GarmanKohlhagen()
        self.gvv = GVVModel()

        # Initialize hedger
        if enable_hedging:
            self.hedger = DeltaHedger(hedge_threshold=hedge_threshold)
        else:
            self.hedger = None

        # Track strategy-specific metrics
        self.vol_predictions = []
        self.hedge_history = []

    def calculate_realized_vol(self, prices: pd.Series, window: int = None) -> pd.Series:
        """
        Calculate realized volatility without look-ahead bias
        """
        if window is None:
            window = self.lookback_window

        # Use log returns
        log_returns = np.log(prices / prices.shift(1))

        # Calculate rolling realized vol (annualized)
        # Use shift(1) to ensure we only use past returns
        realized_vol = log_returns.shift(1).rolling(window).std() * np.sqrt(252)

        return realized_vol

    def calculate_vol_premium(self, data: pd.DataFrame, tenor: str) -> pd.Series:
        """
        Calculate volatility premium (IV - RV)
        """
        # Get implied vol
        iv_col = f'atm_vol_{tenor}'
        if iv_col not in data.columns:
            return pd.Series(index=data.index)

        implied_vol = data[iv_col]

        # Calculate realized vol
        realized_vol = self.calculate_realized_vol(data['spot'])

        # Vol premium
        vol_premium = implied_vol - realized_vol

        return vol_premium

    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Signal]:
        """
        Generate volatility arbitrage signals with no look-ahead bias
        """
        signals = []

        for pair, pair_data in data.items():
            if date not in pair_data.index:
                continue

            # Get index of current date
            date_idx = pair_data.index.get_loc(date)

            # Only use data up to current date (no look-ahead)
            historical_data = pair_data.iloc[:date_idx + 1]

            if len(historical_data) < self.lookback_window * 3:
                continue  # Need enough history

            # Current data point
            current = historical_data.iloc[-1]

            # Calculate realized volatility using only past data
            realized_vol = self.calculate_realized_vol(historical_data['spot'])
            current_realized = realized_vol.iloc[-1]

            if pd.isna(current_realized):
                continue

            # Check each tenor
            for tenor in ['1M', '3M', '6M']:
                if f'atm_vol_{tenor}' not in historical_data.columns:
                    continue

                # Get implied vol
                implied_vol = current[f'atm_vol_{tenor}']

                if pd.isna(implied_vol):
                    continue

                # Calculate vol premium
                vol_premium = implied_vol - current_realized

                # Calculate historical statistics (using only past data)
                historical_premium = historical_data[f'atm_vol_{tenor}'].iloc[:-1] - realized_vol.iloc[:-1]
                historical_premium = historical_premium.dropna()

                if len(historical_premium) < self.lookback_window:
                    continue

                # Use expanding window for mean/std to avoid look-ahead
                premium_mean = historical_premium.expanding().mean().iloc[-1]
                premium_std = historical_premium.expanding().std().iloc[-1]

                if pd.isna(premium_std) or premium_std == 0:
                    continue

                # Calculate z-score
                zscore = (vol_premium - premium_mean) / premium_std

                # Generate signal if threshold exceeded
                if abs(zscore) > self.zscore_threshold:
                    # Determine trade type
                    if zscore > self.zscore_threshold:
                        # IV too high relative to RV - sell volatility
                        direction = -1
                        trade_type = 'sell_straddle'
                    else:
                        # IV too low relative to RV - buy volatility
                        direction = 1
                        trade_type = 'buy_straddle'

                    # Calculate expected edge
                    expected_edge = abs(vol_premium) * 0.01  # Conservative estimate

                    # Calculate confidence based on z-score magnitude
                    confidence = min(abs(zscore) / 4, 1.0)

                    # Additional filters for signal quality
                    # 1. Check if vol regime is stable
                    vol_change = historical_data[f'atm_vol_{tenor}'].pct_change().iloc[-5:].std()
                    if vol_change > 0.2:  # High vol of vol - reduce confidence
                        confidence *= 0.7

                    # 2. Check if RV is converging to IV
                    rv_trend = realized_vol.iloc[-5:].mean() - realized_vol.iloc[-10:-5].mean()
                    iv_trend = historical_data[f'atm_vol_{tenor}'].iloc[-5:].mean() - historical_data[f'atm_vol_{tenor}'].iloc[-10:-5].mean()

                    if direction == -1 and rv_trend > 0 and iv_trend < 0:
                        # Good setup for selling vol
                        confidence *= 1.2
                    elif direction == 1 and rv_trend < 0 and iv_trend > 0:
                        # Good setup for buying vol
                        confidence *= 1.2

                    confidence = min(confidence, 1.0)

                    signal = Signal(
                        pair=pair,
                        tenor=tenor,
                        direction=direction,
                        confidence=confidence,
                        expected_edge=expected_edge,
                        strategy_name=self.name,
                        signal_type='option',
                        metadata={
                            'trade_type': trade_type,
                            'implied_vol': implied_vol,
                            'realized_vol': current_realized,
                            'vol_premium': vol_premium,
                            'zscore': zscore,
                            'strike': current['spot']  # ATM
                        }
                    )

                    signals.append(signal)

        return signals

    def execute_hedge(self, positions: List[Position], data: Dict[str, pd.DataFrame],
                     date: pd.Timestamp, capital: float) -> Dict:
        """
        Execute delta hedging for current positions
        """
        if not self.enable_hedging or not self.hedger:
            return {}

        # Calculate portfolio delta
        models = {'gk': self.gk, 'gvv': self.gvv}

        # Convert positions to format expected by hedger
        position_dicts = []
        for pos in positions:
            if hasattr(pos, 'strategy_name') and pos.strategy_name == self.name:
                position_dicts.append({
                    'pair': pos.pair,
                    'tenor': pos.tenor,
                    'direction': pos.direction,
                    'size': pos.quantity,
                    'strike': pos.strike,
                    'option_type': pos.option_type
                })

        # Update hedges
        hedge_result = self.hedger.update_hedges(
            position_dicts, data, models, date, capital
        )

        # Store hedge history
        self.hedge_history.append({
            'date': date,
            **hedge_result
        })

        return hedge_result

    def calculate_position_size(self, signal: Signal, capital: float,
                               current_positions: List[Position] = None) -> float:
        """
        Calculate position size using Kelly criterion with adjustments
        """
        # Base size as percentage of capital
        base_size = capital * 0.02  # 2% per trade

        # Adjust by confidence
        size = base_size * signal.confidence

        # Adjust by expected edge (Kelly fraction)
        # Kelly = edge / variance
        # We use a conservative estimate
        win_rate = 0.55  # Historical win rate for vol arb
        avg_win = signal.expected_edge
        avg_loss = signal.expected_edge * 0.8  # Slightly smaller losses

        # Kelly fraction
        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, min(kelly, 0.25))  # Cap at 25%

        size = size * kelly

        # Risk management adjustments
        if current_positions:
            # Reduce size if we have many positions
            num_positions = len([p for p in current_positions
                               if hasattr(p, 'strategy_name') and p.strategy_name == self.name])

            if num_positions > 5:
                size *= 0.8  # Reduce by 20%
            if num_positions > 10:
                size *= 0.6  # Reduce by 40%

        # Volatility adjustment
        if 'implied_vol' in signal.metadata:
            iv = signal.metadata['implied_vol']
            # Higher vol = smaller position
            vol_adjustment = 0.15 / max(iv, 0.05)  # Normalize to 15% vol
            size = size * min(vol_adjustment, 1.5)

        # Cap maximum position size
        max_size = capital * 0.05  # Max 5% per position

        return min(size, max_size)

    def should_exit_position(self, position: Position, current_data: pd.Series,
                            current_date: pd.Timestamp) -> bool:
        """
        Determine if position should be closed with vol-specific rules
        """
        # Standard exit conditions from base class
        if super().should_exit_position(position, current_data, current_date):
            return True

        # Vol arbitrage specific exits
        days_held = (current_date - position.entry_date).days

        # 1. Check if vol premium has reversed
        if f'atm_vol_{position.tenor}' in current_data:
            current_iv = current_data[f'atm_vol_{position.tenor}']
            entry_iv = position.entry_vol

            if position.direction == -1:  # Short vol
                # Exit if IV has increased significantly
                if current_iv > entry_iv * 1.2:
                    return True
            else:  # Long vol
                # Exit if IV has decreased significantly
                if current_iv < entry_iv * 0.8:
                    return True

        # 2. Time decay exit for short vol positions
        if position.direction == -1:
            # For short vol, exit earlier to capture theta
            tenor_days = self.tenor_to_days(position.tenor)
            if days_held > tenor_days * 0.5:  # Exit at 50% of tenor
                return True

        # 3. P&L based exit
        current_spot = current_data['spot']
        entry_spot = position.entry_spot
        spot_return = (current_spot - entry_spot) / entry_spot

        # Estimate P&L (simplified)
        if position.direction == -1:  # Short vol
            # Profit if spot hasn't moved much
            if abs(spot_return) < 0.01:  # Less than 1% move
                estimated_pnl = days_held * 0.001  # Theta profit
            else:
                estimated_pnl = -abs(spot_return) * 2  # Loss from gamma
        else:  # Long vol
            # Profit if spot has moved significantly
            if abs(spot_return) > 0.02:  # More than 2% move
                estimated_pnl = abs(spot_return) * 2  # Gain from gamma
            else:
                estimated_pnl = -days_held * 0.001  # Theta loss

        # Exit if significant profit
        if estimated_pnl > 0.05:  # 5% profit
            return True

        # Exit if significant loss (stop loss)
        if estimated_pnl < -0.03:  # 3% loss
            return True

        return False

    def update_positions(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Position]:
        """
        Update positions with hedging
        """
        # Get positions to close from parent
        positions_to_close = super().update_positions(data, date)

        # Execute hedging for remaining positions
        if self.enable_hedging and len(self.positions) > 0:
            # Calculate available capital (simplified)
            estimated_capital = 10000000  # Would get from portfolio

            hedge_result = self.execute_hedge(
                self.positions, data, date, estimated_capital
            )

            # Log hedging activity
            if hedge_result and 'hedges_executed' in hedge_result:
                num_hedges = len(hedge_result['hedges_executed'])
                if num_hedges > 0:
                    print(f"  Executed {num_hedges} hedges for {self.name}")

        return positions_to_close

    def get_performance_summary(self) -> Dict:
        """
        Get enhanced performance summary including hedging stats
        """
        base_summary = super().get_performance_summary()

        # Add vol arb specific metrics
        if self.hedge_history:
            total_hedges = sum([len(h.get('hedges_executed', []))
                              for h in self.hedge_history])
            total_hedge_cost = sum([h.get('hedge_cost', 0)
                                  for h in self.hedge_history])

            base_summary['total_hedges'] = total_hedges
            base_summary['total_hedge_cost'] = total_hedge_cost
            base_summary['avg_hedge_cost'] = total_hedge_cost / total_hedges if total_hedges > 0 else 0

        # Add vol prediction accuracy if available
        if self.vol_predictions:
            # Calculate accuracy metrics
            predictions_df = pd.DataFrame(self.vol_predictions)
            if 'predicted_vol' in predictions_df.columns and 'actual_vol' in predictions_df.columns:
                mae = (predictions_df['predicted_vol'] - predictions_df['actual_vol']).abs().mean()
                base_summary['vol_prediction_mae'] = mae

        return base_summary
