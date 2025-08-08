"""
Volatility Arbitrage Strategy - Simplified and Fixed
Trades when IV-RV spread deviates from historical mean
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Signal:
    """Trading signal with all required attributes"""
    timestamp: pd.Timestamp
    pair: str
    tenor: str = '1M'
    direction: int = 1  # 1: buy vol, -1: sell vol
    strength: float = 1.0
    confidence: float = 0.5
    expected_edge: float = 0.01
    strategy_name: str = "VolatilityArbitrage"  # Added this attribute
    signal_type: str = 'option'
    option_type: str = 'straddle'
    metadata: dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Position:
    """Option position details"""
    pair: str
    tenor: str
    strike: float
    option_type: str
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_vol: float
    entry_spot: float
    direction: int
    strategy_name: str


class VolatilityArbitrageStrategy:
    """
    Volatility Arbitrage Strategy - Fixed Version
    Trades when IV-RV spread deviates from historical mean
    """
    
    def __init__(self, 
                 lookback_window: int = 21,
                 zscore_threshold: float = 1.5,
                 enable_hedging: bool = True,
                 hedge_threshold: float = 0.01):
        """
        Parameters:
        -----------
        lookback_window : int
            Days to calculate realized volatility
        zscore_threshold : float
            Z-score threshold for signal generation
        enable_hedging : bool
            Whether to enable delta hedging (disabled for simplicity)
        hedge_threshold : float
            Hedge threshold (not used when hedging disabled)
        """
        self.name = "VolatilityArbitrage"
        self.lookback_window = lookback_window
        self.zscore_threshold = zscore_threshold
        self.enable_hedging = enable_hedging
        self.hedge_threshold = hedge_threshold
        self.positions = []
        self.closed_positions = []
        self.hedger = None  # No hedging for now
        
    def calculate_realized_vol(self, prices: pd.Series) -> pd.Series:
        """
        Calculate realized volatility without look-ahead bias
        """
        if len(prices) < 2:
            return pd.Series(index=prices.index, dtype=float)
            
        # Calculate log returns
        log_returns = np.log(prices / prices.shift(1))
        
        # Calculate rolling realized vol (annualized)
        # Use shift(1) to ensure we only use past returns
        realized_vol = log_returns.shift(1).rolling(self.lookback_window).std() * np.sqrt(252)
        
        return realized_vol
    
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Signal]:
        """
        Generate trading signals based on IV-RV spread
        """
        signals = []
        
        for pair, pair_data in data.items():
            # Check if date exists in data
            if date not in pair_data.index:
                continue
            
            # Get position of current date
            try:
                current_idx = pair_data.index.get_loc(date)
            except KeyError:
                continue
            
            # Need enough history
            min_history = self.lookback_window * 3
            if current_idx < min_history:
                continue
            
            # Get data up to current date (no look-ahead)
            historical_data = pair_data.iloc[:current_idx + 1]
            
            # Calculate realized volatility
            realized_vol_series = self.calculate_realized_vol(historical_data['spot'])
            current_realized = realized_vol_series.iloc[-1]
            
            if pd.isna(current_realized):
                continue
            
            # Check different tenors
            for tenor in ['1M', '3M', '6M']:
                iv_col = f'atm_vol_{tenor}'
                
                if iv_col not in historical_data.columns:
                    continue
                
                # Get current implied vol
                current_iv = historical_data[iv_col].iloc[-1]
                
                if pd.isna(current_iv):
                    continue
                
                # Calculate IV-RV spread
                current_spread = current_iv - current_realized
                
                # Calculate historical spreads (using only past data)
                historical_iv = historical_data[iv_col].iloc[:-1]  # Exclude current
                historical_rv = realized_vol_series.iloc[:-1]  # Exclude current
                
                # Align and calculate spreads
                historical_spreads = historical_iv - historical_rv
                historical_spreads = historical_spreads.dropna()
                
                # Need minimum samples
                if len(historical_spreads) < 30:
                    continue
                
                # Calculate z-score using expanding window (no look-ahead)
                mean_spread = historical_spreads.expanding().mean().iloc[-1]
                std_spread = historical_spreads.expanding().std().iloc[-1]
                
                if std_spread < 0.001:  # Avoid division by zero
                    continue
                
                z_score = (current_spread - mean_spread) / std_spread
                
                # Generate signal if threshold exceeded
                if abs(z_score) > self.zscore_threshold:
                    # Determine direction
                    if z_score > self.zscore_threshold:
                        # IV expensive relative to RV - sell volatility
                        direction = -1
                        trade_type = "Sell Vol"
                    else:
                        # IV cheap relative to RV - buy volatility
                        direction = 1
                        trade_type = "Buy Vol"
                    
                    # Calculate signal strength
                    strength = min(abs(z_score) / 3.0, 1.0)
                    
                    # Calculate confidence
                    confidence = 0.5 + 0.1 * min(abs(z_score) - self.zscore_threshold, 2.0)
                    confidence = min(confidence, 0.9)
                    
                    # Calculate expected edge
                    expected_edge = abs(current_spread) * 0.1  # Conservative estimate
                    
                    # Create signal with all required attributes
                    signal = Signal(
                        timestamp=date,
                        pair=pair,
                        tenor=tenor,
                        direction=direction,
                        strength=strength,
                        confidence=confidence,
                        expected_edge=expected_edge,
                        strategy_name=self.name,  # Important: set strategy name
                        signal_type='option',
                        option_type='straddle',
                        metadata={
                            'trade_type': trade_type,
                            'implied_vol': float(current_iv),
                            'realized_vol': float(current_realized),
                            'iv_rv_spread': float(current_spread),
                            'z_score': float(z_score),
                            'mean_spread': float(mean_spread),
                            'std_spread': float(std_spread),
                            'strike': float(historical_data['spot'].iloc[-1])  # ATM
                        }
                    )
                    
                    signals.append(signal)
                    
                    # Debug output for first few signals
                    if len(signals) <= 3:
                        print(f"    Signal: {pair} {tenor} {trade_type} | "
                              f"IV: {current_iv:.3f} RV: {current_realized:.3f} "
                              f"Z-score: {z_score:.2f}")
        
        return signals
    
    def calculate_position_size(self, signal: Signal, capital: float,
                               current_positions: List = None) -> float:
        """
        Calculate position size based on Kelly criterion
        """
        # Base size as percentage of capital
        base_size = capital * 0.02  # 2% base
        
        # Adjust by signal strength and confidence
        size = base_size * signal.strength * signal.confidence
        
        # Risk management: reduce size if many positions
        if current_positions:
            num_positions = len(current_positions)
            if num_positions > 5:
                size *= 0.8
            if num_positions > 10:
                size *= 0.6
        
        # Volatility adjustment
        if 'implied_vol' in signal.metadata:
            iv = signal.metadata['implied_vol']
            # Higher vol = smaller position
            vol_adjustment = 0.15 / max(iv, 0.05)  # Normalize to 15% vol
            size = size * min(vol_adjustment, 1.5)
        
        # Cap at maximum
        max_size = capital * 0.05  # 5% max per position
        
        return min(size, max_size)
    
    def should_exit_position(self, position: Position, current_data: pd.Series,
                           current_date: pd.Timestamp) -> bool:
        """
        Determine if position should be closed
        """
        # Calculate days held
        days_held = (current_date - position.entry_date).days
        
        # Exit after tenor expires (simplified)
        tenor_days = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365}.get(position.tenor, 30)
        if days_held > tenor_days * 0.5:  # Exit at half of tenor
            return True
        
        # Exit on stop loss/take profit based on spot movement
        spot_change = (current_data['spot'] - position.entry_spot) / position.entry_spot
        
        # For straddles, we lose if spot doesn't move (for long vol)
        # We win if spot doesn't move (for short vol)
        if position.direction == 1:  # Long vol
            # Need spot to move significantly
            if abs(spot_change) > 0.05:  # 5% move - take profit
                return True
            if days_held > 10 and abs(spot_change) < 0.01:  # No movement - stop loss
                return True
        else:  # Short vol
            # Need spot to stay still
            if abs(spot_change) < 0.02 and days_held > 10:  # Take profit
                return True
            if abs(spot_change) > 0.05:  # Big move - stop loss
                return True
        
        return False
    
    def update_positions(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Position]:
        """
        Update existing positions and check for exits
        """
        positions_to_close = []
        
        for position in self.positions:
            # Check if we have data for this position
            if position.pair not in data:
                continue
            
            pair_data = data[position.pair]
            if date not in pair_data.index:
                continue
            
            # Check exit conditions
            if self.should_exit_position(position, pair_data.loc[date], date):
                positions_to_close.append(position)
        
        return positions_to_close
    
    def execute_hedge(self, positions: List[Position], data: Dict[str, pd.DataFrame],
                     date: pd.Timestamp, capital: float) -> Dict:
        """
        Placeholder for hedging (disabled by default)
        """
        return {}
    
    def tenor_to_days(self, tenor: str) -> int:
        """Convert tenor string to days"""
        tenor_map = {
            '1W': 7, '2W': 14, '3W': 21,
            '1M': 30, '2M': 60, '3M': 90,
            '4M': 120, '6M': 180, '9M': 270,
            '1Y': 365, '12M': 365
        }
        return tenor_map.get(tenor, 30)
    
    def get_performance_summary(self) -> Dict:
        """
        Get strategy performance summary
        """
        if len(self.closed_positions) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'total_pnl': 0
            }
        
        # Calculate metrics
        pnls = [p.get('pnl', 0) for p in self.closed_positions]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        return {
            'total_trades': len(pnls),
            'win_rate': len(wins) / len(pnls) if len(pnls) > 0 else 0,
            'avg_profit': np.mean(pnls) if len(pnls) > 0 else 0,
            'total_pnl': sum(pnls),
            'avg_win': np.mean(wins) if len(wins) > 0 else 0,
            'avg_loss': np.mean(losses) if len(losses) > 0 else 0,
            'sharpe': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0
        }