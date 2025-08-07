"""
Base strategy class for FX options trading
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    """Option position details"""
    pair: str
    tenor: str
    strike: float
    option_type: str  # 'call' or 'put'
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    entry_vol: float
    entry_spot: float
    direction: int  # 1 for long, -1 for short
    strategy_name: str

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return {
            'pair': self.pair,
            'tenor': self.tenor,
            'strike': self.strike,
            'option_type': self.option_type,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_date': self.entry_date,
            'entry_vol': self.entry_vol,
            'entry_spot': self.entry_spot,
            'direction': self.direction,
            'strategy_name': self.strategy_name
        }

@dataclass
class Signal:
    """Trading signal"""
    pair: str
    tenor: str
    direction: int  # 1: buy, -1: sell, 0: neutral
    confidence: float  # 0 to 1
    expected_edge: float  # Expected profit as fraction
    strategy_name: str
    signal_type: str = 'option'  # 'option', 'spot', 'forward'
    metadata: dict = None  # Additional signal information

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseStrategy(ABC):
    """Base class for trading strategies"""

    def __init__(self, name: str):
        self.name = name
        self.positions = []
        self.closed_positions = []
        self.signals = []
        self.performance_history = []

    @abstractmethod
    def generate_signals(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Signal]:
        """
        Generate trading signals based on market data

        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Market data for each currency pair
        date : pd.Timestamp
            Current date for signal generation

        Returns:
        --------
        List[Signal] : List of trading signals
        """
        pass

    @abstractmethod
    def calculate_position_size(self, signal: Signal, capital: float,
                               current_positions: List[Position] = None) -> float:
        """
        Calculate position size based on signal and available capital

        Parameters:
        -----------
        signal : Signal
            Trading signal
        capital : float
            Available capital
        current_positions : List[Position]
            Current open positions for risk management

        Returns:
        --------
        float : Position size in currency units
        """
        pass

    def update_positions(self, data: Dict[str, pd.DataFrame], date: pd.Timestamp) -> List[Position]:
        """
        Update existing positions and check for exits

        Parameters:
        -----------
        data : Dict[str, pd.DataFrame]
            Current market data
        date : pd.Timestamp
            Current date

        Returns:
        --------
        List[Position] : Positions to close
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

    def should_exit_position(self, position: Position, current_data: pd.Series,
                            current_date: pd.Timestamp) -> bool:
        """
        Determine if position should be closed

        Override in derived classes for specific exit rules
        """
        # Default: exit if position is older than tenor
        days_held = (current_date - position.entry_date).days
        tenor_days = self.tenor_to_days(position.tenor)

        if days_held >= tenor_days:
            return True

        # Stop loss: exit if spot moved adversely by more than 5%
        spot_change = (current_data['spot'] - position.entry_spot) / position.entry_spot
        if position.direction * spot_change < -0.05:
            return True

        return False

    @staticmethod
    def tenor_to_days(tenor: str) -> int:
        """Convert tenor string to days"""
        tenor_map = {
            '1W': 7, '2W': 14, '3W': 21,
            '1M': 30, '2M': 60, '3M': 90,
            '4M': 120, '6M': 180, '9M': 270,
            '1Y': 365, '12M': 365
        }
        return tenor_map.get(tenor, 30)

    def calculate_signal_score(self, signal: Signal) -> float:
        """
        Calculate a score for signal prioritization
        """
        return signal.confidence * abs(signal.expected_edge)

    def filter_signals(self, signals: List[Signal], max_signals: int = 10) -> List[Signal]:
        """
        Filter and rank signals
        """
        # Sort by score
        signals_with_scores = [(s, self.calculate_signal_score(s)) for s in signals]
        signals_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top signals
        return [s[0] for s in signals_with_scores[:max_signals]]

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
