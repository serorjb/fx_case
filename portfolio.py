"""
Portfolio management for FX options
Tracks positions, P&L, and Greeks
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from config import Config


@dataclass
class Position:
    """Individual option position"""
    date: pd.Timestamp
    pair: str
    tenor: str
    strike: float
    is_call: bool
    direction: int  # 1 for long, -1 for short
    size: float
    entry_price: float
    entry_spot: float
    entry_vol: float
    model_price: float
    delta_at_entry: float

    # Updated fields
    current_price: float = 0
    current_delta: float = 0
    current_gamma: float = 0
    current_vega: float = 0
    current_theta: float = 0
    unrealized_pnl: float = 0
    days_held: int = 0

    # Exit fields
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    realized_pnl: Optional[float] = None

    def get_expiry_date(self, config: Config) -> pd.Timestamp:
        """Calculate expiry date"""
        days = config.TENOR_DAYS[self.tenor]
        return self.date + pd.Timedelta(days=days)

    def mark_to_market(self, current_price: float, greeks: dict):
        """Update position with current market values"""
        self.current_price = current_price
        self.current_delta = greeks.get('delta', 0)
        self.current_gamma = greeks.get('gamma', 0)
        self.current_vega = greeks.get('vega', 0)
        self.current_theta = greeks.get('theta', 0)

        # Calculate unrealized P&L
        price_change = current_price - self.entry_price
        self.unrealized_pnl = price_change * self.size * self.direction

    def close(self, exit_date: pd.Timestamp, exit_price: float):
        """Close the position"""
        self.exit_date = exit_date
        self.exit_price = exit_price

        price_change = exit_price - self.entry_price
        self.realized_pnl = price_change * self.size * self.direction


class Portfolio:
    """Portfolio of option positions"""

    def __init__(self, config: Config):
        self.config = config
        self.cash = config.INITIAL_CAPITAL
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []

        # Portfolio Greeks
        self.total_delta = 0
        self.total_gamma = 0
        self.total_vega = 0
        self.total_theta = 0

        # Spot hedge positions
        self.spot_hedges = {}  # pair -> size

    def add_position(self, position: Position):
        """Add a new position to portfolio"""
        self.positions.append(position)

    def close_position(self, position: Position, exit_date: pd.Timestamp, exit_price: float):
        """Close a position"""
        position.close(exit_date, exit_price)
        self.positions.remove(position)
        self.closed_positions.append(position)

        # Update cash
        if position.realized_pnl:
            self.cash += position.realized_pnl

    def close_expired_positions(self, current_date: pd.Timestamp) -> List[Position]:
        """Close positions that have expired"""
        expired = []

        for position in self.positions[:]:  # Copy list to modify during iteration
            expiry_date = position.get_expiry_date(self.config)

            if current_date >= expiry_date:
                # Calculate expiry value
                spot = position.entry_spot  # Simplified - should get current spot
                if position.is_call:
                    expiry_value = max(0, spot - position.strike)
                else:
                    expiry_value = max(0, position.strike - spot)

                position.close(current_date, expiry_value)
                self.positions.remove(position)
                self.closed_positions.append(position)
                expired.append(position)

                # Update cash
                self.cash += position.realized_pnl

        return expired

    def mark_to_market(self, current_date: pd.Timestamp, market_data: Dict, curves: pd.DataFrame):
        """Mark all positions to market"""

        from models import GarmanKohlhagen
        gk = GarmanKohlhagen()

        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0

        for position in self.positions:
            if position.pair not in market_data:
                continue

            pair_data = market_data[position.pair]
            if current_date not in pair_data.index:
                continue

            current = pair_data.loc[current_date]
            spot = current['spot']

            # Get current volatility
            atm_vol_col = f'atm_vol_{position.tenor}'
            if atm_vol_col in current:
                vol = current[atm_vol_col]
            else:
                vol = position.entry_vol

            # Calculate time to expiry
            expiry_date = position.get_expiry_date(self.config)
            days_to_expiry = (expiry_date - current_date).days
            T = max(0, days_to_expiry / 365)

            if T > 0:
                # Get rates (simplified)
                r_d = 0.05
                r_f = 0.02

                # Calculate current price and Greeks
                current_price = gk.price(spot, position.strike, T, r_d, r_f, vol, position.is_call)
                greeks = gk.calculate_greeks(spot, position.strike, T, r_d, r_f, vol, position.is_call)

                # Update position
                position.mark_to_market(current_price, greeks)
                position.days_held = (current_date - position.date).days

                # Aggregate Greeks (considering position direction and size)
                total_delta += position.current_delta * position.size * position.direction
                total_gamma += position.current_gamma * position.size * abs(position.direction)
                total_vega += position.current_vega * position.size * abs(position.direction)
                total_theta += position.current_theta * position.size * abs(position.direction)

        # Include spot hedges in delta
        for pair, hedge_size in self.spot_hedges.items():
            total_delta += hedge_size

        # Update portfolio Greeks
        self.total_delta = total_delta
        self.total_gamma = total_gamma
        self.total_vega = total_vega
        self.total_theta = total_theta

    def add_spot_hedge(self, pair: str, size: float):
        """Add spot hedge position"""
        if pair not in self.spot_hedges:
            self.spot_hedges[pair] = 0
        self.spot_hedges[pair] += size

    def get_portfolio_greeks(self) -> Dict:
        """Get current portfolio Greeks"""
        return {
            'delta': self.total_delta,
            'gamma': self.total_gamma,
            'vega': self.total_vega,
            'theta': self.total_theta
        }

    def get_total_value(self) -> float:
        """Calculate total portfolio value"""
        # Cash plus unrealized P&L
        total = self.cash

        for position in self.positions:
            total += position.unrealized_pnl

        return total

    def get_margin_usage(self) -> float:
        """Calculate margin usage"""
        if self.cash < 0:
            return -self.cash
        return 0

    def get_leverage(self) -> float:
        """Calculate current leverage"""
        total_value = self.get_total_value()
        if total_value > 0:
            margin = self.get_margin_usage()
            return 1 + (margin / total_value)
        return 1

    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions"""
        if not self.positions:
            return pd.DataFrame()

        data = []
        for pos in self.positions:
            data.append({
                'pair': pos.pair,
                'tenor': pos.tenor,
                'strike': pos.strike,
                'type': 'Call' if pos.is_call else 'Put',
                'direction': 'Long' if pos.direction > 0 else 'Short',
                'size': pos.size,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'delta': pos.current_delta,
                'days_held': pos.days_held
            })

        return pd.DataFrame(data)

    def calculate_interest(self, days: int = 1) -> float:
        """Calculate daily interest on cash/margin"""
        annual_rate = self.config.get_interest_rate(self.cash < 0)
        daily_rate = annual_rate / 365

        interest = abs(self.cash) * daily_rate * days

        if self.cash < 0:  # Paying interest on margin
            return -interest
        else:  # Earning interest on cash
            return interest