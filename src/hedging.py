"""
Delta hedging module for options positions
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class HedgePosition:
    """Hedge position details"""
    pair: str
    hedge_type: str  # 'spot' or 'forward'
    quantity: float
    entry_price: float
    entry_date: pd.Timestamp
    option_positions: List  # Related option positions


class DeltaHedger:
    """
    Automated delta hedging for option positions
    """

    def __init__(self, hedge_threshold: float = 0.01, use_forwards: bool = False):
        """
        Initialize delta hedger

        Parameters:
        -----------
        hedge_threshold : float
            Minimum delta exposure (as % of notional) to trigger hedge
        use_forwards : bool
            If True, use forwards for hedging; if False, use spot
        """
        self.hedge_threshold = hedge_threshold
        self.use_forwards = use_forwards
        self.hedge_positions = []
        self.hedge_history = []

    def calculate_portfolio_delta(self, positions: List, market_data: Dict[str, pd.DataFrame],
                                  models: Dict, date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate net delta exposure by currency pair
        """
        delta_by_pair = {}

        for position in positions:
            pair = position.get('pair', position.pair if hasattr(position, 'pair') else None)
            if not pair or pair not in market_data:
                continue

            pair_data = market_data[pair]
            if date not in pair_data.index:
                continue

            current_data = pair_data.loc[date]

            # Get position details
            tenor = position.get('tenor', position.tenor if hasattr(position, 'tenor') else '1M')
            direction = position.get('direction', position.direction if hasattr(position, 'direction') else 1)
            size = position.get('size', position.quantity if hasattr(position, 'quantity') else 0)

            # Calculate delta using appropriate model
            S = current_data['spot']
            K = position.get('strike', S)  # Default to ATM if strike not specified

            # Time to maturity
            tenor_days = {'1W': 7, '2W': 14, '3W': 21, '1M': 30, '2M': 60,
                          '3M': 90, '4M': 120, '6M': 180, '9M': 270, '1Y': 365}
            T = tenor_days.get(tenor, 30) / 365

            # Get volatility
            vol_col = f'atm_vol_{tenor}'
            sigma = current_data[vol_col] if vol_col in current_data else 0.1

            # Simplified rates (should use discount curves)
            r_d = 0.05
            r_f = 0.01

            # Calculate delta
            if 'gk' in models:
                try:
                    greeks = models['gk'].calculate_greeks(S, K, T, r_d, r_f, sigma, 'call')
                    delta = greeks.get('delta', 0.5)
                except:
                    delta = 0.5  # Default to 0.5 for ATM
            else:
                # Simple approximation
                delta = 0.5

            # Aggregate delta by pair
            position_delta = delta * size * direction

            if pair not in delta_by_pair:
                delta_by_pair[pair] = 0
            delta_by_pair[pair] += position_delta

        return delta_by_pair

    def determine_hedges(self, delta_by_pair: Dict[str, float],
                         market_data: Dict[str, pd.DataFrame],
                         date: pd.Timestamp) -> List[Dict]:
        """
        Determine required hedges based on delta exposure
        """
        hedges = []

        for pair, net_delta in delta_by_pair.items():
            if pair not in market_data:
                continue

            pair_data = market_data[pair]
            if date not in pair_data.index:
                continue

            current_spot = pair_data.loc[date, 'spot']

            # Check if hedge is needed (threshold is % of notional)
            notional_exposure = abs(net_delta * current_spot)
            threshold_notional = self.hedge_threshold * current_spot

            if notional_exposure > threshold_notional:
                # Need to hedge
                hedge_quantity = -net_delta  # Opposite sign to neutralize

                if self.use_forwards:
                    # Use 1M forward for hedging
                    if 'forward_1M' in pair_data.columns:
                        hedge_price = pair_data.loc[date, 'forward_1M']
                        hedge_type = 'forward_1M'
                    else:
                        hedge_price = current_spot
                        hedge_type = 'spot'
                else:
                    hedge_price = current_spot
                    hedge_type = 'spot'

                hedge = {
                    'pair': pair,
                    'hedge_type': hedge_type,
                    'quantity': hedge_quantity,
                    'entry_price': hedge_price,
                    'entry_date': date,
                    'net_delta_before': net_delta,
                    'notional': abs(hedge_quantity * hedge_price)
                }

                hedges.append(hedge)

        return hedges

    def execute_hedges(self, hedges: List[Dict], capital: float) -> Tuple[List[Dict], float]:
        """
        Execute hedges and return updated positions and capital
        """
        executed_hedges = []
        total_cost = 0

        for hedge in hedges:
            # Check if we have enough capital
            hedge_cost = abs(hedge['quantity'] * hedge['entry_price'])

            if hedge_cost > capital * 0.1:  # Max 10% of capital per hedge
                # Scale down hedge
                scale_factor = (capital * 0.1) / hedge_cost
                hedge['quantity'] *= scale_factor
                hedge_cost *= scale_factor

            # Execute hedge
            executed_hedge = HedgePosition(
                pair=hedge['pair'],
                hedge_type=hedge['hedge_type'],
                quantity=hedge['quantity'],
                entry_price=hedge['entry_price'],
                entry_date=hedge['entry_date'],
                option_positions=[]  # Link to related options
            )

            self.hedge_positions.append(executed_hedge)
            executed_hedges.append(hedge)
            total_cost += hedge_cost

            # Record in history
            self.hedge_history.append({
                **hedge,
                'executed': True,
                'cost': hedge_cost
            })

        return executed_hedges, total_cost

    def update_hedges(self, positions: List, market_data: Dict[str, pd.DataFrame],
                      models: Dict, date: pd.Timestamp, capital: float) -> Dict:
        """
        Main method to update all hedges
        """
        # Calculate current delta exposure
        delta_by_pair = self.calculate_portfolio_delta(positions, market_data, models, date)

        # Determine required hedges
        required_hedges = self.determine_hedges(delta_by_pair, market_data, date)

        # Execute hedges
        executed_hedges, total_cost = self.execute_hedges(required_hedges, capital)

        # Calculate residual delta after hedging
        residual_delta = {}
        for pair, delta in delta_by_pair.items():
            hedge_delta = sum([h['quantity'] for h in executed_hedges if h['pair'] == pair])
            residual_delta[pair] = delta + hedge_delta

        return {
            'delta_before': delta_by_pair,
            'hedges_executed': executed_hedges,
            'hedge_cost': total_cost,
            'residual_delta': residual_delta,
            'total_hedge_positions': len(self.hedge_positions)
        }

    def calculate_hedge_pnl(self, market_data: Dict[str, pd.DataFrame],
                            date: pd.Timestamp) -> float:
        """
        Calculate P&L from hedge positions
        """
        total_pnl = 0

        for hedge in self.hedge_positions:
            if hedge.pair not in market_data:
                continue

            pair_data = market_data[hedge.pair]
            if date not in pair_data.index:
                continue

            # Get current price
            if hedge.hedge_type == 'spot':
                current_price = pair_data.loc[date, 'spot']
            elif 'forward' in hedge.hedge_type:
                # Use forward or spot if forward not available
                col_name = f"forward_{hedge.hedge_type.split('_')[1]}"
                current_price = pair_data.loc[date, col_name] if col_name in pair_data.columns else pair_data.loc[
                    date, 'spot']
            else:
                current_price = pair_data.loc[date, 'spot']

            # Calculate P&L
            price_change = current_price - hedge.entry_price
            hedge_pnl = hedge.quantity * price_change
            total_pnl += hedge_pnl

        return total_pnl
