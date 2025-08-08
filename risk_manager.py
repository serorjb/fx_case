"""
===============================================================================
risk_manager.py - Risk management and delta hedging
===============================================================================
"""
from typing import Dict, List

import pandas as pd

from config import Config


class RiskManager:
    """Risk management and hedging"""

    def __init__(self, config: Config):
        self.config = config
        self.hedge_history = []

    def delta_hedge(self, portfolio, market_data: Dict, date: pd.Timestamp) -> List:
        """Execute delta hedging to maintain delta neutrality"""

        trades = []

        # Get current portfolio delta
        portfolio_delta = portfolio.total_delta

        # Check if hedging needed (delta exceeds threshold)
        portfolio_value = portfolio.get_total_value()
        delta_pct = abs(portfolio_delta) / portfolio_value if portfolio_value > 0 else 0

        if delta_pct > self.config.MAX_DELTA_PCT:
            # Need to hedge
            for pair in self.config.CURRENCY_PAIRS:
                if pair not in market_data:
                    continue

                pair_data = market_data[pair]
                if date not in pair_data.index:
                    continue

                spot = pair_data.loc[date, 'spot']

                # Calculate pair's contribution to portfolio delta
                pair_positions = [p for p in portfolio.positions if p.pair == pair]
                pair_delta = sum(p.current_delta * p.size * p.direction for p in pair_positions)

                if abs(pair_delta) > self.config.DELTA_HEDGE_THRESHOLD * portfolio_value:
                    # Hedge this pair's delta with spot
                    hedge_size = -pair_delta

                    # Add spot hedge
                    portfolio.add_spot_hedge(pair, hedge_size)

                    # Calculate cost including spread
                    notional = abs(hedge_size * spot)
                    cost = self.config.get_transaction_cost('spot', notional)

                    # Update cash
                    portfolio.cash -= cost

                    # Record hedge
                    self.hedge_history.append({
                        'date': date,
                        'pair': pair,
                        'size': hedge_size,
                        'spot': spot,
                        'cost': cost,
                        'delta_before': pair_delta,
                        'delta_after': 0
                    })

        return trades

    def check_risk_limits(self, portfolio) -> Dict[str, bool]:
        """Check if portfolio violates any risk limits"""

        checks = {
            'delta_limit': True,
            'leverage_limit': True,
            'var_limit': True
        }

        # Check delta limit
        portfolio_value = portfolio.get_total_value()
        if portfolio_value > 0:
            delta_pct = abs(portfolio.total_delta) / portfolio_value
            checks['delta_limit'] = delta_pct <= self.config.MAX_DELTA_PCT

        # Check leverage
        leverage = portfolio.get_leverage()
        checks['leverage_limit'] = leverage <= self.config.MAX_LEVERAGE

        return checks
