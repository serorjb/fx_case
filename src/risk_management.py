"""
Risk management module
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class RiskManager:
    """
    Risk management for FX options trading
    """

    def __init__(self, max_var: float = 0.02, max_leverage: float = 5):
        self.max_var = max_var  # Maximum VaR as % of capital
        self.max_leverage = max_leverage
        self.risk_limits = {
            'delta': 1000000,
            'gamma': 50000,
            'vega': 100000,
            'var': 0.02
        }

    def check_position_limits(self, position: dict, portfolio_greeks: dict,
                              capital: float) -> bool:
        """
        Check if new position violates risk limits
        """
        # Check Greeks limits
        for greek, limit in self.risk_limits.items():
            if greek in portfolio_greeks:
                if abs(portfolio_greeks[greek]) > limit:
                    print(f"Position rejected: {greek} limit exceeded")
                    return False

        # Check leverage
        total_exposure = sum([p['size'] for p in position])
        leverage = total_exposure / capital

        if leverage > self.max_leverage:
            print(f"Position rejected: Leverage {leverage:.2f} exceeds limit")
            return False

        return True

    def calculate_var(self, positions: List, returns: pd.Series,
                      confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk
        """
        if len(positions) == 0:
            return 0

        # Calculate portfolio returns
        portfolio_value = sum([p['size'] for p in positions])

        # Historical VaR
        var_percentile = (1 - confidence) * 100
        var = np.percentile(returns, var_percentile) * portfolio_value

        return abs(var)

    def calculate_stress_scenarios(self, positions: List,
                                   scenarios: Dict) -> pd.DataFrame:
        """
        Calculate P&L under stress scenarios
        """
        results = []

        for scenario_name, params in scenarios.items():
            scenario_pnl = 0

            for position in positions:
                # Simplified stress test
                spot_shock = params.get('spot_shock', 0)
                vol_shock = params.get('vol_shock', 0)

                # Approximate P&L
                delta_pnl = position['size'] * spot_shock
                vega_pnl = position['size'] * vol_shock * 0.01

                scenario_pnl += delta_pnl + vega_pnl

            results.append({
                'scenario': scenario_name,
                'pnl': scenario_pnl,
                'pnl_pct': scenario_pnl / sum([p['size'] for p in positions])
            })

        return pd.DataFrame(results)