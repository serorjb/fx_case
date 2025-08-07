"""
Greeks calculation module
"""
import numpy as np
from scipy.stats import norm
from typing import Dict


class GreeksCalculator:
    """
    Calculate option Greeks for risk management
    """

    @staticmethod
    def calculate_portfolio_greeks(positions: list, market_data: dict) -> Dict[str, float]:
        """
        Calculate aggregate Greeks for portfolio
        """
        total_delta = 0
        total_gamma = 0
        total_vega = 0
        total_theta = 0
        total_rho = 0

        for position in positions:
            # Get position details
            pair = position['pair']
            tenor = position['tenor']
            size = position['size']
            direction = position['direction']

            # Calculate individual Greeks (simplified)
            # In practice, would use proper models
            greeks = GreeksCalculator.calculate_single_greek(
                position, market_data[pair]
            )

            # Aggregate
            total_delta += greeks['delta'] * size * direction
            total_gamma += greeks['gamma'] * size * abs(direction)
            total_vega += greeks['vega'] * size * abs(direction)
            total_theta += greeks['theta'] * size * abs(direction)
            total_rho += greeks['rho'] * size * direction

        return {
            'delta': total_delta,
            'gamma': total_gamma,
            'vega': total_vega,
            'theta': total_theta,
            'rho': total_rho
        }

    @staticmethod
    def calculate_single_greek(position: dict, market_data: dict) -> Dict[str, float]:
        """
        Calculate Greeks for single position
        """
        # Simplified Greeks calculation
        # In practice, would use GK model or other pricing models

        spot = market_data.get('spot', 100)
        vol = market_data.get('atm_vol_1M', 0.1)

        # Approximate Greeks
        delta = 0.5 * position.get('direction', 1)
        gamma = 0.01 / spot
        vega = spot * 0.4 * np.sqrt(30 / 365)
        theta = -spot * vol * 0.4 / (2 * np.sqrt(30 / 365))
        rho = 0.3 * spot * 30 / 365

        return {
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'theta': theta,
            'rho': rho
        }
