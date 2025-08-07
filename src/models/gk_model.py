"""
Garman-Kohlhagen model for FX options
"""
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from typing import Dict, Tuple, Optional
from src.models.base_model import BaseOptionModel

class GarmanKohlhagen(BaseOptionModel):
    """
    Garman-Kohlhagen model for pricing European FX options
    Extension of Black-Scholes for currency options
    """

    def __init__(self):
        super().__init__("Garman-Kohlhagen")

    def price_option(self, S: float, K: float, T: float, r_d: float, r_f: float,
                    sigma: float, option_type: str = 'call', **kwargs) -> float:
        """
        Price FX option using Garman-Kohlhagen model

        Parameters:
        -----------
        S : float
            Spot exchange rate (domestic/foreign)
        K : float
            Strike price
        T : float
            Time to maturity (years)
        r_d : float
            Domestic risk-free rate
        r_f : float
            Foreign risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        """
        if T <= 0:
            return max(0, S - K) if option_type == 'call' else max(0, K - S)

        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * np.exp(-r_f * T) * norm.cdf(d1) - K * np.exp(-r_d * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r_d * T) * norm.cdf(-d2) - S * np.exp(-r_f * T) * norm.cdf(-d1)

        return price

    def calculate_greeks(self, S: float, K: float, T: float, r_d: float, r_f: float,
                         sigma: float, option_type: str = 'call', **kwargs) -> Dict[str, float]:
        """Calculate option Greeks"""

        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r_d - r_f + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        # Common terms
        exp_rf_T = np.exp(-r_f * T)
        exp_rd_T = np.exp(-r_d * T)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        n_d1 = norm.pdf(d1)

        greeks = {}

        # Delta
        if option_type == 'call':
            greeks['delta'] = exp_rf_T * N_d1
        else:
            greeks['delta'] = -exp_rf_T * norm.cdf(-d1)

        # Gamma (same for calls and puts)
        greeks['gamma'] = exp_rf_T * n_d1 / (S * sigma * sqrt_T)

        # Vega (same for calls and puts)
        greeks['vega'] = S * exp_rf_T * n_d1 * sqrt_T / 100  # Divide by 100 for 1% vol move

        # Theta
        term1 = -S * exp_rf_T * n_d1 * sigma / (2 * sqrt_T)
        if option_type == 'call':
            term2 = r_f * S * exp_rf_T * N_d1
            term3 = -r_d * K * exp_rd_T * N_d2
        else:
            term2 = -r_f * S * exp_rf_T * norm.cdf(-d1)
            term3 = r_d * K * exp_rd_T * norm.cdf(-d2)
        greeks['theta'] = (term1 + term2 + term3) / 365  # Daily theta

        # Rho (domestic)
        if option_type == 'call':
            greeks['rho'] = K * T * exp_rd_T * N_d2 / 100
        else:
            greeks['rho'] = -K * T * exp_rd_T * norm.cdf(-d2) / 100

        # Phi (foreign rho)
        if option_type == 'call':
            greeks['phi'] = -S * T * exp_rf_T * N_d1 / 100
        else:
            greeks['phi'] = S * T * exp_rf_T * norm.cdf(-d1) / 100

        return greeks

    def implied_volatility(self, market_price: float, S: float, K: float, T: float,
                           r_d: float, r_f: float, option_type: str = 'call') -> float:
        """Calculate implied volatility using Newton-Raphson method"""

        # Initial guess
        sigma = 0.2

        # Newton-Raphson iteration
        for _ in range(100):
            price = self.price_option(S, K, T, r_d, r_f, sigma, option_type)
            vega = self.calculate_greeks(S, K, T, r_d, r_f, sigma, option_type)['vega'] * 100

            if abs(vega) < 1e-10:
                break

            diff = market_price - price
            if abs(diff) < 1e-6:
                break

            sigma = sigma + diff / vega
            sigma = max(0.001, min(sigma, 5.0))  # Bound volatility

        return sigma