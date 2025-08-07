"""
SABR model for FX options
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from typing import Tuple, Optional


class SABRModel:
    """
    SABR (Stochastic Alpha Beta Rho) model for volatility smile
    """

    def __init__(self):
        self.name = "SABR"
        self.alpha = None
        self.beta = 0.5  # Often fixed for FX
        self.rho = None
        self.nu = None

    def calibrate(self, F: float, strikes: np.ndarray, T: float,
                  market_vols: np.ndarray, beta: float = 0.5) -> Tuple[float, float, float]:
        """
        Calibrate SABR parameters to market volatilities

        Parameters:
        -----------
        F : float
            Forward price
        strikes : np.ndarray
            Array of strike prices
        T : float
            Time to maturity
        market_vols : np.ndarray
            Market implied volatilities
        beta : float
            Beta parameter (often fixed)
        """
        self.beta = beta

        # Initial guess
        atm_vol = np.interp(F, strikes, market_vols)
        x0 = [atm_vol, 0.0, 0.3]  # [alpha, rho, nu]

        # Bounds
        bounds = [(0.001, 10), (-0.999, 0.999), (0.001, 3)]

        # Objective function
        def objective(params):
            alpha, rho, nu = params
            model_vols = np.array([
                self.sabr_vol(F, K, T, alpha, beta, rho, nu)
                for K in strikes
            ])
            return np.sum((model_vols - market_vols) ** 2)

        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')

        if result.success:
            self.alpha, self.rho, self.nu = result.x
            return self.alpha, self.rho, self.nu
        else:
            raise ValueError("SABR calibration failed")

    def sabr_vol(self, F: float, K: float, T: float, alpha: float,
                 beta: float, rho: float, nu: float) -> float:
        """
        Calculate SABR implied volatility using Hagan's approximation
        """
        if K <= 0 or F <= 0:
            return 0

        # Handle ATM case
        if abs(F - K) < 1e-6:
            var = alpha ** 2 * ((1 + (2 - 3 * rho ** 2) * nu ** 2 * T / 24))
            return alpha * var ** 0.5 / (F ** (1 - beta))

        # General case
        z = nu / alpha * (F * K) ** ((1 - beta) / 2) * np.log(F / K)
        x = np.log((np.sqrt(1 - 2 * rho * z + z ** 2) + z - rho) / (1 - rho))

        if abs(x) < 1e-6:
            return alpha / (F ** (1 - beta))

        A = alpha / ((F * K) ** ((1 - beta) / 2) *
                     (1 + (1 - beta) ** 2 * np.log(F / K) ** 2 / 24 +
                      (1 - beta) ** 4 * np.log(F / K) ** 4 / 1920))

        B = 1 + T * ((1 - beta) ** 2 * alpha ** 2 / (24 * (F * K) ** (1 - beta)) +
                     rho * beta * nu * alpha / (4 * (F * K) ** ((1 - beta) / 2)) +
                     nu ** 2 * (2 - 3 * rho ** 2) / 24)

        return A * z / x * B

    def price_option(self, S: float, K: float, T: float, r_d: float, r_f: float,
                     option_type: str = 'call') -> float:
        """
        Price option using SABR model
        """
        if self.alpha is None:
            raise ValueError("Model not calibrated")

        # Calculate forward
        F = S * np.exp((r_d - r_f) * T)

        # Get SABR vol
        sigma = self.sabr_vol(F, K, T, self.alpha, self.beta, self.rho, self.nu)

        # Use Black formula for pricing
        d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = np.exp(-r_d * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = np.exp(-r_d * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))

        return price