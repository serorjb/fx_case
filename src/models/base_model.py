# src/models/base_model.py
"""
Base model class for all option pricing models
"""
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import pandas as pd


class BaseOptionModel(ABC):
    """Base class for option pricing models"""

    def __init__(self, name: str = "BaseModel"):
        self.name = name
        self.is_calibrated = False

    @abstractmethod
    def price_option(self, S: float, K: float, T: float, r_d: float, r_f: float,
                     sigma: float, option_type: str = 'call', **kwargs) -> float:
        """
        Calculate option price

        Parameters:
        -----------
        S : float
            Spot price
        K : float
            Strike price
        T : float
            Time to maturity in years
        r_d : float
            Domestic risk-free rate
        r_f : float
            Foreign risk-free rate
        sigma : float
            Volatility
        option_type : str
            'call' or 'put'
        **kwargs : additional model-specific parameters

        Returns:
        --------
        float : Option price
        """
        pass

    @abstractmethod
    def calculate_greeks(self, S: float, K: float, T: float, r_d: float, r_f: float,
                         sigma: float, option_type: str = 'call', **kwargs) -> Dict[str, float]:
        """
        Calculate option Greeks

        Returns:
        --------
        Dict with keys: delta, gamma, vega, theta, rho
        """
        pass

    def calculate_implied_volatility(self, market_price: float, S: float, K: float,
                                     T: float, r_d: float, r_f: float,
                                     option_type: str = 'call', **kwargs) -> float:
        """
        Calculate implied volatility using Newton-Raphson method

        Parameters:
        -----------
        market_price : float
            Observed market price of the option
        Other parameters same as price_option

        Returns:
        --------
        float : Implied volatility
        """
        # Initial guess based on ATM approximation
        # Brenner and Subrahmanyam (1988) approximation
        initial_guess = market_price / (0.4 * S * np.sqrt(T))
        sigma = max(0.1, min(initial_guess, 2.0))  # Bound initial guess

        tolerance = 1e-6
        max_iterations = 100

        for i in range(max_iterations):
            # Calculate price and vega
            try:
                price = self.price_option(S, K, T, r_d, r_f, sigma, option_type, **kwargs)
                greeks = self.calculate_greeks(S, K, T, r_d, r_f, sigma, option_type, **kwargs)
                vega = greeks.get('vega', 0) * 100  # Convert from 1% to actual

                # Check for convergence
                price_diff = market_price - price

                if abs(price_diff) < tolerance:
                    return sigma

                # Avoid division by zero or very small vega
                if abs(vega) < 1e-10:
                    # Try bisection method as fallback
                    return self._implied_vol_bisection(market_price, S, K, T, r_d, r_f,
                                                       option_type, **kwargs)

                # Newton-Raphson update
                sigma = sigma + price_diff / vega

                # Bound sigma to reasonable range
                sigma = max(0.001, min(sigma, 5.0))

            except Exception as e:
                # If calculation fails, try bisection
                return self._implied_vol_bisection(market_price, S, K, T, r_d, r_f,
                                                   option_type, **kwargs)

        # If didn't converge, return last estimate
        return sigma

    def _implied_vol_bisection(self, market_price: float, S: float, K: float,
                               T: float, r_d: float, r_f: float,
                               option_type: str = 'call', **kwargs) -> float:
        """
        Fallback bisection method for implied volatility
        """
        # Set bounds
        vol_low = 0.001
        vol_high = 5.0
        tolerance = 1e-6
        max_iterations = 100

        for i in range(max_iterations):
            vol_mid = (vol_low + vol_high) / 2

            try:
                price_mid = self.price_option(S, K, T, r_d, r_f, vol_mid, option_type, **kwargs)

                if abs(price_mid - market_price) < tolerance:
                    return vol_mid

                if price_mid < market_price:
                    vol_low = vol_mid
                else:
                    vol_high = vol_mid

                if abs(vol_high - vol_low) < tolerance:
                    return vol_mid

            except:
                return vol_mid

        return (vol_low + vol_high) / 2

    def validate_inputs(self, S: float, K: float, T: float, sigma: float) -> bool:
        """
        Validate input parameters

        Returns:
        --------
        bool : True if inputs are valid

        Raises:
        -------
        ValueError : If inputs are invalid
        """
        if S <= 0:
            raise ValueError(f"Spot price must be positive, got {S}")
        if K <= 0:
            raise ValueError(f"Strike price must be positive, got {K}")
        if T < 0:
            raise ValueError(f"Time to maturity cannot be negative, got {T}")
        if sigma < 0:
            raise ValueError(f"Volatility cannot be negative, got {sigma}")
        return True

    def calculate_forward_price(self, S: float, r_d: float, r_f: float, T: float) -> float:
        """
        Calculate forward price for FX

        Parameters:
        -----------
        S : float
            Spot price
        r_d : float
            Domestic risk-free rate
        r_f : float
            Foreign risk-free rate
        T : float
            Time to maturity

        Returns:
        --------
        float : Forward price
        """
        return S * np.exp((r_d - r_f) * T)

    def calculate_moneyness(self, S: float, K: float, r_d: float, r_f: float, T: float) -> float:
        """
        Calculate moneyness (forward moneyness for FX)

        Returns:
        --------
        float : Moneyness (F/K)
        """
        F = self.calculate_forward_price(S, r_d, r_f, T)
        return F / K if K > 0 else np.inf

    def get_option_type_multiplier(self, option_type: str) -> int:
        """
        Get multiplier for option type

        Returns:
        --------
        int : 1 for call, -1 for put
        """
        if option_type.lower() == 'call':
            return 1
        elif option_type.lower() == 'put':
            return -1
        else:
            raise ValueError(f"Invalid option type: {option_type}. Must be 'call' or 'put'")

    def calculate_intrinsic_value(self, S: float, K: float, option_type: str) -> float:
        """
        Calculate intrinsic value of option

        Returns:
        --------
        float : Intrinsic value
        """
        if option_type.lower() == 'call':
            return max(0, S - K)
        else:
            return max(0, K - S)

    def calculate_time_value(self, option_price: float, S: float, K: float,
                             option_type: str) -> float:
        """
        Calculate time value of option

        Returns:
        --------
        float : Time value (extrinsic value)
        """
        intrinsic = self.calculate_intrinsic_value(S, K, option_type)
        return max(0, option_price - intrinsic)

    def is_american_optimal_exercise(self, S: float, K: float, T: float,
                                     r_d: float, r_f: float, sigma: float,
                                     option_type: str) -> bool:
        """
        Check if early exercise is optimal for American option
        (Simplified check - full implementation would use binomial/finite difference)

        Returns:
        --------
        bool : True if early exercise might be optimal
        """
        # For American calls on non-dividend paying assets, early exercise is never optimal
        # For FX, we need to consider the interest rate differential

        if option_type.lower() == 'call':
            # Early exercise might be optimal if foreign rate is high
            return r_f > r_d
        else:
            # For puts, early exercise might be optimal if domestic rate is high
            return r_d > r_f

    def __str__(self) -> str:
        """String representation"""
        return f"{self.name} Option Pricing Model"

    def __repr__(self) -> str:
        """Object representation"""
        return f"{self.__class__.__name__}(name='{self.name}')"