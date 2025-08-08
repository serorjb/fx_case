"""
===============================================================================
ml_model.py - Machine Learning model for option pricing
===============================================================================
"""
from typing import Dict, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from config import Config
from models import BaseModel


class MLPricer(BaseModel):
    """LightGBM-based option pricer"""

    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = config.ML_FEATURES
        self.last_train_date = None
        self.training_data = []

    def train(self, train_data: Dict[str, pd.DataFrame], val_dates: pd.DatetimeIndex,
              other_models: Dict):
        """Train ML model on historical data"""

        print("  Preparing ML training data...")

        # Prepare features and targets
        X_train = []
        y_train = []

        for pair, pair_data in train_data.items():
            # Skip if not enough data
            if len(pair_data) < 100:
                continue

            # Generate features for each date
            for i in range(20, len(pair_data) - 1):
                features = self._extract_features(
                    pair_data.iloc[i],
                    pair_data.iloc[max(0, i-20):i],
                    pair,
                    other_models
                )

                if features is not None:
                    X_train.append(features)

                    # Target: next day's return
                    next_return = (pair_data.iloc[i+1]['spot'] / pair_data.iloc[i]['spot']) - 1
                    y_train.append(next_return)

        if len(X_train) == 0:
            print("  Warning: No training data available for ML model")
            return

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Scale features
        X_train = self.scaler.fit_transform(X_train)

        # Train LightGBM model
        print(f"  Training on {len(X_train)} samples...")

        train_data = lgb.Dataset(X_train, label=y_train)

        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'verbose': -1
        }

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100
        )

        print("  ML model trained successfully")

    def price(self, spot: float, strike: float, T: float, r_d: float, r_f: float,
             sigma: float, is_call: bool = True, **kwargs) -> float:
        """Price option using ML model"""

        if self.model is None:
            # Fallback to Black-Scholes
            from models import GarmanKohlhagen
            gk = GarmanKohlhagen()
            return gk.price(spot, strike, T, r_d, r_f, sigma, is_call)

        # Extract features
        features = {
            'moneyness': strike / spot,
            'time_to_maturity': T,
            'implied_vol': sigma,
            'rate_differential': r_d - r_f
        }

        # Add additional features from kwargs
        for key in ['realized_vol', 'vol_premium', 'risk_reversal', 'butterfly']:
            if key in kwargs:
                features[key] = kwargs[key]

        # Create feature vector
        feature_vector = []
        for fname in self.feature_names:
            if fname in features:
                feature_vector.append(features[fname])
            else:
                feature_vector.append(0)  # Default value

        # Scale and predict
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector = self.scaler.transform(feature_vector)

        # Get prediction
        prediction = self.model.predict(feature_vector)[0]

        # Convert prediction to price
        # Simplified: use BS price with adjusted vol
        vol_adjustment = 1 + prediction * 10  # Scale prediction to vol adjustment
        adjusted_vol = sigma * vol_adjustment

        from models import GarmanKohlhagen
        gk = GarmanKohlhagen()
        return gk.price(spot, strike, T, r_d, r_f, adjusted_vol, is_call)

    def _extract_features(self, current: pd.Series, historical: pd.DataFrame,
                         pair: str, other_models: Dict) -> Optional[np.ndarray]:
        """Extract features for ML model"""

        try:
            features = []

            # Basic features
            features.append(1.0)  # Moneyness (ATM for now)
            features.append(0.25)  # Time to maturity (3 months)

            # Price features
            if 'spot' in current:
                spot = current['spot']
                returns = np.log(historical['spot'] / historical['spot'].shift(1))
                features.append(returns.mean())  # Mean return
                features.append(returns.std() * np.sqrt(252))  # Realized vol
            else:
                features.extend([0, 0])

            # Volatility features
            if 'atm_vol_3M' in current:
                features.append(current['atm_vol_3M'])
            else:
                features.append(0.1)

            # Vol premium
            if 'realized_vol_20d' in current and 'atm_vol_3M' in current:
                features.append(current['atm_vol_3M'] - current['realized_vol_20d'])
            else:
                features.append(0)

            # Smile features
            if 'rr_25_3M' in current:
                features.append(current['rr_25_3M'])
            else:
                features.append(0)

            if 'bf_25_3M' in current:
                features.append(current['bf_25_3M'])
            else:
                features.append(0)

            # Term structure
            if 'atm_vol_1M' in current and 'atm_vol_1Y' in current:
                features.append(current['atm_vol_1Y'] - current['atm_vol_1M'])
            else:
                features.append(0)

            # Rates (simplified)
            features.append(0.03)  # Rate differential

            # Prices from other models (placeholder)
            features.extend([0, 0, 0])  # GK, GVV, SABR prices

            return np.array(features)

        except Exception as e:
            return None
