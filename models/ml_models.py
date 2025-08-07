"""
Machine Learning models for FX options pricing and trading
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from typing import Dict, Tuple, Optional
import joblib


class LightGBMModel:
    """
    LightGBM model for option pricing and signal generation
    """

    def __init__(self, model_type: str = 'pricing'):
        """
        Initialize LightGBM model

        Parameters:
        -----------
        model_type : str
            'pricing' for option pricing, 'signal' for trading signals
        """
        self.name = "LightGBM"
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None

    def prepare_features(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Prepare features for ML model
        """
        features = pd.DataFrame(index=data.index)

        # Price features
        features['spot'] = data['spot']
        features['spot_return'] = data['spot'].pct_change()
        features['spot_log_return'] = np.log(data['spot'] / data['spot'].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50]:
            features[f'spot_ma_{window}'] = data['spot'].rolling(window).mean()
            features[f'spot_ma_ratio_{window}'] = data['spot'] / features[f'spot_ma_{window}']

        # Volatility features
        for tenor in ['1M', '3M', '6M', '1Y']:
            if f'atm_vol_{tenor}' in data.columns:
                vol_col = f'atm_vol_{tenor}'
                features[f'vol_{tenor}'] = data[vol_col]
                features[f'vol_{tenor}_ma_20'] = data[vol_col].rolling(20).mean()
                features[f'vol_{tenor}_zscore'] = (data[vol_col] - features[f'vol_{tenor}_ma_20']) / data[
                    vol_col].rolling(20).std()

        # Term structure features
        if 'atm_vol_1M' in data.columns and 'atm_vol_1Y' in data.columns:
            features['vol_term_structure'] = data['atm_vol_1Y'] - data['atm_vol_1M']

        # Skew features (Risk Reversals)
        for tenor in ['1M', '3M', '6M']:
            if f'rr_25_{tenor}' in data.columns:
                features[f'rr_25_{tenor}'] = data[f'rr_25_{tenor}']
                features[f'rr_25_{tenor}_ma'] = data[f'rr_25_{tenor}'].rolling(20).mean()

        # Butterfly features
        for tenor in ['1M', '3M', '6M']:
            if f'bf_25_{tenor}' in data.columns:
                features[f'bf_25_{tenor}'] = data[f'bf_25_{tenor}']
                features[f'bf_25_{tenor}_ma'] = data[f'bf_25_{tenor}'].rolling(20).mean()

        # Forward points features
        for tenor in ['1M', '3M', '6M']:
            if f'fwd_points_{tenor}' in data.columns:
                features[f'fwd_points_{tenor}'] = data[f'fwd_points_{tenor}']
                features[f'carry_{tenor}'] = data[f'fwd_points_{tenor}'] / data['spot']

        # Realized volatility
        for window in [5, 10, 20]:
            features[f'realized_vol_{window}'] = data['spot_log_return'].rolling(window).std() * np.sqrt(252)

        # Vol risk premium
        if 'atm_vol_1M' in data.columns:
            features['vol_premium_1M'] = data['atm_vol_1M'] - features['realized_vol_20']

        # Time features
        features['day_of_week'] = data.index.dayofweek
        features['month'] = data.index.month
        features['quarter'] = data.index.quarter

        return features

    def train_pricing_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Train model for option pricing
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # LightGBM parameters
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4
        }

        # Create datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        # Store feature importance
        importance = self.model.feature_importance(importance_type='gain')
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

    def train_signal_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series) -> None:
        """
        Train model for trading signal generation
        """
        # Convert to classification problem (1: buy, 0: hold, -1: sell)
        y_train_class = pd.cut(y_train, bins=[-np.inf, -0.001, 0.001, np.inf],
                               labels=[-1, 0, 1])
        y_val_class = pd.cut(y_val, bins=[-np.inf, -0.001, 0.001, np.inf],
                             labels=[-1, 0, 1])

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # LightGBM parameters for classification
        params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'num_threads': 4
        }

        # Create datasets
        train_data = lgb.Dataset(X_train_scaled, label=y_train_class.codes)
        val_data = lgb.Dataset(X_val_scaled, label=y_val_class.codes, reference=train_data)

        # Train model
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        X_scaled = self.scaler.transform(X)

        if self.model_type == 'pricing':
            return self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
        else:
            # For signal model, return class probabilities
            probs = self.model.predict(X_scaled, num_iteration=self.model.best_iteration)
            return probs

    def save_model(self, path: str) -> None:
        """Save model to disk"""
        self.model.save_model(f"{path}_model.txt")
        joblib.dump(self.scaler, f"{path}_scaler.pkl")
        if self.feature_importance is not None:
            self.feature_importance.to_csv(f"{path}_importance.csv", index=False)

    def load_model(self, path: str) -> None:
        """Load model from disk"""
        self.model = lgb.Booster(model_file=f"{path}_model.txt")
        self.scaler = joblib.load(f"{path}_scaler.pkl")