"""
Feature engineering for FX options
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats


class FeatureEngineer:
    """
    Create features for option trading strategies
    """

    def __init__(self):
        self.features = {}

    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create volatility-based features
        """
        features = pd.DataFrame(index=data.index)

        # Realized volatility
        log_returns = np.log(data['spot'] / data['spot'].shift(1))

        for window in [5, 10, 20, 60]:
            features[f'realized_vol_{window}d'] = log_returns.rolling(window).std() * np.sqrt(252)

        # Parkinson volatility (high-low estimator if available)
        # Yang-Zhang volatility
        # Garman-Klass volatility

        # ATM vol features
        tenors = ['1W', '2W', '1M', '3M', '6M', '1Y']
        for tenor in tenors:
            if f'atm_vol_{tenor}' in data.columns:
                vol_col = f'atm_vol_{tenor}'

                # Level
                features[f'iv_{tenor}'] = data[vol_col]

                # Changes
                features[f'iv_{tenor}_change_1d'] = data[vol_col].diff()
                features[f'iv_{tenor}_change_5d'] = data[vol_col].diff(5)

                # Moving averages
                features[f'iv_{tenor}_ma20'] = data[vol_col].rolling(20).mean()
                features[f'iv_{tenor}_ma60'] = data[vol_col].rolling(60).mean()

                # Z-scores
                ma = data[vol_col].rolling(60).mean()
                std = data[vol_col].rolling(60).std()
                features[f'iv_{tenor}_zscore'] = (data[vol_col] - ma) / std

                # Percentile rank
                features[f'iv_{tenor}_pct_rank'] = data[vol_col].rolling(252).rank(pct=True)

        # Term structure
        if 'atm_vol_1M' in data.columns and 'atm_vol_1Y' in data.columns:
            features['term_structure'] = data['atm_vol_1Y'] - data['atm_vol_1M']
            features['term_structure_ratio'] = data['atm_vol_1Y'] / data['atm_vol_1M']

        # Vol of vol
        if 'atm_vol_1M' in data.columns:
            features['vol_of_vol'] = data['atm_vol_1M'].rolling(20).std()

        # IV/RV ratio
        if 'atm_vol_1M' in data.columns:
            features['ivrv_ratio_1M'] = data['atm_vol_1M'] / features['realized_vol_20d']
            features['vol_premium_1M'] = data['atm_vol_1M'] - features['realized_vol_20d']

        return features

    def create_skew_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create skew and smile features
        """
        features = pd.DataFrame(index=data.index)

        tenors = ['1M', '3M', '6M']

        for tenor in tenors:
            # Risk reversal features
            if f'rr_25_{tenor}' in data.columns:
                rr_col = f'rr_25_{tenor}'

                features[f'rr25_{tenor}'] = data[rr_col]
                features[f'rr25_{tenor}_ma20'] = data[rr_col].rolling(20).mean()
                features[f'rr25_{tenor}_zscore'] = (data[rr_col] - data[rr_col].rolling(60).mean()) / data[
                    rr_col].rolling(60).std()

            if f'rr_10_{tenor}' in data.columns:
                features[f'rr10_{tenor}'] = data[f'rr_10_{tenor}']

            # Butterfly features
            if f'bf_25_{tenor}' in data.columns:
                bf_col = f'bf_25_{tenor}'

                features[f'bf25_{tenor}'] = data[bf_col]
                features[f'bf25_{tenor}_ma20'] = data[bf_col].rolling(20).mean()
                features[f'bf25_{tenor}_zscore'] = (data[bf_col] - data[bf_col].rolling(60).mean()) / data[
                    bf_col].rolling(60).std()

            if f'bf_10_{tenor}' in data.columns:
                features[f'bf10_{tenor}'] = data[f'bf_10_{tenor}']

            # Skew slope
            if f'rr_25_{tenor}' in data.columns and f'rr_10_{tenor}' in data.columns:
                features[f'skew_slope_{tenor}'] = (data[f'rr_10_{tenor}'] - data[
                    f'rr_25_{tenor}']) / 15  # Delta difference

            # Smile curvature
            if f'bf_25_{tenor}' in data.columns and f'bf_10_{tenor}' in data.columns:
                features[f'smile_curve_{tenor}'] = data[f'bf_10_{tenor}'] - data[f'bf_25_{tenor}']

        return features

    def create_carry_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create carry and forward-based features
        """
        features = pd.DataFrame(index=data.index)

        tenors = ['1M', '3M', '6M', '1Y']
        days_map = {'1M': 30, '3M': 90, '6M': 180, '1Y': 365}

        for tenor in tenors:
            if f'fwd_points_{tenor}' in data.columns:
                fwd_col = f'fwd_points_{tenor}'

                # Forward points
                features[f'fwd_pts_{tenor}'] = data[fwd_col]

                # Annualized carry
                features[f'carry_{tenor}'] = (data[fwd_col] / data['spot']) * (365 / days_map[tenor])

                # Carry to vol ratio
                if f'atm_vol_{tenor}' in data.columns:
                    features[f'carry_vol_ratio_{tenor}'] = features[f'carry_{tenor}'] / data[f'atm_vol_{tenor}']

                # Forward premium/discount
                features[f'fwd_premium_{tenor}'] = data[fwd_col] / data['spot']

        # Carry curve slope
        if 'carry_1M' in features.columns and 'carry_1Y' in features.columns:
            features['carry_slope'] = features['carry_1Y'] - features['carry_1M']

        return features

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create technical indicators
        """
        features = pd.DataFrame(index=data.index)

        # Price features
        features['spot'] = data['spot']

        # Returns
        features['return_1d'] = data['spot'].pct_change()
        features['return_5d'] = data['spot'].pct_change(5)
        features['return_20d'] = data['spot'].pct_change(20)

        # Log returns
        features['log_return_1d'] = np.log(data['spot'] / data['spot'].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            features[f'ma_{window}'] = data['spot'].rolling(window).mean()
            features[f'ma_ratio_{window}'] = data['spot'] / features[f'ma_{window}']

        # Bollinger bands
        ma20 = data['spot'].rolling(20).mean()
        std20 = data['spot'].rolling(20).std()
        features['bb_upper'] = ma20 + 2 * std20
        features['bb_lower'] = ma20 - 2 * std20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / ma20
        features['bb_position'] = (data['spot'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])

        # RSI
        delta = data['spot'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['spot'].ewm(span=12, adjust=False).mean()
        exp2 = data['spot'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']

        # ATR (if high/low available)
        features['atr'] = data['spot'].rolling(14).apply(lambda x: np.std(np.diff(x)))

        return features

    def create_regime_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create market regime features
        """
        features = pd.DataFrame(index=data.index)

        # Volatility regime
        if 'atm_vol_1M' in data.columns:
            vol = data['atm_vol_1M']
            vol_ma = vol.rolling(60).mean()
            vol_std = vol.rolling(60).std()

            # Vol regime (low/normal/high)
            features['vol_regime'] = pd.cut(vol, bins=[0, vol_ma - vol_std, vol_ma + vol_std, np.inf],
                                            labels=[0, 1, 2])

            # Vol trend
            features['vol_trend'] = np.sign(vol - vol.shift(20))

        # Price trend regime
        ma50 = data['spot'].rolling(50).mean()
        ma200 = data['spot'].rolling(200).mean()

        features['trend_regime'] = np.where(ma50 > ma200, 1, -1)
        features['trend_strength'] = (ma50 - ma200) / ma200

        # Momentum regime
        returns_20d = data['spot'].pct_change(20)
        features['momentum_regime'] = np.sign(returns_20d)

        # Risk regime (based on cross-asset correlations if available)
        # This would require additional data like equity indices, bonds, etc.

        return features

    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features
        """
        all_features = pd.DataFrame(index=data.index)

        # Add all feature sets
        vol_features = self.create_volatility_features(data)
        skew_features = self.create_skew_features(data)
        carry_features = self.create_carry_features(data)
        tech_features = self.create_technical_features(data)
        regime_features = self.create_regime_features(data)

        # Combine
        for features_df in [vol_features, skew_features, carry_features, tech_features, regime_features]:
            all_features = pd.concat([all_features, features_df], axis=1)

        # Remove duplicates
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Add interaction features
        if 'carry_vol_ratio_1M' in all_features.columns and 'rr25_1M' in all_features.columns:
            all_features['carry_skew_interact'] = all_features['carry_vol_ratio_1M'] * all_features['rr25_1M']

        if 'vol_premium_1M' in all_features.columns and 'momentum_regime' in all_features.columns:
            all_features['vol_momentum_interact'] = all_features['vol_premium_1M'] * all_features['momentum_regime']

        return all_features