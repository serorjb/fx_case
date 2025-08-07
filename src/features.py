"""
Enhanced feature engineering for FX options with HMM regime detection
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

class HMMRegimeDetector:
    """
    Hidden Markov Model for volatility regime detection
    Ensures consistent regime labeling (0=low vol, 1=high vol)
    """

    def __init__(self, n_states: int = 2, window: int = 252):
        self.n_states = n_states
        self.window = window
        self.models = {}  # Store model per currency pair
        self.regime_history = {}  # Store historical regimes

    def fit_predict(self, returns: pd.Series, pair: str) -> pd.Series:
        """
        Fit HMM and predict regimes with consistent labeling
        Uses only past data to avoid look-ahead bias
        """
        regimes = pd.Series(index=returns.index, dtype=float)

        # Need minimum data to fit
        min_data_points = max(100, self.window)

        for i in range(len(returns)):
            if i < min_data_points:
                regimes.iloc[i] = np.nan
                continue

            # Use only past data (no look-ahead)
            start_idx = max(0, i - self.window)
            historical_returns = returns.iloc[start_idx:i+1].values.reshape(-1, 1)

            if len(historical_returns) < min_data_points:
                regimes.iloc[i] = np.nan
                continue

            # Initialize or reuse model
            if pair not in self.models:
                model = hmm.GaussianHMM(
                    n_components=self.n_states,
                    covariance_type="full",
                    n_iter=1,  # todo: reset to 200 for actual fitting
                    random_state=42,
                    tol=5e-2,
                    init_params=''  # disables automatic initialization,
                )
            else:
                model = self.models[pair]

            try:
                # Fit model on historical data only
                model.fit(historical_returns)

                # Predict current state
                current_state = model.predict(historical_returns[-1:].reshape(-1, 1))[0]

                # Ensure consistent labeling: 0=low vol, 1=high vol
                # Calculate volatility for each state
                state_vols = []
                for state in range(self.n_states):
                    state_mask = model.predict(historical_returns) == state
                    if state_mask.sum() > 0:
                        state_vol = np.std(historical_returns[state_mask])
                        state_vols.append((state, state_vol))

                # Sort by volatility
                state_vols.sort(key=lambda x: x[1])

                # Create mapping (lowest vol = 0, highest vol = n_states-1)
                state_mapping = {old_state: new_state
                               for new_state, (old_state, _) in enumerate(state_vols)}

                # Map current state
                regimes.iloc[i] = state_mapping.get(current_state, current_state)

                # Store model for reuse
                self.models[pair] = model

            except Exception as e:
                # If fitting fails, use previous regime or 0
                if i > 0 and not pd.isna(regimes.iloc[i-1]):
                    regimes.iloc[i] = regimes.iloc[i-1]
                else:
                    regimes.iloc[i] = 0

        # Store regime history
        self.regime_history[pair] = regimes

        return regimes

    def get_regime_probabilities(self, returns: pd.Series, pair: str) -> pd.DataFrame:
        """
        Get probability of being in each regime
        """
        if pair not in self.models:
            return pd.DataFrame(index=returns.index)

        model = self.models[pair]
        probs = pd.DataFrame(index=returns.index, columns=[f'regime_{i}_prob' for i in range(self.n_states)])

        min_data_points = max(100, self.window)

        for i in range(len(returns)):
            if i < min_data_points:
                continue

            start_idx = max(0, i - self.window)
            historical_returns = returns.iloc[start_idx:i+1].values.reshape(-1, 1)

            try:
                # Get posterior probabilities
                _, posteriors = model.score_samples(historical_returns[-1:].reshape(-1, 1))

                for state in range(self.n_states):
                    probs.iloc[i, state] = posteriors[-1, state]
            except:
                pass

        return probs

class FeatureEngineer:
    """
    Enhanced feature engineering with model outputs and regime detection
    """

    def __init__(self):
        self.features = {}
        self.hmm_detector = HMMRegimeDetector(n_states=2)
        self.model_outputs = {}

    def create_volatility_features(self, data: pd.DataFrame, lookback_only: bool = True) -> pd.DataFrame:
        """
        Create volatility-based features
        lookback_only: If True, only use past data (no look-ahead bias)
        """
        features = pd.DataFrame(index=data.index)

        # Realized volatility - using only past data
        log_returns = np.log(data['spot'] / data['spot'].shift(1))

        for window in [5, 10, 20, 60]:
            if lookback_only:
                # Use .shift(1) to ensure we only use past returns
                features[f'realized_vol_{window}d'] = log_returns.shift(1).rolling(window).std() * np.sqrt(252)
            else:
                features[f'realized_vol_{window}d'] = log_returns.rolling(window).std() * np.sqrt(252)

        # ATM vol features
        tenors = ['1W', '2W', '1M', '3M', '6M', '1Y']
        for tenor in tenors:
            if f'atm_vol_{tenor}' in data.columns:
                vol_col = f'atm_vol_{tenor}'

                # Level
                features[f'iv_{tenor}'] = data[vol_col]

                # Changes - using shift to avoid look-ahead
                features[f'iv_{tenor}_change_1d'] = data[vol_col] - data[vol_col].shift(1)
                features[f'iv_{tenor}_change_5d'] = data[vol_col] - data[vol_col].shift(5)

                # Moving averages - use shift to avoid including current value
                if lookback_only:
                    features[f'iv_{tenor}_ma20'] = data[vol_col].shift(1).rolling(20).mean()
                    features[f'iv_{tenor}_ma60'] = data[vol_col].shift(1).rolling(60).mean()
                else:
                    features[f'iv_{tenor}_ma20'] = data[vol_col].rolling(20).mean()
                    features[f'iv_{tenor}_ma60'] = data[vol_col].rolling(60).mean()

                # Z-scores - using past data only
                if lookback_only:
                    ma = data[vol_col].shift(1).rolling(60).mean()
                    std = data[vol_col].shift(1).rolling(60).std()
                else:
                    ma = data[vol_col].rolling(60).mean()
                    std = data[vol_col].rolling(60).std()
                features[f'iv_{tenor}_zscore'] = (data[vol_col] - ma) / std.replace(0, np.nan)

                # Percentile rank - using expanding window up to current point
                if lookback_only:
                    features[f'iv_{tenor}_pct_rank'] = data[vol_col].shift(1).expanding().rank(pct=True)
                else:
                    features[f'iv_{tenor}_pct_rank'] = data[vol_col].rolling(252).rank(pct=True)

        # Term structure
        if 'atm_vol_1M' in data.columns and 'atm_vol_1Y' in data.columns:
            features['term_structure'] = data['atm_vol_1Y'] - data['atm_vol_1M']
            features['term_structure_ratio'] = data['atm_vol_1Y'] / data['atm_vol_1M'].replace(0, np.nan)

        # Vol of vol - using past data
        if 'atm_vol_1M' in data.columns:
            if lookback_only:
                features['vol_of_vol'] = data['atm_vol_1M'].shift(1).rolling(20).std()
            else:
                features['vol_of_vol'] = data['atm_vol_1M'].rolling(20).std()

        # IV/RV ratio
        if 'atm_vol_1M' in data.columns and 'realized_vol_20d' in features.columns:
            features['ivrv_ratio_1M'] = data['atm_vol_1M'] / features['realized_vol_20d'].replace(0, np.nan)
            features['vol_premium_1M'] = data['atm_vol_1M'] - features['realized_vol_20d']

        return features

    def create_hmm_regime_features(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Create HMM-based regime features
        """
        features = pd.DataFrame(index=data.index)

        # Calculate returns
        returns = data['spot'].pct_change()

        # Detect regimes (uses only past data internally)
        regimes = self.hmm_detector.fit_predict(returns, pair)
        features[f'hmm_regime_{pair}'] = regimes

        # Get regime probabilities
        regime_probs = self.hmm_detector.get_regime_probabilities(returns, pair)
        features = pd.concat([features, regime_probs], axis=1)

        # Regime transition features
        features[f'regime_change_{pair}'] = regimes != regimes.shift(1)
        features[f'days_in_regime_{pair}'] = regimes.groupby((regimes != regimes.shift()).cumsum()).cumcount()

        # Regime statistics (using expanding window to avoid look-ahead)
        for regime in [0, 1]:
            mask = regimes == regime
            if mask.any():
                # Calculate statistics only on past occurrences
                expanding_mask = mask.expanding().apply(lambda x: x[-1] if len(x) > 0 else False)
                features[f'regime_{regime}_frequency'] = mask.expanding().mean()

        return features

    def create_model_output_features(self, data: pd.DataFrame, models: Dict) -> pd.DataFrame:
        """
        Create features from pricing model outputs
        Ensures no look-ahead bias by using current market data only
        """
        features = pd.DataFrame(index=data.index)

        # Skip if no models provided
        if not models:
            return features

        # Common parameters (these would come from market data)
        spot_col = 'spot'

        for date_idx in range(len(data)):
            if date_idx < 20:  # Need some history
                continue

            current_date = data.index[date_idx]
            current_data = data.iloc[date_idx]

            # Only use data up to current date
            historical_data = data.iloc[:date_idx+1]

            # Calculate parameters from historical data
            S = current_data[spot_col]

            # Use historical volatility as proxy (no future information)
            returns = np.log(historical_data[spot_col] / historical_data[spot_col].shift(1))
            hist_vol = returns.iloc[-20:].std() * np.sqrt(252)

            # For each model and tenor
            for tenor in ['1M', '3M', '6M']:
                if f'atm_vol_{tenor}' not in data.columns:
                    continue

                # Current market vol
                market_vol = current_data[f'atm_vol_{tenor}']

                if pd.isna(market_vol):
                    continue

                # Model prices (simplified - in practice would use actual rates)
                K = S  # ATM
                T = {'1M': 1/12, '3M': 0.25, '6M': 0.5}[tenor]
                r_d = 0.05  # Simplified - should use discount curves
                r_f = 0.01  # Simplified

                # Get model prices if available
                if 'gk' in models:
                    try:
                        gk_price = models['gk'].price_option(S, K, T, r_d, r_f, market_vol, 'call')
                        features.loc[current_date, f'gk_price_{tenor}'] = gk_price / S  # Normalize

                        # Greeks from model
                        greeks = models['gk'].calculate_greeks(S, K, T, r_d, r_f, market_vol, 'call')
                        features.loc[current_date, f'gk_delta_{tenor}'] = greeks.get('delta', 0)
                        features.loc[current_date, f'gk_gamma_{tenor}'] = greeks.get('gamma', 0)
                        features.loc[current_date, f'gk_vega_{tenor}'] = greeks.get('vega', 0)
                    except:
                        pass

                if 'gvv' in models and f'rr_25_{tenor}' in data.columns:
                    try:
                        rr = current_data[f'rr_25_{tenor}']
                        bf = current_data[f'bf_25_{tenor}'] if f'bf_25_{tenor}' in data.columns else 0

                        gvv_price = models['gvv'].price_option(
                            S, K, T, r_d, r_f, market_vol, 'call',
                            rr_25=rr, bf_25=bf
                        )
                        features.loc[current_date, f'gvv_price_{tenor}'] = gvv_price / S

                        # Price difference as signal
                        if f'gk_price_{tenor}' in features.columns:
                            features.loc[current_date, f'gvv_gk_diff_{tenor}'] = (
                                features.loc[current_date, f'gvv_price_{tenor}'] -
                                features.loc[current_date, f'gk_price_{tenor}']
                            )
                    except:
                        pass

        return features

    def create_all_features(self, data: pd.DataFrame, pair: str, models: Dict = None) -> pd.DataFrame:
        """
        Create all features with look-ahead bias prevention
        """
        all_features = pd.DataFrame(index=data.index)

        # Add all feature sets (all with lookback_only=True)
        print(f"Creating volatility features for {pair}...")
        vol_features = self.create_volatility_features(data, lookback_only=True)
        print(f"Creating skew features for {pair}...")
        skew_features = self.create_skew_features(data, lookback_only=True)
        print(f"Creating carry features for {pair}...")
        carry_features = self.create_carry_features(data, lookback_only=True)
        print(f"Creating technical features for {pair}...")
        tech_features = self.create_technical_features(data, lookback_only=True)
        print(f"Creating regime features for {pair}...")
        regime_features = self.create_regime_features(data, lookback_only=True)

        # Add HMM regime features
        hmm_features = self.create_hmm_regime_features(data, pair)

        # Add model output features
        if models:
            model_features = self.create_model_output_features(data, models)
            all_features = pd.concat([all_features, model_features], axis=1)

        # Combine all features
        for features_df in [vol_features, skew_features, carry_features, tech_features,
                           regime_features, hmm_features]:
            all_features = pd.concat([all_features, features_df], axis=1)

        # Remove duplicates
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]

        # Add interaction features (using current and past data only)
        if 'carry_vol_ratio_1M' in all_features.columns and 'rr25_1M' in all_features.columns:
            all_features['carry_skew_interact'] = all_features['carry_vol_ratio_1M'] * all_features['rr25_1M']

        if 'vol_premium_1M' in all_features.columns and f'hmm_regime_{pair}' in all_features.columns:
            all_features['vol_regime_interact'] = all_features['vol_premium_1M'] * all_features[f'hmm_regime_{pair}']

        return all_features

    def create_skew_features(self, data: pd.DataFrame, lookback_only: bool = True) -> pd.DataFrame:
        """
        Create skew and smile features without look-ahead bias
        """
        features = pd.DataFrame(index=data.index)

        tenors = ['1M', '3M', '6M']

        for tenor in tenors:
            # Risk reversal features
            if f'rr_25_{tenor}' in data.columns:
                rr_col = f'rr_25_{tenor}'

                features[f'rr25_{tenor}'] = data[rr_col]

                if lookback_only:
                    features[f'rr25_{tenor}_ma20'] = data[rr_col].shift(1).rolling(20).mean()
                    ma = data[rr_col].shift(1).rolling(60).mean()
                    std = data[rr_col].shift(1).rolling(60).std()
                else:
                    features[f'rr25_{tenor}_ma20'] = data[rr_col].rolling(20).mean()
                    ma = data[rr_col].rolling(60).mean()
                    std = data[rr_col].rolling(60).std()

                features[f'rr25_{tenor}_zscore'] = (data[rr_col] - ma) / std.replace(0, np.nan)

            if f'rr_10_{tenor}' in data.columns:
                features[f'rr10_{tenor}'] = data[f'rr_10_{tenor}']

            # Butterfly features
            if f'bf_25_{tenor}' in data.columns:
                bf_col = f'bf_25_{tenor}'

                features[f'bf25_{tenor}'] = data[bf_col]

                if lookback_only:
                    features[f'bf25_{tenor}_ma20'] = data[bf_col].shift(1).rolling(20).mean()
                    ma = data[bf_col].shift(1).rolling(60).mean()
                    std = data[bf_col].shift(1).rolling(60).std()
                else:
                    features[f'bf25_{tenor}_ma20'] = data[bf_col].rolling(20).mean()
                    ma = data[bf_col].rolling(60).mean()
                    std = data[bf_col].rolling(60).std()

                features[f'bf25_{tenor}_zscore'] = (data[bf_col] - ma) / std.replace(0, np.nan)

            if f'bf_10_{tenor}' in data.columns:
                features[f'bf10_{tenor}'] = data[f'bf_10_{tenor}']

            # Skew slope
            if f'rr_25_{tenor}' in data.columns and f'rr_10_{tenor}' in data.columns:
                features[f'skew_slope_{tenor}'] = (data[f'rr_10_{tenor}'] - data[f'rr_25_{tenor}']) / 15

            # Smile curvature
            if f'bf_25_{tenor}' in data.columns and f'bf_10_{tenor}' in data.columns:
                features[f'smile_curve_{tenor}'] = data[f'bf_10_{tenor}'] - data[f'bf_25_{tenor}']

        return features

    def create_carry_features(self, data: pd.DataFrame, lookback_only: bool = True) -> pd.DataFrame:
        """
        Create carry and forward-based features without look-ahead bias
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
                    features[f'carry_vol_ratio_{tenor}'] = features[f'carry_{tenor}'] / data[f'atm_vol_{tenor}'].replace(0, np.nan)

                # Forward premium/discount
                features[f'fwd_premium_{tenor}'] = data[fwd_col] / data['spot']

        # Carry curve slope
        if 'carry_1M' in features.columns and 'carry_1Y' in features.columns:
            features['carry_slope'] = features['carry_1Y'] - features['carry_1M']

        return features

    def create_technical_features(self, data: pd.DataFrame, lookback_only: bool = True) -> pd.DataFrame:
        """
        Create technical indicators without look-ahead bias
        """
        features = pd.DataFrame(index=data.index)

        # Price features
        features['spot'] = data['spot']

        # Returns - always use past data
        features['return_1d'] = data['spot'].pct_change()
        features['return_5d'] = data['spot'].pct_change(5)
        features['return_20d'] = data['spot'].pct_change(20)

        # Log returns
        features['log_return_1d'] = np.log(data['spot'] / data['spot'].shift(1))

        # Moving averages
        for window in [5, 10, 20, 50, 100, 200]:
            if lookback_only:
                # Don't include current price in MA
                features[f'ma_{window}'] = data['spot'].shift(1).rolling(window).mean()
            else:
                features[f'ma_{window}'] = data['spot'].rolling(window).mean()

            features[f'ma_ratio_{window}'] = data['spot'] / features[f'ma_{window}']

        # Bollinger bands
        if lookback_only:
            ma20 = data['spot'].shift(1).rolling(20).mean()
            std20 = data['spot'].shift(1).rolling(20).std()
        else:
            ma20 = data['spot'].rolling(20).mean()
            std20 = data['spot'].rolling(20).std()

        features['bb_upper'] = ma20 + 2 * std20
        features['bb_lower'] = ma20 - 2 * std20
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / ma20.replace(0, np.nan)
        features['bb_position'] = (data['spot'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower']).replace(0, np.nan)

        # RSI
        delta = data['spot'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = data['spot'].ewm(span=12, adjust=False).mean()
        exp2 = data['spot'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        features['macd_diff'] = features['macd'] - features['macd_signal']

        # ATR
        if lookback_only:
            features['atr'] = data['spot'].shift(1).rolling(14).apply(lambda x: np.std(np.diff(x)))
        else:
            features['atr'] = data['spot'].rolling(14).apply(lambda x: np.std(np.diff(x)))

        return features

    def create_regime_features(self, data: pd.DataFrame, lookback_only: bool = True) -> pd.DataFrame:
        """
        Create market regime features without look-ahead bias
        """
        features = pd.DataFrame(index=data.index)

        # Volatility regime
        if 'atm_vol_1M' in data.columns:
            vol = data['atm_vol_1M']

            if lookback_only:
                # Use expanding window to avoid look-ahead
                vol_ma = vol.expanding().mean()
                vol_std = vol.expanding().std()
            else:
                vol_ma = vol.rolling(60).mean()
                vol_std = vol.rolling(60).std()

            # Vol regime (low/normal/high) - using only past data
            lower_bound = vol_ma - vol_std
            upper_bound = vol_ma + vol_std

            features['vol_regime'] = 1  # Default to normal
            features.loc[vol < lower_bound, 'vol_regime'] = 0  # Low
            features.loc[vol > upper_bound, 'vol_regime'] = 2  # High

            # Vol trend
            features['vol_trend'] = np.sign(vol - vol.shift(20))

        # Price trend regime
        if lookback_only:
            ma50 = data['spot'].shift(1).rolling(50).mean()
            ma200 = data['spot'].shift(1).rolling(200).mean()
        else:
            ma50 = data['spot'].rolling(50).mean()
            ma200 = data['spot'].rolling(200).mean()

        features['trend_regime'] = np.where(ma50 > ma200, 1, -1)
        features['trend_strength'] = (ma50 - ma200) / ma200.replace(0, np.nan)

        # Momentum regime
        returns_20d = data['spot'].pct_change(20)
        features['momentum_regime'] = np.sign(returns_20d)

        return features