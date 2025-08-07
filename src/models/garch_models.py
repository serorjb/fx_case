# src/models/garch_models.py
"""
GARCH volatility forecasting models for FX options
Includes univariate GARCH, EGARCH, GJR-GARCH and multivariate DCC-GARCH
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from arch import arch_model
from arch.univariate import GARCH, EGARCH, HARCH, ConstantMean, ZeroMean
import warnings

warnings.filterwarnings('ignore')


class GARCHForecaster:
    """
    Univariate and Multivariate GARCH models for volatility forecasting
    """

    def __init__(self, model_type: str = 'GARCH', p: int = 1, q: int = 1):
        """
        Initialize GARCH forecaster

        Parameters:
        -----------
        model_type : str
            'GARCH', 'EGARCH', 'GJR-GARCH', 'HARCH'
        p : int
            GARCH lag order
        q : int
            ARCH lag order
        """
        self.model_type = model_type
        self.p = p
        self.q = q
        self.models = {}
        self.forecasts = {}
        self.model_params = {}

    def prepare_returns(self, prices: pd.Series, method: str = 'log') -> pd.Series:
        """
        Prepare returns for GARCH modeling
        """
        if method == 'log':
            returns = np.log(prices / prices.shift(1))
        else:
            returns = prices.pct_change()

        # Scale to percentage for better convergence
        returns = returns * 100
        returns = returns.dropna()

        return returns

    def fit_univariate_garch(self, returns: pd.Series, pair: str,
                             refit_window: int = 252) -> Dict:
        """
        Fit univariate GARCH model without look-ahead bias
        """
        results = {
            'fitted_values': pd.Series(index=returns.index, dtype=float),
            'forecasts': pd.Series(index=returns.index, dtype=float),
            'standardized_residuals': pd.Series(index=returns.index, dtype=float)
        }

        # Minimum data required
        min_obs = max(100, refit_window)

        for i in range(len(returns)):
            if i < min_obs:
                results['fitted_values'].iloc[i] = np.nan
                results['forecasts'].iloc[i] = np.nan
                continue

            # Use only data up to current point (no look-ahead)
            train_data = returns.iloc[:i]

            # Refit model periodically or use cached model
            if i % refit_window == 0 or pair not in self.models:
                try:
                    # Create model based on type
                    if self.model_type == 'GARCH':
                        model = arch_model(
                            train_data,
                            mean='Constant',
                            vol='GARCH',
                            p=self.p,
                            q=self.q,
                            rescale=True
                        )
                    elif self.model_type == 'EGARCH':
                        model = arch_model(
                            train_data,
                            mean='Constant',
                            vol='EGARCH',
                            p=self.p,
                            o=1,  # Asymmetric term
                            q=self.q
                        )
                    elif self.model_type == 'GJR-GARCH':
                        model = arch_model(
                            train_data,
                            mean='Constant',
                            vol='GARCH',
                            p=self.p,
                            o=1,  # Threshold term
                            q=self.q
                        )
                    else:  # HARCH
                        model = arch_model(
                            train_data,
                            mean='Constant',
                            vol='HARCH',
                            lags=[1, 5, 22]  # Daily, weekly, monthly
                        )

                    # Fit model
                    fitted = model.fit(disp='off', show_warning=False)

                    # Store model and parameters
                    self.models[pair] = fitted
                    self.model_params[pair] = {
                        'omega': fitted.params.get('omega', 0),
                        'alpha': fitted.params.get('alpha[1]', 0),
                        'beta': fitted.params.get('beta[1]', 0),
                        'gamma': fitted.params.get('gamma[1]', 0)  # For GJR
                    }

                except Exception as e:
                    # If fitting fails, use simple volatility
                    vol = train_data.iloc[-20:].std()
                    results['fitted_values'].iloc[i] = vol
                    results['forecasts'].iloc[i] = vol
                    continue

            # Get fitted model
            if pair in self.models:
                fitted_model = self.models[pair]

                try:
                    # One-step ahead forecast
                    forecast = fitted_model.forecast(horizon=1, reindex=False)

                    # Store forecast (annualized)
                    vol_forecast = np.sqrt(forecast.variance.values[-1, 0]) * np.sqrt(252) / 100
                    results['forecasts'].iloc[i] = vol_forecast

                    # Fitted value for current observation
                    if i > 0:
                        results['fitted_values'].iloc[i - 1] = vol_forecast

                    # Standardized residuals
                    if hasattr(fitted_model, 'std_resid'):
                        if len(fitted_model.std_resid) > 0:
                            results['standardized_residuals'].iloc[i - 1] = fitted_model.std_resid.iloc[-1]

                except Exception:
                    # Fallback to historical vol
                    vol = train_data.iloc[-20:].std() * np.sqrt(252) / 100
                    results['forecasts'].iloc[i] = vol

        return results

    def fit_multivariate_garch(self, returns_dict: Dict[str, pd.Series],
                               window: int = 252) -> Dict:
        """
        Fit DCC-GARCH model for multiple currency pairs
        """
        from arch.unitroot import ADF
        from statsmodels.stats.diagnostic import acorr_ljungbox

        # Align all returns series
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()

        if len(returns_df) < window:
            return {}

        results = {
            'correlations': {},
            'forecasts': {},
            'covariance_matrix': []
        }

        # Fit univariate GARCH models first
        univariate_vols = {}
        for col in returns_df.columns:
            garch_results = self.fit_univariate_garch(returns_df[col], col, window)
            univariate_vols[col] = garch_results['forecasts']

        # Calculate dynamic correlations using DCC approach
        # Standardize returns by univariate volatilities
        standardized_returns = pd.DataFrame(index=returns_df.index)
        for col in returns_df.columns:
            if col in univariate_vols:
                vol = univariate_vols[col].reindex(returns_df.index).fillna(method='ffill')
                # Avoid division by zero
                vol = vol.replace(0, np.nan).fillna(vol.mean())
                standardized_returns[col] = returns_df[col] / (vol * 100)  # Rescale

        standardized_returns = standardized_returns.dropna()

        # Rolling correlation estimation (DCC-like)
        for i in range(len(standardized_returns)):
            if i < window:
                continue

            # Use only past data
            historical = standardized_returns.iloc[:i]

            # Calculate correlation matrix
            if len(historical) >= window:
                recent_data = historical.iloc[-window:]
                corr_matrix = recent_data.corr()

                # Store correlation
                date = standardized_returns.index[i]
                results['correlations'][date] = corr_matrix

                # Build covariance matrix
                cov_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
                for row in corr_matrix.index:
                    for col in corr_matrix.columns:
                        if row in univariate_vols and col in univariate_vols:
                            vol_row = univariate_vols[row].loc[date] if date in univariate_vols[row].index else 0.1
                            vol_col = univariate_vols[col].loc[date] if date in univariate_vols[col].index else 0.1
                            cov_matrix.loc[row, col] = corr_matrix.loc[row, col] * vol_row * vol_col

                results['covariance_matrix'].append({
                    'date': date,
                    'matrix': cov_matrix
                })

                # Portfolio volatility forecast (equal weight)
                weights = np.ones(len(corr_matrix)) / len(corr_matrix)
                portfolio_var = weights @ cov_matrix.values @ weights.T
                results['forecasts'][date] = np.sqrt(portfolio_var)

        return results

    def forecast_multiple_horizons(self, returns: pd.Series, pair: str,
                                   horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Forecast volatility for multiple horizons
        """
        forecasts = pd.DataFrame(index=returns.index)

        # Ensure model is fitted
        if pair not in self.models:
            self.fit_univariate_garch(returns, pair)

        if pair not in self.models:
            return forecasts

        model = self.models[pair]

        for i in range(len(returns)):
            if i < 100:
                continue

            try:
                # Multi-step forecast
                forecast = model.forecast(horizon=max(horizons), reindex=False, start=i)

                for h in horizons:
                    # Aggregate variance over horizon
                    var_h = forecast.variance.values[:h, 0].sum()
                    vol_h = np.sqrt(var_h / h) * np.sqrt(252) / 100

                    forecasts.loc[returns.index[i], f'vol_forecast_{h}d'] = vol_h

            except Exception:
                # Use historical vol as fallback
                for h in horizons:
                    if i >= h:
                        hist_vol = returns.iloc[i - h:i].std() * np.sqrt(252) / 100
                        forecasts.loc[returns.index[i], f'vol_forecast_{h}d'] = hist_vol

        return forecasts

    def calculate_vol_risk_premium(self, implied_vols: pd.Series,
                                   garch_forecasts: pd.Series) -> pd.Series:
        """
        Calculate volatility risk premium (IV - GARCH forecast)
        """
        vrp = implied_vols - garch_forecasts
        return vrp

    def evaluate_forecasts(self, realized_vols: pd.Series,
                           forecasts: pd.Series) -> Dict:
        """
        Evaluate forecast accuracy
        """
        # Align series
        aligned = pd.DataFrame({
            'realized': realized_vols,
            'forecast': forecasts
        }).dropna()

        if len(aligned) == 0:
            return {}

        # Calculate metrics
        errors = aligned['realized'] - aligned['forecast']

        metrics = {
            'mae': np.abs(errors).mean(),
            'rmse': np.sqrt((errors ** 2).mean()),
            'mape': (np.abs(errors) / aligned['realized'].replace(0, np.nan)).mean() * 100,
            'bias': errors.mean(),
            'corr': aligned['realized'].corr(aligned['forecast']),
            'hit_rate': ((aligned['realized'] > aligned['realized'].median()) ==
                         (aligned['forecast'] > aligned['forecast'].median())).mean()
        }

        return metrics


class VolatilityComparator:
    """
    Compare different volatility models and create features
    """

    def __init__(self):
        self.garch_forecaster = GARCHForecaster(model_type='GARCH')
        self.egarch_forecaster = GARCHForecaster(model_type='EGARCH')
        self.gjr_forecaster = GARCHForecaster(model_type='GJR-GARCH')
        self.comparison_results = {}

    def compare_vol_models(self, data: pd.DataFrame, pair: str) -> pd.DataFrame:
        """
        Compare implied vol with various GARCH forecasts
        """
        features = pd.DataFrame(index=data.index)

        # Prepare returns
        returns = self.garch_forecaster.prepare_returns(data['spot'])

        # Fit different GARCH models
        garch_results = self.garch_forecaster.fit_univariate_garch(returns, f"{pair}_GARCH")
        egarch_results = self.egarch_forecaster.fit_univariate_garch(returns, f"{pair}_EGARCH")
        gjr_results = self.gjr_forecaster.fit_univariate_garch(returns, f"{pair}_GJR")

        # Store forecasts
        features[f'garch_forecast_{pair}'] = garch_results['forecasts']
        features[f'egarch_forecast_{pair}'] = egarch_results['forecasts']
        features[f'gjr_forecast_{pair}'] = gjr_results['forecasts']

        # Ensemble forecast (average)
        features[f'garch_ensemble_{pair}'] = features[[
            f'garch_forecast_{pair}',
            f'egarch_forecast_{pair}',
            f'gjr_forecast_{pair}'
        ]].mean(axis=1)

        # Compare with implied volatility for different tenors
        for tenor in ['1M', '3M', '6M']:
            if f'atm_vol_{tenor}' in data.columns:
                iv = data[f'atm_vol_{tenor}']

                # Vol risk premium vs different models
                features[f'vrp_garch_{tenor}_{pair}'] = iv - features[f'garch_forecast_{pair}']
                features[f'vrp_egarch_{tenor}_{pair}'] = iv - features[f'egarch_forecast_{pair}']
                features[f'vrp_ensemble_{tenor}_{pair}'] = iv - features[f'garch_ensemble_{pair}']

                # Ratios
                features[f'iv_garch_ratio_{tenor}_{pair}'] = iv / features[f'garch_forecast_{pair}'].replace(0, np.nan)

                # Model disagreement
                features[f'garch_dispersion_{tenor}_{pair}'] = features[[
                    f'garch_forecast_{pair}',
                    f'egarch_forecast_{pair}',
                    f'gjr_forecast_{pair}'
                ]].std(axis=1)

        # Multi-horizon forecasts
        horizons = [1, 5, 10, 20]
        multi_forecasts = self.garch_forecaster.forecast_multiple_horizons(returns, pair, horizons)

        for h in horizons:
            if f'vol_forecast_{h}d' in multi_forecasts.columns:
                features[f'garch_{h}d_{pair}'] = multi_forecasts[f'vol_forecast_{h}d']

        # Volatility term structure from GARCH
        if 'garch_1d_' + pair in features.columns and 'garch_20d_' + pair in features.columns:
            features[f'garch_term_structure_{pair}'] = (
                    features[f'garch_20d_{pair}'] - features[f'garch_1d_{pair}']
            )

        return features

    def simulate_option_prices_with_garch(self, data: pd.DataFrame, pair: str,
                                          models: Dict) -> pd.DataFrame:
        """
        Simulate option prices using GARCH forecasted volatility
        """
        features = pd.DataFrame(index=data.index)

        # Get GARCH forecasts
        returns = self.garch_forecaster.prepare_returns(data['spot'])
        garch_results = self.garch_forecaster.fit_univariate_garch(returns, pair)
        garch_vol = garch_results['forecasts']

        # Simple rates (should use discount curves)
        r_d = 0.05
        r_f = 0.01

        # Price options with different volatilities
        for tenor in ['1M', '3M', '6M']:
            T = {'1M': 1 / 12, '3M': 0.25, '6M': 0.5}[tenor]

            for i in range(len(data)):
                if i < 100:  # Need history
                    continue

                S = data.iloc[i]['spot']
                K = S  # ATM

                # Price with implied vol
                if f'atm_vol_{tenor}' in data.columns:
                    iv = data.iloc[i][f'atm_vol_{tenor}']
                    if not pd.isna(iv) and 'gk' in models:
                        iv_price = models['gk'].price_option(S, K, T, r_d, r_f, iv, 'call')
                        features.loc[data.index[i], f'iv_price_{tenor}_{pair}'] = iv_price

                # Price with GARCH vol
                if i in garch_vol.index:
                    garch_sigma = garch_vol.loc[i] if not pd.isna(garch_vol.loc[i]) else 0.1
                    if 'gk' in models:
                        garch_price = models['gk'].price_option(S, K, T, r_d, r_f, garch_sigma, 'call')
                        features.loc[data.index[i], f'garch_price_{tenor}_{pair}'] = garch_price

                        # Price difference as signal
                        if f'iv_price_{tenor}_{pair}' in features.columns:
                            features.loc[data.index[i], f'price_diff_{tenor}_{pair}'] = (
                                    features.loc[data.index[i], f'iv_price_{tenor}_{pair}'] -
                                    garch_price
                            )

        return features


def create_time_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive time-based features
    """
    features = pd.DataFrame(index=data.index)

    # Basic time features
    features['day_of_week'] = data.index.dayofweek
    features['day_of_month'] = data.index.day
    features['business_day_of_month'] = data.index.to_series().groupby(
        pd.Grouper(freq='M')
    ).cumcount() + 1
    features['month'] = data.index.month
    features['quarter'] = data.index.quarter
    features['year'] = data.index.year
    features['week_of_year'] = data.index.isocalendar().week

    # Binary features
    features['is_monday'] = (features['day_of_week'] == 0).astype(int)
    features['is_friday'] = (features['day_of_week'] == 4).astype(int)
    features['is_month_start'] = data.index.is_month_start.astype(int)
    features['is_month_end'] = data.index.is_month_end.astype(int)
    features['is_quarter_start'] = data.index.is_quarter_start.astype(int)
    features['is_quarter_end'] = data.index.is_quarter_end.astype(int)

    # Days to events
    features['days_to_month_end'] = (
        pd.Series(data.index, index=data.index)
        .apply(lambda x: (x + pd.offsets.MonthEnd(0) - x).days)
    )

    # Cyclical encoding for periodic features
    features['day_of_week_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
    features['day_of_month_sin'] = np.sin(2 * np.pi * features['day_of_month'] / 31)
    features['day_of_month_cos'] = np.cos(2 * np.pi * features['day_of_month'] / 31)
    features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
    features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)

    # Special periods
    features['is_december'] = (features['month'] == 12).astype(int)
    features['is_january'] = (features['month'] == 1).astype(int)
    features['is_summer'] = features['month'].isin([6, 7, 8]).astype(int)

    # Trading patterns
    features['is_opex'] = (
            (features['day_of_week'] == 4) &  # Friday
            (features['day_of_month'] >= 15) &
            (features['day_of_month'] <= 21)
    ).astype(int)

    return features