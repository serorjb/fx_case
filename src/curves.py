# src/curves.py
"""
Discount curves construction using consistent DTB (Treasury Bill) data
All rates are on discount basis, ensuring consistency
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import PchipInterpolator, CubicSpline
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')


class DiscountCurveBuilder:
    """
    Discount curve construction using consistent Treasury Bill rates
    All inputs use the same discount rate convention
    """

    def __init__(self, method: str = 'nelson_siegel_svensson'):
        """
        Initialize curve builder

        Parameters:
        -----------
        method : str
            Interpolation method: 'linear', 'cubic', 'pchip', 'nelson_siegel', 'nelson_siegel_svensson'
        """
        self.method = method
        self.curves = {}
        self.parameters = {}

    def discount_to_yield(self, discount_rate: float, days_to_maturity: int) -> float:
        """
        Convert Treasury Bill discount rate to bond equivalent yield
        This ensures we work with yields consistently

        Parameters:
        -----------
        discount_rate : float
            Discount rate (as decimal, e.g., 0.05 for 5%)
        days_to_maturity : int
            Days to maturity

        Returns:
        --------
        float : Bond equivalent yield
        """
        # Avoid division by zero
        if days_to_maturity == 0:
            return discount_rate

        # Treasury bill discount to yield conversion
        # BEY = (365 * discount_rate) / (360 - discount_rate * days_to_maturity)
        denominator = 360 - discount_rate * days_to_maturity

        if denominator <= 0:
            # Fallback for extreme cases
            return discount_rate

        bey = (365 * discount_rate) / denominator

        return bey

    def yield_to_continuous(self, yield_rate: float, compound_freq: int = 2) -> float:
        """
        Convert yield to continuous compounding rate

        Parameters:
        -----------
        yield_rate : float
            Yield rate (as decimal)
        compound_freq : int
            Compounding frequency (2 for semi-annual US Treasury convention)

        Returns:
        --------
        float : Continuous rate
        """
        if compound_freq == 0 or yield_rate <= -1:
            return yield_rate

        # r_continuous = n * ln(1 + r_periodic/n)
        return compound_freq * np.log(1 + yield_rate / compound_freq)

    def load_fred_data(self, data_dir: Path) -> pd.DataFrame:
        """
        Load FRED Treasury Bill data (all on discount basis)
        """

        def load_fred_file(filename: Path, label: str) -> pd.DataFrame:
            """
            Load single FRED file
            """
            try:
                df = pd.read_csv(filename)
                df.columns = ['date', label]
                df['date'] = pd.to_datetime(df['date'])

                # Convert from percentage to decimal
                df[label] = pd.to_numeric(df[label], errors='coerce') / 100

                return df.set_index('date')

            except Exception as e:
                print(f"Error loading {filename}: {e}")
                return pd.DataFrame()

        # Load only DTB files (all Treasury Bills on discount basis)
        # This ensures consistency - all rates use the same convention
        files = {
            "DTB4WK.csv": "1M",  # 4-week T-bill
            "DTB3.csv": "3M",  # 3-month T-bill
            "DTB6.csv": "6M",  # 6-month T-bill
            "DTB1YR.csv": "1Y"  # 1-year T-bill
        }

        dfs = []
        for filename, label in files.items():
            filepath = data_dir / filename
            if filepath.exists():
                df = load_fred_file(filepath, label)
                if not df.empty:
                    dfs.append(df)
                    print(f"  ✓ Loaded {filename}: {label} T-bill rate")
            else:
                print(f"  ⚠️ File not found: {filepath}")

        if not dfs:
            raise ValueError("No FRED DTB data files found")

        # Merge all dataframes
        df_rates = dfs[0]
        for df in dfs[1:]:
            df_rates = df_rates.join(df, how='outer')

        # Sort and forward fill
        df_rates = df_rates.sort_index().ffill()

        print(f"  Loaded rates data: {df_rates.shape[0]} days, {df_rates.shape[1]} tenors")

        return df_rates

    def nelson_siegel(self, t: np.ndarray, beta0: float, beta1: float,
                      beta2: float, tau: float) -> np.ndarray:
        """
        Nelson-Siegel model for yield curve
        """
        tau_t = t / tau
        exp_tau = np.exp(-tau_t)

        # Three factors: level, slope, curvature
        factor1 = np.ones_like(t)
        factor2 = (1 - exp_tau) / tau_t
        factor3 = factor2 - exp_tau

        # Handle t=0 case
        factor2[t == 0] = 1
        factor3[t == 0] = 0

        return beta0 * factor1 + beta1 * factor2 + beta2 * factor3

    def nelson_siegel_svensson(self, t: np.ndarray, beta0: float, beta1: float,
                               beta2: float, beta3: float, tau1: float, tau2: float) -> np.ndarray:
        """
        Nelson-Siegel-Svensson model (extended with additional hump)
        """
        tau1_t = t / tau1
        tau2_t = t / tau2
        exp_tau1 = np.exp(-tau1_t)
        exp_tau2 = np.exp(-tau2_t)

        # Four factors
        factor1 = np.ones_like(t)
        factor2 = (1 - exp_tau1) / tau1_t
        factor3 = factor2 - exp_tau1
        factor4 = (1 - exp_tau2) / tau2_t - exp_tau2

        # Handle t=0 case
        factor2[t == 0] = 1
        factor3[t == 0] = 0
        factor4[t == 0] = 0

        return beta0 * factor1 + beta1 * factor2 + beta2 * factor3 + beta3 * factor4

    def fit_nelson_siegel_svensson(self, tenors: np.ndarray, rates: np.ndarray) -> Dict:
        """
        Fit NSS model using global optimization
        """

        def objective(params):
            beta0, beta1, beta2, beta3, tau1, tau2 = params
            fitted = self.nelson_siegel_svensson(tenors, beta0, beta1, beta2, beta3, tau1, tau2)
            return np.sum((fitted - rates) ** 2)

        # Bounds for parameters (adjusted for T-bill rates)
        bounds = [
            (0, 0.15),  # beta0 (level) - max 15%
            (-0.15, 0.15),  # beta1 (slope)
            (-0.15, 0.15),  # beta2 (curvature)
            (-0.15, 0.15),  # beta3 (second curvature)
            (0.1, 5),  # tau1
            (0.1, 5)  # tau2
        ]

        # Use differential evolution for global optimization
        result = differential_evolution(objective, bounds, seed=42, maxiter=1000)

        if result.success:
            params = result.x
            return {
                'beta0': params[0],
                'beta1': params[1],
                'beta2': params[2],
                'beta3': params[3],
                'tau1': params[4],
                'tau2': params[5],
                'rmse': np.sqrt(result.fun / len(rates))
            }
        else:
            return None

    def build_curve(self, date: pd.Timestamp, market_rates: pd.Series,
                    target_tenors: Dict[str, float]) -> pd.DataFrame:
        """
        Build discount curve for a given date
        """
        # Remove NaN values
        available = market_rates.dropna()
        if len(available) < 2:
            return pd.DataFrame()

        # Convert discount rates to yields for better interpolation
        # Map tenor labels to approximate days
        days_map = {'1M': 30, '3M': 91, '6M': 182, '1Y': 365}

        market_tenors = []
        market_yields = []

        for label, discount_rate in available.items():
            # Get tenor in years
            if label == '1M':
                tenor_years = 1 / 12
                days = 30
            elif label == '3M':
                tenor_years = 0.25
                days = 91
            elif label == '6M':
                tenor_years = 0.5
                days = 182
            elif label == '1Y':
                tenor_years = 1.0
                days = 365
            else:
                continue

            # Convert discount rate to yield
            yield_rate = self.discount_to_yield(discount_rate, days)

            market_tenors.append(tenor_years)
            market_yields.append(yield_rate)

        market_tenors = np.array(market_tenors)
        market_yields = np.array(market_yields)

        # Sort by tenor
        sort_idx = np.argsort(market_tenors)
        market_tenors = market_tenors[sort_idx]
        market_yields = market_yields[sort_idx]

        records = []

        if self.method == 'nelson_siegel_svensson':
            # Fit NSS model
            params = self.fit_nelson_siegel_svensson(market_tenors, market_yields)

            if params:
                # Use NSS model for interpolation/extrapolation
                for tenor_label, tenor_years in target_tenors.items():
                    if tenor_years > 1.0:
                        continue  # Skip tenors beyond 1 year

                    rate = self.nelson_siegel_svensson(
                        np.array([tenor_years]),
                        params['beta0'], params['beta1'], params['beta2'],
                        params['beta3'], params['tau1'], params['tau2']
                    )[0]

                    # Ensure reasonable bounds for short-term rates
                    rate = np.clip(rate, -0.01, 0.20)

                    # Calculate discount factor
                    df = np.exp(-rate * tenor_years)

                    records.append({
                        'date': date,
                        'tenor': tenor_label,
                        'tenor_years': tenor_years,
                        'interpolated_rate': rate,
                        'discount_factor': df,
                        'method': 'NSS'
                    })
            else:
                # Fallback to PCHIP
                self.method = 'pchip'

        if self.method in ['pchip', 'cubic', 'linear']:
            # Build in discount factor space
            discount_factors = np.exp(-market_yields * market_tenors)

            # Choose interpolator
            if self.method == 'pchip':
                # PCHIP preserves monotonicity (preferred for discount curves)
                interpolator = PchipInterpolator(market_tenors, discount_factors, extrapolate=True)
            elif self.method == 'cubic':
                interpolator = CubicSpline(market_tenors, discount_factors, extrapolate=True)
            else:  # linear
                from scipy.interpolate import interp1d
                interpolator = interp1d(market_tenors, discount_factors,
                                        kind='linear', fill_value='extrapolate')

            # Interpolate for target tenors
            for tenor_label, tenor_years in target_tenors.items():
                if tenor_years > 1.0:
                    continue  # Skip tenors beyond 1 year

                df = float(interpolator(tenor_years))

                # Ensure arbitrage-free constraints
                df = np.clip(df, 0.8, 1.0)  # Reasonable bounds for up to 1Y

                # Calculate implied rate
                if tenor_years > 0:
                    rate = -np.log(df) / tenor_years
                    rate = np.clip(rate, -0.01, 0.20)
                    # Recalculate df with bounded rate
                    df = np.exp(-rate * tenor_years)
                else:
                    rate = 0
                    df = 1

                records.append({
                    'date': date,
                    'tenor': tenor_label,
                    'tenor_years': tenor_years,
                    'interpolated_rate': rate,
                    'discount_factor': df,
                    'method': self.method
                })

        return pd.DataFrame(records)

    def build_curves_for_period(self, df_rates: pd.DataFrame,
                                target_tenors: Dict[str, float]) -> pd.DataFrame:
        """
        Build curves for entire period with consistency checks
        """
        all_records = []

        for date, row in df_rates.iterrows():
            curve_data = self.build_curve(date, row, target_tenors)
            if not curve_data.empty:
                all_records.append(curve_data)

        if not all_records:
            return pd.DataFrame()

        df_curves = pd.concat(all_records, ignore_index=True)

        # Apply consistency checks
        df_curves = self.apply_arbitrage_constraints(df_curves)

        return df_curves

    def apply_arbitrage_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply no-arbitrage constraints to ensure consistency
        """
        # Group by date
        for date in df['date'].unique():
            date_mask = df['date'] == date
            date_data = df[date_mask].sort_values('tenor_years')

            # Ensure discount factors are decreasing
            prev_df = 1.0
            for idx in date_data.index:
                current_df = df.loc[idx, 'discount_factor']
                tenor = df.loc[idx, 'tenor_years']

                if current_df > prev_df and tenor > 0:
                    # Violation: adjust to maintain monotonicity
                    df.loc[idx, 'discount_factor'] = prev_df * (1 - 0.001 * tenor)

                    # Recalculate rate
                    if tenor > 0:
                        df.loc[idx, 'interpolated_rate'] = -np.log(df.loc[idx, 'discount_factor']) / tenor

                prev_df = df.loc[idx, 'discount_factor']

        return df

    def calculate_forward_rates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate forward rates from discount factors
        """
        df = df.copy()

        # Group by date
        for date in df['date'].unique():
            date_mask = df['date'] == date
            date_data = df[date_mask].sort_values('tenor_years')

            # Calculate forward rates between consecutive tenors
            for i in range(len(date_data) - 1):
                t1 = date_data.iloc[i]['tenor_years']
                t2 = date_data.iloc[i + 1]['tenor_years']
                df1 = date_data.iloc[i]['discount_factor']
                df2 = date_data.iloc[i + 1]['discount_factor']

                if t2 > t1 and df2 > 0:
                    # Forward rate from t1 to t2
                    forward_rate = -np.log(df2 / df1) / (t2 - t1)

                    # Store forward rate
                    tenor1 = date_data.iloc[i]['tenor']
                    tenor2 = date_data.iloc[i + 1]['tenor']

                    idx = date_data.iloc[i].name
                    df.loc[idx, f'forward_{tenor1}_{tenor2}'] = forward_rate

        return df

    def plot_curve_3d(self, df: pd.DataFrame, output_path: str = 'plots/discount_curves_3d.html'):
        """
        Create 3D visualization of yield curves
        """
        # Pivot data
        pivot = df.pivot_table(index='date', columns='tenor_years', values='interpolated_rate')

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'surface', 'colspan': 2}, None],
                   [{'type': 'scatter'}, {'type': 'scatter'}]],
            subplot_titles=('T-Bill Yield Surface Over Time',
                            'Current Yield Curve',
                            'Historical 3M Rate'),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )

        # 3D Surface
        fig.add_trace(
            go.Surface(
                x=pivot.columns.values,
                y=pivot.index,
                z=pivot.values * 100,  # Convert to percentage
                colorscale='Viridis',
                name='Yield Surface'
            ),
            row=1, col=1
        )

        # Current yield curve (latest date)
        if len(pivot) > 0:
            latest_date = pivot.index[-1]
            latest_curve = pivot.loc[latest_date] * 100

            fig.add_trace(
                go.Scatter(
                    x=pivot.columns.values,
                    y=latest_curve.values,
                    mode='lines+markers',
                    name=f'Curve on {latest_date.strftime("%Y-%m-%d")}',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )

        # Historical 3M rate
        if 0.25 in pivot.columns:  # 3M = 0.25 years
            historical_3m = pivot[0.25] * 100

            fig.add_trace(
                go.Scatter(
                    x=pivot.index,
                    y=historical_3m.values,
                    mode='lines',
                    name='3M T-Bill Rate',
                    line=dict(color='red', width=2)
                ),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title='U.S. Treasury Bill Discount Curves (Consistent DTB Data)',
            height=900,
            showlegend=True,
            scene=dict(
                xaxis_title='Tenor (years)',
                yaxis_title='Date',
                zaxis_title='Rate (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )

        # Update axes
        fig.update_xaxes(title_text='Tenor (years)', row=2, col=1)
        fig.update_yaxes(title_text='Rate (%)', row=2, col=1)
        fig.update_xaxes(title_text='Date', row=2, col=2)
        fig.update_yaxes(title_text='3M T-Bill Rate (%)', row=2, col=2)

        # Save
        fig.write_html(output_path)
        print(f"  ✓ Discount curve visualization saved to {output_path}")

        return fig


def process_discount_curves(data_dir: Path = Path("../data/FRED"),
                            output_dir: Path = Path("data"),
                            plot_dir: Path = Path("plots")):
    """
    Main function to process discount curves using consistent T-Bill data
    """
    # Create directories
    output_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)

    # Initialize builder
    builder = DiscountCurveBuilder(method='nelson_siegel_svensson')

    # Load FRED T-Bill data
    print("📊 Loading FRED Treasury Bill data...")
    df_rates = builder.load_fred_data(data_dir)

    # Define target tenors for FX options (up to 1Y only)
    target_tenors = {
        '1W': 1 / 52,
        '2W': 2 / 52,
        '3W': 3 / 52,
        '1M': 1 / 12,
        '2M': 2 / 12,
        '3M': 3 / 12,
        '4M': 4 / 12,
        '6M': 6 / 12,
        '9M': 9 / 12,
        '1Y': 1.0
    }

    # Build curves
    print("📈 Building discount curves...")
    df_curves = builder.build_curves_for_period(df_rates, target_tenors)

    # Calculate forward rates
    print("🔄 Calculating forward rates...")
    df_curves = builder.calculate_forward_rates(df_curves)

    # Sort
    df_curves = df_curves.sort_values(['date', 'tenor_years'])

    # Save to parquet
    output_path = output_dir / "discount_curves.parquet"
    df_curves.to_parquet(output_path, index=False)
    print(f"✅ Discount curves saved to {output_path}")

    # Create visualization
    print("📊 Creating visualizations...")
    builder.plot_curve_3d(df_curves, str(plot_dir / "discount_curves_3d.html"))

    # Print summary statistics
    print("\n📈 Summary Statistics:")
    print(f"  Date range: {df_curves['date'].min().date()} to {df_curves['date'].max().date()}")
    print(f"  Number of dates: {df_curves['date'].nunique()}")
    print(f"  Tenors included: {sorted(df_curves['tenor'].unique())}")
    print(f"  Rate range: {df_curves['interpolated_rate'].min():.4f} to {df_curves['interpolated_rate'].max():.4f}")
    print(f"  Data consistency: All rates from Treasury Bills (discount basis)")

    return df_curves


if __name__ == "__main__":
    # Run as standalone script
    df_curves = process_discount_curves()