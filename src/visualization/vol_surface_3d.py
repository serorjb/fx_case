"""
3D Volatility Surface Visualization with Historical Evolution
Shows how the volatility skew changes over time
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional
from pathlib import Path
from scipy.interpolate import griddata
import warnings

warnings.filterwarnings('ignore')


class VolatilitySurface3D:
    """
    Creates 3D volatility surfaces showing historical evolution
    """

    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def calculate_implied_strikes(self, spot: float, delta: float, atm_vol: float,
                                  T: float = 1 / 12, r_d: float = 0.05, r_f: float = 0.01) -> float:
        """Calculate strike from delta"""
        from scipy.stats import norm

        # Simplified strike calculation from delta
        # For calls: Delta = N(d1)
        # For puts: Delta = -N(-d1)

        if delta > 0:  # Call
            d1 = norm.ppf(delta)
        else:  # Put
            d1 = -norm.ppf(-delta)

        # Back out strike from d1
        # d1 = (ln(S/K) + (r_d - r_f + 0.5*vol^2)*T) / (vol*sqrt(T))
        vol_sqrt_t = atm_vol * np.sqrt(T)
        ln_s_over_k = d1 * vol_sqrt_t - (r_d - r_f + 0.5 * atm_vol ** 2) * T

        K = spot * np.exp(-ln_s_over_k)
        return K

    def construct_vol_smile(self, spot: float, atm_vol: float, rr_25: float,
                            bf_25: float, rr_10: float = None, bf_10: float = None) -> Dict:
        """Construct volatility smile from market quotes"""

        # Calculate implied volatilities
        vol_25_call = atm_vol + bf_25 + rr_25 / 2
        vol_25_put = atm_vol + bf_25 - rr_25 / 2

        if rr_10 is not None and bf_10 is not None:
            vol_10_call = atm_vol + bf_10 + rr_10 / 2
            vol_10_put = atm_vol + bf_10 - rr_10 / 2
        else:
            # Extrapolate 10 delta vols
            vol_10_call = vol_25_call + (vol_25_call - atm_vol) * 0.5
            vol_10_put = vol_25_put + (vol_25_put - atm_vol) * 0.5

        # Calculate strikes
        T = 1 / 12  # 1 month
        K_atm = spot  # Simplified ATM
        K_25_call = self.calculate_implied_strikes(spot, 0.25, vol_25_call, T)
        K_25_put = self.calculate_implied_strikes(spot, -0.25, vol_25_put, T)
        K_10_call = self.calculate_implied_strikes(spot, 0.10, vol_10_call, T)
        K_10_put = self.calculate_implied_strikes(spot, -0.10, vol_10_put, T)

        return {
            'strikes': [K_10_put, K_25_put, K_atm, K_25_call, K_10_call],
            'vols': [vol_10_put, vol_25_put, atm_vol, vol_25_call, vol_10_call],
            'deltas': [-0.10, -0.25, 0.50, 0.25, 0.10]
        }

    def plot_historical_vol_surface(self, data: pd.DataFrame, pair: str,
                                    lookback_days: int = 60,
                                    sample_freq: int = 5) -> None:
        """
        Create 3D volatility surface showing historical evolution

        Args:
            data: DataFrame with FX option data
            pair: Currency pair name
            lookback_days: Number of days to look back
            sample_freq: Sample every N days for performance
        """

        # Get recent data
        recent_data = data.iloc[-lookback_days:].copy()

        # Sample data for better visualization
        sampled_data = recent_data.iloc[::sample_freq]

        # Prepare data for 3D surface
        dates = []
        strikes_normalized = []
        vols = []

        for idx, (date, row) in enumerate(sampled_data.iterrows()):
            spot = row.get('spot', 100)
            atm_vol = row.get('atm_vol_1M', 0.1)
            rr_25 = row.get('rr_25_1M', 0)
            bf_25 = row.get('bf_25_1M', 0)
            rr_10 = row.get('rr_10_1M', None)
            bf_10 = row.get('bf_10_1M', None)

            # Skip if missing critical data
            if pd.isna(spot) or pd.isna(atm_vol):
                continue

            # Replace NaN with defaults
            rr_25 = 0 if pd.isna(rr_25) else rr_25
            bf_25 = 0 if pd.isna(bf_25) else bf_25

            # Construct smile
            smile = self.construct_vol_smile(spot, atm_vol, rr_25, bf_25, rr_10, bf_10)

            # Normalize strikes as moneyness (K/S)
            for strike, vol in zip(smile['strikes'], smile['vols']):
                dates.append(idx)  # Use index for time axis
                strikes_normalized.append(strike / spot)  # Moneyness
                vols.append(vol * 100)  # Convert to percentage

        if len(dates) == 0:
            print(f"No valid data for {pair}")
            return

        # Create 3D surface plot
        fig = go.Figure()

        # Create grid for interpolation
        date_grid = np.linspace(min(dates), max(dates), 30)
        strike_grid = np.linspace(min(strikes_normalized), max(strikes_normalized), 30)
        date_mesh, strike_mesh = np.meshgrid(date_grid, strike_grid)

        # Interpolate volatility surface
        points = np.column_stack((dates, strikes_normalized))
        vol_grid = griddata(points, vols, (date_mesh, strike_mesh), method='cubic')

        # Add surface
        fig.add_trace(go.Surface(
            x=date_mesh,
            y=strike_mesh,
            z=vol_grid,
            colorscale='Viridis',
            name='Volatility Surface',
            showscale=True,
            colorbar=dict(
                title="IV (%)",
                x=1.0
            ),
            hovertemplate='Time: %{x}<br>Moneyness: %{y:.3f}<br>IV: %{z:.2f}%<extra></extra>'
        ))

        # Add scatter points for actual data
        fig.add_trace(go.Scatter3d(
            x=dates,
            y=strikes_normalized,
            z=vols,
            mode='markers',
            marker=dict(
                size=3,
                color=vols,
                colorscale='Viridis',
                showscale=False
            ),
            name='Market Quotes',
            hovertemplate='IV: %{z:.2f}%<extra></extra>'
        ))

        # Get date labels for x-axis
        date_labels = []
        date_positions = []
        for i in range(0, len(sampled_data), max(1, len(sampled_data) // 5)):
            date_positions.append(i)
            date_labels.append(sampled_data.index[i].strftime('%Y-%m-%d'))

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{pair} Implied Volatility Surface - Historical Evolution<br>' +
                     f'<sub>Last {lookback_days} days</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis=dict(
                    title='Time',
                    ticktext=date_labels,
                    tickvals=date_positions,
                    showgrid=True,
                    gridcolor='lightgray'
                ),
                yaxis=dict(
                    title='Moneyness (K/S)',
                    showgrid=True,
                    gridcolor='lightgray',
                    tickformat='.2f'
                ),
                zaxis=dict(
                    title='Implied Vol (%)',
                    showgrid=True,
                    gridcolor='lightgray',
                    tickformat='.1f'
                ),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.2, y=1, z=0.8)
            ),
            height=700,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=0, r=0, t=80, b=0)
        )

        # Save plot
        output_file = self.output_dir / f'vol_surface_{pair}.html'
        fig.write_html(str(output_file))
        print(f"3D volatility surface saved to {output_file}")

        # Also create a 2D heatmap view
        self.plot_vol_heatmap(data, pair, lookback_days, sample_freq)

    def plot_vol_heatmap(self, data: pd.DataFrame, pair: str,
                         lookback_days: int = 60,
                         sample_freq: int = 5) -> None:
        """Create 2D heatmap of volatility evolution"""

        recent_data = data.iloc[-lookback_days:].copy()
        sampled_data = recent_data.iloc[::sample_freq]

        # Prepare data matrix
        dates_list = []
        vol_matrix = []

        for date, row in sampled_data.iterrows():
            spot = row.get('spot', 100)
            atm_vol = row.get('atm_vol_1M', 0.1)
            rr_25 = row.get('rr_25_1M', 0)
            bf_25 = row.get('bf_25_1M', 0)

            if pd.isna(spot) or pd.isna(atm_vol):
                continue

            rr_25 = 0 if pd.isna(rr_25) else rr_25
            bf_25 = 0 if pd.isna(bf_25) else bf_25

            smile = self.construct_vol_smile(spot, atm_vol, rr_25, bf_25)

            dates_list.append(date.strftime('%Y-%m-%d'))
            vol_matrix.append([v * 100 for v in smile['vols']])

        if len(vol_matrix) == 0:
            return

        vol_matrix = np.array(vol_matrix).T

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=vol_matrix,
            x=dates_list,
            y=['10Δ Put', '25Δ Put', 'ATM', '25Δ Call', '10Δ Call'],
            colorscale='RdYlBu_r',
            colorbar=dict(title="IV (%)"),
            hovertemplate='Date: %{x}<br>Strike: %{y}<br>IV: %{z:.2f}%<extra></extra>'
        ))

        fig.update_layout(
            title=f'{pair} Volatility Smile Evolution',
            xaxis_title='Date',
            yaxis_title='Strike (by Delta)',
            height=500,
            xaxis=dict(
                tickangle=-45,
                tickmode='auto',
                nticks=10
            )
        )

        output_file = self.output_dir / f'vol_heatmap_{pair}.html'
        fig.write_html(str(output_file))
        print(f"Volatility heatmap saved to {output_file}")

    def plot_skew_evolution(self, data: pd.DataFrame, pair: str,
                            lookback_days: int = 90) -> None:
        """Plot the evolution of volatility skew metrics over time"""

        recent_data = data.iloc[-lookback_days:].copy()

        # Calculate skew metrics
        recent_data['skew_25'] = recent_data['rr_25_1M']  # Risk reversal is a skew measure
        recent_data['convexity'] = recent_data['bf_25_1M']  # Butterfly is convexity

        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('ATM Volatility', '25Δ Risk Reversal (Skew)', '25Δ Butterfly (Convexity)'),
            vertical_spacing=0.1
        )

        # ATM Vol
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['atm_vol_1M'] * 100,
                mode='lines',
                name='ATM Vol',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )

        # Risk Reversal
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['skew_25'] * 100,
                mode='lines',
                name='25Δ RR',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )

        # Add zero line for RR
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Butterfly
        fig.add_trace(
            go.Scatter(
                x=recent_data.index,
                y=recent_data['convexity'] * 100,
                mode='lines',
                name='25Δ BF',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Vol (%)", row=1, col=1)
        fig.update_yaxes(title_text="RR (%)", row=2, col=1)
        fig.update_yaxes(title_text="BF (%)", row=3, col=1)

        fig.update_layout(
            title=f'{pair} Volatility Smile Components Evolution',
            height=800,
            showlegend=False
        )

        output_file = self.output_dir / f'skew_evolution_{pair}.html'
        fig.write_html(str(output_file))
        print(f"Skew evolution saved to {output_file}")


def create_enhanced_vol_surfaces(data: Dict[str, pd.DataFrame], output_dir: str = "plots"):
    """
    Create enhanced 3D volatility surfaces for all currency pairs
    """
    viz = VolatilitySurface3D(output_dir)

    for pair, pair_data in data.items():
        print(f"\nCreating 3D volatility surface for {pair}...")

        # Create 3D surface
        viz.plot_historical_vol_surface(pair_data, pair, lookback_days=60, sample_freq=3)

        # Create skew evolution
        viz.plot_skew_evolution(pair_data, pair, lookback_days=90)

    print("\n✅ All 3D volatility surfaces created successfully")


# Integration function for main.py
def add_3d_vol_surface_to_visualization(self, data: Dict[str, pd.DataFrame]) -> None:
    """
    Add this method to the FXOptionsSystem class in main.py
    Call it in the generate_comprehensive_visualizations method
    """
    from src.visualization.vol_surface_3d import create_enhanced_vol_surfaces

    print("\n📊 Creating 3D volatility surfaces...")
    try:
        create_enhanced_vol_surfaces(data, str(self.PLOTS_DIR))
        print("   ✓ 3D volatility surfaces created")
    except Exception as e:
        print(f"   ⚠️ Could not create 3D vol surfaces: {e}")