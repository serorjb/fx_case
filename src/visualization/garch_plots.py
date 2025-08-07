"""
Visualization for GARCH models and volatility analysis
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List


class GARCHVisualizer:
    """
    Create visualizations for GARCH analysis
    """

    def __init__(self, output_dir: str = 'plots'):
        self.output_dir = output_dir

    def plot_volatility_comparison(self, data: pd.DataFrame, pair: str,
                                   garch_features: pd.DataFrame) -> None:
        """
        Plot comparison of different volatility measures
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=(
                f'{pair} Volatility Comparison',
                'Volatility Risk Premium',
                'Model Disagreement'
            ),
            vertical_spacing=0.05,
            row_heights=[0.4, 0.3, 0.3]
        )

        # Plot 1: Volatility comparison
        if 'atm_vol_1M' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['atm_vol_1M'] * 100,
                           name='Implied Vol 1M', line=dict(color='blue', width=2)),
                row=1, col=1
            )

        if f'garch_forecast_{pair}' in garch_features.columns:
            fig.add_trace(
                go.Scatter(x=garch_features.index, y=garch_features[f'garch_forecast_{pair}'] * 100,
                           name='GARCH Forecast', line=dict(color='red', width=2)),
                row=1, col=1
            )

        if f'egarch_forecast_{pair}' in garch_features.columns:
            fig.add_trace(
                go.Scatter(x=garch_features.index, y=garch_features[f'egarch_forecast_{pair}'] * 100,
                           name='EGARCH Forecast', line=dict(color='green', width=1, dash='dash')),
                row=1, col=1
            )

        if f'garch_ensemble_{pair}' in garch_features.columns:
            fig.add_trace(
                go.Scatter(x=garch_features.index, y=garch_features[f'garch_ensemble_{pair}'] * 100,
                           name='Ensemble Forecast', line=dict(color='purple', width=2, dash='dot')),
                row=1, col=1
            )

        # Calculate realized volatility
        returns = np.log(data['spot'] / data['spot'].shift(1))
        realized_vol = returns.rolling(20).std() * np.sqrt(252) * 100

        fig.add_trace(
            go.Scatter(x=data.index, y=realized_vol,
                       name='Realized Vol (20d)', line=dict(color='orange', width=1)),
            row=1, col=1
        )

        # Plot 2: Volatility Risk Premium
        if f'vrp_garch_1M_{pair}' in garch_features.columns:
            vrp = garch_features[f'vrp_garch_1M_{pair}'] * 100

            fig.add_trace(
                go.Scatter(x=garch_features.index, y=vrp,
                           name='VRP (IV - GARCH)', fill='tozeroy',
                           line=dict(color='darkblue', width=1)),
                row=2, col=1
            )

            # Add zero line
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

        # Plot 3: Model Disagreement
        if f'garch_dispersion_1M_{pair}' in garch_features.columns:
            dispersion = garch_features[f'garch_dispersion_1M_{pair}'] * 100

            fig.add_trace(
                go.Scatter(x=garch_features.index, y=dispersion,
                           name='Model Dispersion', fill='tozeroy',
                           line=dict(color='red', width=1)),
                row=3, col=1
            )

        # Update layout
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Volatility (%)", row=1, col=1)
        fig.update_yaxes(title_text="VRP (%)", row=2, col=1)
        fig.update_yaxes(title_text="Dispersion (%)", row=3, col=1)

        fig.update_layout(
            title=f"{pair} GARCH Volatility Analysis",
            height=900,
            hovermode='x unified',
            showlegend=True
        )

        # Save
        output_path = f"{self.output_dir}/garch_analysis_{pair}.html"
        fig.write_html(output_path)
        print(f"GARCH analysis plot saved to {output_path}")

    def plot_multivariate_analysis(self, mv_features: pd.DataFrame,
                                   pairs: List[str]) -> None:
        """
        Plot multivariate GARCH analysis
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Portfolio Volatility Forecast',
                'Average Correlation',
                'Correlation Heatmap (Latest)',
                'Correlation Dynamics'
            )
        )

        # Plot 1: Portfolio volatility
        if 'mv_garch_portfolio_vol' in mv_features.columns:
            fig.add_trace(
                go.Scatter(x=mv_features.index, y=mv_features['mv_garch_portfolio_vol'] * 100,
                           name='Portfolio Vol', line=dict(color='blue', width=2)),
                row=1, col=1
            )

        # Plot 2: Average correlation
        if 'avg_correlation' in mv_features.columns:
            fig.add_trace(
                go.Scatter(x=mv_features.index, y=mv_features['avg_correlation'],
                           name='Avg Correlation', line=dict(color='green', width=2)),
                row=1, col=2
            )

            # Add bands
            if 'max_correlation' in mv_features.columns:
                fig.add_trace(
                    go.Scatter(x=mv_features.index, y=mv_features['max_correlation'],
                               name='Max', line=dict(color='red', width=1, dash='dash')),
                    row=1, col=2
                )

            if 'min_correlation' in mv_features.columns:
                fig.add_trace(
                    go.Scatter(x=mv_features.index, y=mv_features['min_correlation'],
                               name='Min', line=dict(color='blue', width=1, dash='dash')),
                    row=1, col=2
                )

        # Plot 3: Correlation heatmap (latest)
        corr_cols = [col for col in mv_features.columns if col.startswith('corr_') and '_' in col[5:]]
        if corr_cols:
            # Create correlation matrix from latest data
            latest_data = mv_features[corr_cols].iloc[-1]

            # Reconstruct correlation matrix
            n_pairs = len(pairs)
            corr_matrix = np.eye(n_pairs)

            for col in corr_cols:
                parts = col.replace('corr_', '').split('_')
                if len(parts) >= 2:
                    pair1, pair2 = parts[0], parts[1]
                    if pair1 in pairs and pair2 in pairs:
                        i, j = pairs.index(pair1), pairs.index(pair2)
                        corr_matrix[i, j] = latest_data[col]
                        corr_matrix[j, i] = latest_data[col]

            fig.add_trace(
                go.Heatmap(z=corr_matrix, x=pairs, y=pairs,
                           colorscale='RdBu', zmid=0),
                row=2, col=1
            )

        # Plot 4: Correlation dynamics
        for col in corr_cols[:3]:  # Plot first 3 correlations
            fig.add_trace(
                go.Scatter(x=mv_features.index, y=mv_features[col],
                           name=col.replace('corr_', '').replace('_', ' vs '),
                           line=dict(width=2)),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title="Multivariate GARCH Analysis",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )

        # Save
        output_path = f"{self.output_dir}/multivariate_garch_analysis.html"
        fig.write_html(output_path)
        print(f"Multivariate GARCH analysis saved to {output_path}")

    def plot_vol_forecast_accuracy(self, realized_vols: pd.Series,
                                   forecasts_dict: Dict[str, pd.Series]) -> None:
        """
        Plot forecast accuracy comparison
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Volatility Forecasts vs Realized',
                'Forecast Errors',
                'Error Distribution',
                'Forecast Accuracy Metrics'
            )
        )

        # Plot 1: Forecasts vs Realized
        fig.add_trace(
            go.Scatter(x=realized_vols.index, y=realized_vols * 100,
                       name='Realized', line=dict(color='black', width=2)),
            row=1, col=1
        )

        colors = ['blue', 'red', 'green', 'purple', 'orange']
        for i, (name, forecast) in enumerate(forecasts_dict.items()):
            fig.add_trace(
                go.Scatter(x=forecast.index, y=forecast * 100,
                           name=name, line=dict(color=colors[i % len(colors)], width=1)),
                row=1, col=1
            )

        # Plot 2: Forecast Errors
        for i, (name, forecast) in enumerate(forecasts_dict.items()):
            aligned = pd.DataFrame({'realized': realized_vols, 'forecast': forecast}).dropna()
            errors = (aligned['realized'] - aligned['forecast']) * 100

            fig.add_trace(
                go.Scatter(x=errors.index, y=errors,
                           name=f'{name} Error', line=dict(color=colors[i % len(colors)], width=1)),
                row=1, col=2
            )

        # Plot 3: Error Distribution
        for i, (name, forecast) in enumerate(forecasts_dict.items()):
            aligned = pd.DataFrame({'realized': realized_vols, 'forecast': forecast}).dropna()
            errors = (aligned['realized'] - aligned['forecast']) * 100

            fig.add_trace(
                go.Histogram(x=errors, name=name, opacity=0.5,
                             marker_color=colors[i % len(colors)]),
                row=2, col=1
            )

        # Plot 4: Accuracy Metrics
        metrics_data = []
        for name, forecast in forecasts_dict.items():
            aligned = pd.DataFrame({'realized': realized_vols, 'forecast': forecast}).dropna()
            if len(aligned) > 0:
                errors = aligned['realized'] - aligned['forecast']
                mae = np.abs(errors).mean() * 100
                rmse = np.sqrt((errors ** 2).mean()) * 100
                corr = aligned['realized'].corr(aligned['forecast'])

                metrics_data.append({
                    'Model': name,
                    'MAE (%)': mae,
                    'RMSE (%)': rmse,
                    'Correlation': corr
                })

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)

            fig.add_trace(
                go.Bar(x=metrics_df['Model'], y=metrics_df['MAE (%)'],
                       name='MAE', marker_color='blue'),
                row=2, col=2
            )

        # Update layout
        fig.update_layout(
            title="Volatility Forecast Accuracy Analysis",
            height=800,
            showlegend=True
        )

        # Save
        output_path = f"{self.output_dir}/vol_forecast_accuracy.html"
        fig.write_html(output_path)
        print(f"Forecast accuracy plot saved to {output_path}")

    def create_comprehensive_report(self, data_dict: Dict[str, pd.DataFrame],
                                    features_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Create comprehensive GARCH analysis report
        """
        for pair, data in data_dict.items():
            if pair in features_dict:
                self.plot_volatility_comparison(data, pair, features_dict[pair])

        # Multivariate analysis
        if features_dict:
            # Get multivariate features from any pair's features
            first_pair_features = list(features_dict.values())[0]
            mv_cols = [col for col in first_pair_features.columns
                       if 'mv_garch' in col or 'corr_' in col or 'correlation' in col]

            if mv_cols:
                mv_features = first_pair_features[mv_cols]
                self.plot_multivariate_analysis(mv_features, list(data_dict.keys()))

        print(f"✅ All GARCH visualizations saved to {self.output_dir}/")