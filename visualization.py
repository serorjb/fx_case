"""
===============================================================================
visualization.py - Plotting and visualization functions
===============================================================================
"""
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from config import Config


class Visualizer:
    """Create all visualizations"""

    def __init__(self, config: Config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)

    def plot_equity_curves(self, results: Dict):
        """Plot equity curves for all models"""

        fig = go.Figure()

        for model_name, model_results in results.items():
            if 'equity_curve' not in model_results:
                continue

            equity = model_results['equity_curve']

            fig.add_trace(go.Scatter(
                x=equity.index,
                y=equity['equity'],
                mode='lines',
                name=model_name,
                line=dict(width=2)
            ))

        fig.update_layout(
            title='Model Comparison - Equity Curves',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=600,
            hovermode='x unified'
        )

        fig.write_html(self.config.PLOTS_DIR / 'equity_curves.html')
        print("  Saved equity curves plot")

    def plot_rolling_sortino(self, results: Dict):
        """Plot rolling Sortino ratio for all models"""

        fig = go.Figure()

        for model_name, model_results in results.items():
            if 'equity_curve' not in model_results:
                continue

            returns = model_results['equity_curve']['returns'].dropna()

            # Calculate rolling Sortino (60-day window)
            window = 60
            rolling_sortino = []

            for i in range(window, len(returns)):
                window_returns = returns.iloc[i - window:i]
                downside_returns = window_returns[window_returns < 0]

                if len(downside_returns) > 0:
                    sortino = (window_returns.mean() / downside_returns.std()) * np.sqrt(252)
                else:
                    sortino = 0

                rolling_sortino.append(sortino)

            dates = returns.index[window:]

            fig.add_trace(go.Scatter(
                x=dates,
                y=rolling_sortino,
                mode='lines',
                name=model_name,
                line=dict(width=2)
            ))

        fig.update_layout(
            title='Rolling Sortino Ratio (60-day)',
            xaxis_title='Date',
            yaxis_title='Sortino Ratio',
            height=600,
            hovermode='x unified'
        )

        fig.write_html(self.config.PLOTS_DIR / 'rolling_sortino.html')
        print("  Saved rolling Sortino plot")

    def plot_greeks_evolution(self, backtesters: Dict):
        """Plot Greeks over time"""

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Delta', 'Gamma', 'Vega', 'Theta')
        )

        for model_name, backtester in backtesters.items():
            if not hasattr(backtester, 'greeks_history'):
                continue

            greeks_df = pd.DataFrame(backtester.greeks_history)
            if greeks_df.empty:
                continue

            # Delta
            fig.add_trace(
                go.Scatter(x=greeks_df['date'], y=greeks_df['delta'],
                           mode='lines', name=f'{model_name}'),
                row=1, col=1
            )

            # Gamma
            fig.add_trace(
                go.Scatter(x=greeks_df['date'], y=greeks_df['gamma'],
                           mode='lines', name=f'{model_name}', showlegend=False),
                row=1, col=2
            )

            # Vega
            fig.add_trace(
                go.Scatter(x=greeks_df['date'], y=greeks_df['vega'],
                           mode='lines', name=f'{model_name}', showlegend=False),
                row=2, col=1
            )

            # Theta
            fig.add_trace(
                go.Scatter(x=greeks_df['date'], y=greeks_df['theta'],
                           mode='lines', name=f'{model_name}', showlegend=False),
                row=2, col=2
            )

        fig.update_layout(height=800, title='Portfolio Greeks Evolution')
        fig.write_html(self.config.PLOTS_DIR / 'greeks_evolution.html')
        print("  Saved Greeks evolution plot")

    def plot_margin_usage(self, results: Dict):
        """Plot margin usage over time"""

        fig = go.Figure()

        for model_name, model_results in results.items():
            if 'margin_history' not in model_results:
                continue

            margin_df = model_results['margin_history']

            fig.add_trace(go.Scatter(
                x=margin_df.index,
                y=margin_df['margin_ratio'] * 100,
                mode='lines',
                name=model_name,
                line=dict(width=2)
            ))

        # Add max leverage line
        fig.add_hline(
            y=self.config.MAX_LEVERAGE * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Max Leverage"
        )

        fig.update_layout(
            title='Margin Usage Over Time',
            xaxis_title='Date',
            yaxis_title='Margin Usage (%)',
            height=600,
            hovermode='x unified'
        )

        fig.write_html(self.config.PLOTS_DIR / 'margin_usage.html')
        print("  Saved margin usage plot")

    def plot_volatility_surfaces(self, data: Dict):
        """Plot 3D volatility surfaces"""

        for pair in self.config.CURRENCY_PAIRS[:1]:  # Plot first pair as example
            if pair not in data:
                continue

            pair_data = data[pair]

            # Get last 60 days
            recent_data = pair_data.iloc[-60:]

            # Create mesh grid
            dates = list(range(len(recent_data)))
            tenors = [self.config.TENOR_YEARS[t] for t in self.config.TENORS]

            X, Y = np.meshgrid(dates, tenors)
            Z = np.zeros_like(X)

            # Fill volatility surface
            for i, date_idx in enumerate(dates):
                for j, tenor in enumerate(self.config.TENORS):
                    col = f'atm_vol_{tenor}'
                    if col in recent_data.columns:
                        Z[j, i] = recent_data.iloc[date_idx][col] * 100

            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

            fig.update_layout(
                title=f'{pair} Volatility Surface',
                scene=dict(
                    xaxis_title='Days',
                    yaxis_title='Tenor (Years)',
                    zaxis_title='Implied Vol (%)'
                ),
                height=700
            )

            fig.write_html(self.config.PLOTS_DIR / f'vol_surface_{pair}.html')
            print(f"  Saved volatility surface for {pair}")

    def plot_model_comparison(self, results: Dict):
        """Create comparison table of model performance"""

        metrics_data = []

        for model_name, model_results in results.items():
            if 'metrics' not in model_results:
                continue

            metrics = model_results['metrics']
            metrics_data.append({
                'Model': model_name,
                'Total Return': f"{metrics.get('total_return', 0):.2%}",
                'Sharpe Ratio': f"{metrics.get('sharpe_ratio', 0):.3f}",
                'Sortino Ratio': f"{metrics.get('sortino_ratio', 0):.3f}",
                'Max Drawdown': f"{metrics.get('max_drawdown', 0):.2%}",
                'Win Rate': f"{metrics.get('win_rate', 0):.1%}",
                'Total Trades': metrics.get('total_trades', 0)
            })

        if metrics_data:
            df = pd.DataFrame(metrics_data)

            # Create table figure
            fig = go.Figure(data=[go.Table(
                header=dict(values=list(df.columns),
                            fill_color='paleturquoise',
                            align='left'),
                cells=dict(values=[df[col] for col in df.columns],
                           fill_color='lavender',
                           align='left'))
            ])

            fig.update_layout(title='Model Performance Comparison', height=400)
            fig.write_html(self.config.PLOTS_DIR / 'model_comparison.html')
            print("  Saved model comparison table")

    def plot_risk_metrics(self, results: Dict):
        """Plot risk metrics over time"""

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Value at Risk (95%)', 'Maximum Drawdown'),
            shared_xaxes=True
        )

        for model_name, model_results in results.items():
            if 'equity_curve' not in model_results:
                continue

            equity = model_results['equity_curve']
            returns = equity['returns'].dropna()

            # Calculate rolling VaR
            window = 60
            rolling_var = returns.rolling(window).quantile(0.05) * np.sqrt(252)

            fig.add_trace(
                go.Scatter(x=rolling_var.index, y=rolling_var * 100,
                           mode='lines', name=model_name),
                row=1, col=1
            )

            # Calculate rolling drawdown
            rolling_max = equity['equity'].expanding().max()
            drawdown = (equity['equity'] - rolling_max) / rolling_max * 100

            fig.add_trace(
                go.Scatter(x=drawdown.index, y=drawdown,
                           mode='lines', name=model_name, showlegend=False),
                row=2, col=1
            )

        fig.update_layout(height=800, title='Risk Metrics Over Time')
        fig.write_html(self.config.PLOTS_DIR / 'risk_metrics.html')
        print("  Saved risk metrics plot")

    def plot_pnl_attribution(self, backtesters: Dict):
        """Plot P&L attribution by components"""

        fig = go.Figure()

        for model_name, backtester in backtesters.items():
            if not hasattr(backtester, 'trade_history'):
                continue

            trades = backtester.trade_history
            if not trades:
                continue

            # Group P&L by tenor
            tenor_pnl = {}
            for trade in trades:
                if hasattr(trade, 'realized_pnl') and trade.realized_pnl is not None:
                    tenor = getattr(trade, 'tenor', 'Unknown')
                    if tenor not in tenor_pnl:
                        tenor_pnl[tenor] = 0
                    tenor_pnl[tenor] += trade.realized_pnl

            if tenor_pnl:
                fig.add_trace(go.Bar(
                    x=list(tenor_pnl.keys()),
                    y=list(tenor_pnl.values()),
                    name=model_name
                ))

        fig.update_layout(
            title='P&L Attribution by Tenor',
            xaxis_title='Tenor',
            yaxis_title='Total P&L ($)',
            height=600
        )

        fig.write_html(self.config.PLOTS_DIR / 'pnl_attribution.html')
        print("  Saved P&L attribution plot")