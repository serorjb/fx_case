"""
Visualization functions for FX options trading system
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class TradingVisualizer:
    """
    Create visualizations for trading results
    """

    def __init__(self, results_dir: str = 'plots'):
        self.results_dir = results_dir

    def plot_equity_curve(self, results_df: pd.DataFrame, title: str = "Equity Curve") -> None:
        """
        Plot equity curve with drawdown
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Equity Curve', 'Drawdown'),
            row_heights=[0.7, 0.3]
        )

        # Equity curve
        fig.add_trace(
            go.Scatter(x=results_df.index, y=results_df['equity'],
                       mode='lines', name='Equity',
                       line=dict(color='blue', width=2)),
            row=1, col=1
        )

        # Calculate drawdown
        rolling_max = results_df['equity'].expanding().max()
        drawdown = (results_df['equity'] - rolling_max) / rolling_max * 100

        # Drawdown
        fig.add_trace(
            go.Scatter(x=results_df.index, y=drawdown,
                       mode='lines', name='Drawdown %',
                       fill='tozeroy',
                       line=dict(color='red', width=1)),
            row=2, col=1
        )

        # Update layout
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

        fig.update_layout(
            title_text=title,
            height=700,
            showlegend=False,
            hovermode='x unified'
        )

        # Save
        fig.write_html(f"{self.results_dir}/equity_curve.html")
        print(f"Equity curve saved to {self.results_dir}/equity_curve.html")

    def plot_rolling_sharpe(self, results_df: pd.DataFrame, window: int = 60) -> None:
        """
        Plot rolling Sharpe ratio
        """
        returns = results_df['returns'].fillna(0)
        rolling_sharpe = returns.rolling(window).apply(
            lambda x: x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(x=results_df.index, y=rolling_sharpe,
                       mode='lines', name=f'{window}-Day Rolling Sharpe',
                       line=dict(color='green', width=2))
        )

        # Add horizontal lines
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_hline(y=1, line_dash="dash", line_color="blue", annotation_text="Sharpe = 1")
        fig.add_hline(y=2, line_dash="dash", line_color="green", annotation_text="Sharpe = 2")

        fig.update_layout(
            title=f"{window}-Day Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio",
            height=500,
            hovermode='x unified'
        )

        fig.write_html(f"{self.results_dir}/rolling_sharpe.html")
        print(f"Rolling Sharpe saved to {self.results_dir}/rolling_sharpe.html")

    def plot_returns_distribution(self, results_df: pd.DataFrame) -> None:
        """
        Plot returns distribution with statistics
        """
        returns = results_df['returns'].fillna(0) * 100  # Convert to percentage

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Returns Distribution', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=returns, nbinsx=50, name='Returns',
                         marker_color='blue', opacity=0.7),
            row=1, col=1
        )

        # Add normal distribution overlay
        from scipy import stats
        x_range = np.linspace(returns.min(), returns.max(), 100)
        normal_dist = stats.norm.pdf(x_range, returns.mean(), returns.std())
        normal_dist = normal_dist * len(returns) * (returns.max() - returns.min()) / 50

        fig.add_trace(
            go.Scatter(x=x_range, y=normal_dist, mode='lines',
                       name='Normal', line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns)

        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                       mode='markers', name='Q-Q Plot',
                       marker=dict(color='blue', size=3)),
            row=1, col=2
        )

        # Add diagonal line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                       mode='lines', name='45° Line',
                       line=dict(color='red', dash='dash')),
            row=1, col=2
        )

        # Update layout
        fig.update_xaxes(title_text="Daily Returns (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=1, col=2)
        fig.update_yaxes(title_text="Sample Quantiles (%)", row=1, col=2)

        fig.update_layout(
            title_text="Returns Analysis",
            height=500,
            showlegend=False
        )

        fig.write_html(f"{self.results_dir}/returns_distribution.html")
        print(f"Returns distribution saved to {self.results_dir}/returns_distribution.html")

    def plot_strategy_performance(self, trades_df: pd.DataFrame) -> None:
        """
        Plot strategy-wise performance
        """
        if len(trades_df) == 0:
            print("No trades to plot")
            return

        # Group by strategy
        strategy_perf = trades_df.groupby('strategy').agg({
            'pnl': ['sum', 'mean', 'count'],
            'days_held': 'mean'
        }).round(2)

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Total P&L by Strategy', 'Average P&L per Trade',
                            'Number of Trades', 'Average Holding Period'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )

        strategies = strategy_perf.index.tolist()

        # Total P&L
        fig.add_trace(
            go.Bar(x=strategies, y=strategy_perf[('pnl', 'sum')],
                   name='Total P&L', marker_color='blue'),
            row=1, col=1
        )

        # Average P&L
        fig.add_trace(
            go.Bar(x=strategies, y=strategy_perf[('pnl', 'mean')],
                   name='Avg P&L', marker_color='green'),
            row=1, col=2
        )

        # Number of trades
        fig.add_trace(
            go.Bar(x=strategies, y=strategy_perf[('pnl', 'count')],
                   name='# Trades', marker_color='orange'),
            row=2, col=1
        )

        # Average holding period
        fig.add_trace(
            go.Bar(x=strategies, y=strategy_perf[('days_held', 'mean')],
                   name='Avg Days', marker_color='red'),
            row=2, col=2
        )

        fig.update_layout(
            title_text="Strategy Performance Breakdown",
            height=700,
            showlegend=False
        )

        fig.write_html(f"{self.results_dir}/strategy_performance.html")
        print(f"Strategy performance saved to {self.results_dir}/strategy_performance.html")

    def plot_volatility_surface(self, data: pd.DataFrame, pair: str, date: pd.Timestamp) -> None:
        """
        Plot volatility surface for a currency pair
        """
        # Extract volatility data
        tenors = ['1W', '2W', '3W', '1M', '2M', '3M', '6M', '9M', '1Y']
        tenor_days = [7, 14, 21, 30, 60, 90, 180, 270, 365]

        # Get ATM vols
        atm_vols = []
        for tenor in tenors:
            col = f'atm_vol_{tenor}'
            if col in data.columns:
                atm_vols.append(data.loc[date, col] * 100 if date in data.index else np.nan)
            else:
                atm_vols.append(np.nan)

        # Get 25-delta vols
        vols_25_call = []
        vols_25_put = []
        for tenor in tenors:
            if f'atm_vol_{tenor}' in data.columns and f'rr_25_{tenor}' in data.columns and f'bf_25_{tenor}' in data.columns:
                atm = data.loc[date, f'atm_vol_{tenor}'] if date in data.index else np.nan
                rr = data.loc[date, f'rr_25_{tenor}'] if date in data.index else np.nan
                bf = data.loc[date, f'bf_25_{tenor}'] if date in data.index else np.nan

                if not pd.isna(atm) and not pd.isna(rr) and not pd.isna(bf):
                    vols_25_call.append((atm + bf + 0.5 * rr) * 100)
                    vols_25_put.append((atm + bf - 0.5 * rr) * 100)
                else:
                    vols_25_call.append(np.nan)
                    vols_25_put.append(np.nan)
            else:
                vols_25_call.append(np.nan)
                vols_25_put.append(np.nan)

        # Create plot
        fig = go.Figure()

        # Add traces
        fig.add_trace(go.Scatter(x=tenor_days, y=vols_25_put,
                                 mode='lines+markers', name='25Δ Put',
                                 line=dict(color='red', width=2)))

        fig.add_trace(go.Scatter(x=tenor_days, y=atm_vols,
                                 mode='lines+markers', name='ATM',
                                 line=dict(color='blue', width=2)))

        fig.add_trace(go.Scatter(x=tenor_days, y=vols_25_call,
                                 mode='lines+markers', name='25Δ Call',
                                 line=dict(color='green', width=2)))

        fig.update_layout(
            title=f"{pair} Volatility Smile - {date.strftime('%Y-%m-%d')}",
            xaxis_title="Days to Expiry",
            yaxis_title="Implied Volatility (%)",
            height=500,
            hovermode='x unified'
        )

        fig.write_html(f"{self.results_dir}/vol_surface_{pair}.html")
        print(f"Volatility surface saved to {self.results_dir}/vol_surface_{pair}.html")

    def create_performance_report(self, performance: Dict, trades_df: pd.DataFrame) -> None:
        """
        Create a comprehensive performance report
        """
        report = []
        report.append("=" * 60)
        report.append("FX OPTIONS TRADING STRATEGY PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("OVERALL PERFORMANCE METRICS:")
        report.append("-" * 30)
        for key, value in performance.items():
            if isinstance(value, float):
                if 'return' in key or 'ratio' in key or 'rate' in key:
                    report.append(f"{key.replace('_', ' ').title()}: {value:.2%}")
                else:
                    report.append(f"{key.replace('_', ' ').title()}: ${value:,.2f}")
            else:
                report.append(f"{key.replace('_', ' ').title()}: {value}")

        if len(trades_df) > 0:
            report.append("")
            report.append("TRADING STATISTICS:")
            report.append("-" * 30)

            # By strategy
            strategy_stats = trades_df.groupby('strategy').agg({
                'pnl': ['sum', 'mean', 'std', 'count']
            })

            report.append("\nBy Strategy:")
            for strategy in strategy_stats.index:
                report.append(f"\n{strategy}:")
                report.append(f"  Total P&L: ${strategy_stats.loc[strategy, ('pnl', 'sum')]:,.2f}")
                report.append(f"  Avg P&L: ${strategy_stats.loc[strategy, ('pnl', 'mean')]:,.2f}")
                report.append(f"  Std P&L: ${strategy_stats.loc[strategy, ('pnl', 'std')]:,.2f}")
                report.append(f"  # Trades: {strategy_stats.loc[strategy, ('pnl', 'count')]:.0f}")

            # By currency pair
            if 'pair' in trades_df.columns:
                pair_stats = trades_df.groupby('pair').agg({
                    'pnl': ['sum', 'mean', 'count']
                })

                report.append("\nBy Currency Pair:")
                for pair in pair_stats.index:
                    report.append(f"\n{pair}:")
                    report.append(f"  Total P&L: ${pair_stats.loc[pair, ('pnl', 'sum')]:,.2f}")
                    report.append(f"  Avg P&L: ${pair_stats.loc[pair, ('pnl', 'mean')]:,.2f}")
                    report.append(f"  # Trades: {pair_stats.loc[pair, ('pnl', 'count')]:.0f}")

        report.append("")
        report.append("=" * 60)

        # Save report
        with open(f"{self.results_dir}/performance_report.txt", 'w') as f:
            f.write('\n'.join(report))

        print(f"Performance report saved to {self.results_dir}/performance_report.txt")