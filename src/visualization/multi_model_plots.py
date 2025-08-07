# src/visualization/multi_model_plots.py
"""
Visualization for comparing multiple model strategies
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import seaborn as sns


class MultiModelVisualizer:
    """
    Create visualizations comparing multiple model strategies
    """

    def __init__(self, output_dir: str = 'plots'):
        self.output_dir = output_dir
        self.colors = {
            'Model_GK': '#1f77b4',  # Blue
            'Model_GK_GARCH': '#ff7f0e',  # Orange
            'Model_GVV': '#2ca02c',  # Green
            'Model_SABR': '#d62728',  # Red
            'ML_USDJPY': '#9467bd',  # Purple
            'ML_GBPNZD': '#8c564b',  # Brown
            'ML_USDCAD': '#e377c2',  # Pink
            'VolatilityArbitrage': '#7f7f7f',  # Gray
            'CarryToVol': '#bcbd22',  # Yellow-green
            'Combined': '#17becf'  # Cyan
        }

    def plot_multi_strategy_equity_curves(self, all_results: Dict) -> None:
        """
        Plot equity curves for all strategies on same chart
        """
        fig = go.Figure()

        # Add trace for each strategy
        for strategy_name, results in all_results.items():
            if len(results.get('results', [])) > 0:
                equity = results['results']['equity']

                fig.add_trace(go.Scatter(
                    x=equity.index,
                    y=equity.values,
                    mode='lines',
                    name=strategy_name,
                    line=dict(
                        color=self.colors.get(strategy_name, '#000000'),
                        width=2
                    ),
                    hovertemplate='%{y:,.0f}'
                ))

        # Add initial capital line
        if all_results:
            first_result = list(all_results.values())[0]
            if len(first_result.get('results', [])) > 0:
                dates = first_result['results'].index
                initial_capital = 10000000  # From settings

                fig.add_trace(go.Scatter(
                    x=dates,
                    y=[initial_capital] * len(dates),
                    mode='lines',
                    name='Initial Capital',
                    line=dict(color='black', width=1, dash='dash'),
                    hovertemplate='%{y:,.0f}'
                ))

        fig.update_layout(
            title='Multi-Model Strategy Comparison - Equity Curves',
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            height=600,
            hovermode='x unified',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        fig.write_html(f"{self.output_dir}/multi_model_equity_curves.html")
        print(f"  Saved to {self.output_dir}/multi_model_equity_curves.html")

    def plot_performance_comparison(self, all_results: Dict) -> None:
        """
        Create performance comparison charts
        """
        # Collect metrics for each strategy
        metrics_data = []

        for strategy_name, results in all_results.items():
            perf = results.get('performance', {})

            metrics_data.append({
                'Strategy': strategy_name,
                'Total Return (%)': perf.get('total_return', 0) * 100,
                'Sharpe Ratio': perf.get('sharpe_ratio', 0),
                'Max Drawdown (%)': abs(perf.get('max_drawdown', 0)) * 100,
                'Win Rate (%)': perf.get('win_rate', 0) * 100,
                'Total Trades': perf.get('total_trades', 0)
            })

        if not metrics_data:
            return

        metrics_df = pd.DataFrame(metrics_data)

        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'Total Return', 'Sharpe Ratio', 'Max Drawdown',
                'Win Rate', 'Number of Trades', 'Return vs Risk'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}]]
        )

        # Sort strategies by return for consistent ordering
        metrics_df = metrics_df.sort_values('Total Return (%)', ascending=False)
        strategies = metrics_df['Strategy'].tolist()

        # 1. Total Return
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=metrics_df['Total Return (%)'],
                marker_color=[self.colors.get(s, '#000000') for s in strategies],
                text=metrics_df['Total Return (%)'].round(2),
                textposition='auto',
            ),
            row=1, col=1
        )

        # 2. Sharpe Ratio
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=metrics_df['Sharpe Ratio'],
                marker_color=[self.colors.get(s, '#000000') for s in strategies],
                text=metrics_df['Sharpe Ratio'].round(2),
                textposition='auto',
            ),
            row=1, col=2
        )

        # 3. Max Drawdown
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=metrics_df['Max Drawdown (%)'],
                marker_color=[self.colors.get(s, '#000000') for s in strategies],
                text=metrics_df['Max Drawdown (%)'].round(2),
                textposition='auto',
            ),
            row=1, col=3
        )

        # 4. Win Rate
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=metrics_df['Win Rate (%)'],
                marker_color=[self.colors.get(s, '#000000') for s in strategies],
                text=metrics_df['Win Rate (%)'].round(1),
                textposition='auto',
            ),
            row=2, col=1
        )

        # 5. Number of Trades
        fig.add_trace(
            go.Bar(
                x=strategies,
                y=metrics_df['Total Trades'],
                marker_color=[self.colors.get(s, '#000000') for s in strategies],
                text=metrics_df['Total Trades'],
                textposition='auto',
            ),
            row=2, col=2
        )

        # 6. Return vs Risk Scatter
        fig.add_trace(
            go.Scatter(
                x=metrics_df['Max Drawdown (%)'],
                y=metrics_df['Total Return (%)'],
                mode='markers+text',
                marker=dict(
                    size=metrics_df['Total Trades'] * 2 + 10,  # Size by number of trades
                    color=[self.colors.get(s, '#000000') for s in strategies],
                ),
                text=strategies,
                textposition='top center',
            ),
            row=2, col=3
        )

        # Update axes
        fig.update_xaxes(title_text="", row=1, col=1, tickangle=45)
        fig.update_xaxes(title_text="", row=1, col=2, tickangle=45)
        fig.update_xaxes(title_text="", row=1, col=3, tickangle=45)
        fig.update_xaxes(title_text="", row=2, col=1, tickangle=45)
        fig.update_xaxes(title_text="", row=2, col=2, tickangle=45)
        fig.update_xaxes(title_text="Risk (Max DD %)", row=2, col=3)

        fig.update_yaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Sharpe", row=1, col=2)
        fig.update_yaxes(title_text="DD (%)", row=1, col=3)
        fig.update_yaxes(title_text="Win Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="# Trades", row=2, col=2)
        fig.update_yaxes(title_text="Return (%)", row=2, col=3)

        fig.update_layout(
            title='Multi-Model Performance Comparison',
            height=800,
            showlegend=False
        )

        fig.write_html(f"{self.output_dir}/multi_model_performance.html")
        print(f"  Saved to {self.output_dir}/multi_model_performance.html")

    def plot_model_mispricings(self, all_results: Dict) -> None:
        """
        Plot distribution of mispricings for each model
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Mispricing Distribution by Model',
                'Average Mispricing Over Time',
                'Trading Signals by Model',
                'Cumulative P&L by Model'
            ]
        )

        row_idx = 1
        col_idx = 1

        for strategy_name, results in all_results.items():
            if 'Model_' not in strategy_name:
                continue

            # Get trades for this model
            trades = results.get('trades', pd.DataFrame())

            if len(trades) > 0 and 'pnl' in trades.columns:
                # Histogram of P&L
                fig.add_trace(
                    go.Histogram(
                        x=trades['pnl'],
                        name=strategy_name,
                        opacity=0.7,
                        marker_color=self.colors.get(strategy_name, '#000000'),
                        nbinsx=20
                    ),
                    row=1, col=1
                )

                # Cumulative P&L over time
                if 'exit_date' in trades.columns:
                    trades_sorted = trades.sort_values('exit_date')
                    trades_sorted['cumulative_pnl'] = trades_sorted['pnl'].cumsum()

                    fig.add_trace(
                        go.Scatter(
                            x=trades_sorted['exit_date'],
                            y=trades_sorted['cumulative_pnl'],
                            mode='lines',
                            name=strategy_name,
                            line=dict(
                                color=self.colors.get(strategy_name, '#000000'),
                                width=2
                            )
                        ),
                        row=2, col=2
                    )

        # Update layout
        fig.update_xaxes(title_text="P&L ($)", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=2, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_yaxes(title_text="Cumulative P&L ($)", row=2, col=2)

        fig.update_layout(
            title='Model Mispricing Analysis',
            height=700,
            showlegend=True
        )

        fig.write_html(f"{self.output_dir}/model_mispricings.html")
        print(f"  Saved to {self.output_dir}/model_mispricings.html")

    def plot_strategy_correlations(self, all_results: Dict) -> None:
        """
        Plot correlation matrix of strategy returns
        """
        # Collect returns for each strategy
        returns_dict = {}

        for strategy_name, results in all_results.items():
            if len(results.get('results', [])) > 0:
                returns = results['results'].get('returns')
                if returns is not None:
                    returns_dict[strategy_name] = returns

        if len(returns_dict) < 2:
            return

        # Create returns DataFrame
        returns_df = pd.DataFrame(returns_dict)

        # Calculate correlation matrix
        corr_matrix = returns_df.corr()

        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))

        fig.update_layout(
            title='Strategy Return Correlations',
            height=600,
            xaxis=dict(tickangle=45),
            yaxis=dict(tickangle=0)
        )

        fig.write_html(f"{self.output_dir}/strategy_correlations.html")
        print(f"  Saved to {self.output_dir}/strategy_correlations.html")

    def create_performance_summary_table(self, all_results: Dict) -> pd.DataFrame:
        """
        Create summary table of all strategies
        """
        summary_data = []

        for strategy_name, results in all_results.items():
            perf = results.get('performance', {})
            trades = results.get('trades', pd.DataFrame())

            summary_data.append({
                'Strategy': strategy_name,
                'Total Return': f"{perf.get('total_return', 0):.2%}",
                'Ann. Return': f"{perf.get('annualized_return', 0):.2%}",
                'Sharpe': f"{perf.get('sharpe_ratio', 0):.2f}",
                'Sortino': f"{perf.get('sortino_ratio', 0):.2f}",
                'Max DD': f"{perf.get('max_drawdown', 0):.2%}",
                'Trades': perf.get('total_trades', 0),
                'Win Rate': f"{perf.get('win_rate', 0):.1%}",
                'Avg Win': f"${perf.get('avg_win', 0):,.0f}",
                'Avg Loss': f"${perf.get('avg_loss', 0):,.0f}"
            })

        summary_df = pd.DataFrame(summary_data)

        # Save to CSV
        summary_df.to_csv(f"{self.output_dir}/strategy_summary.csv", index=False)
        print(f"  Saved summary to {self.output_dir}/strategy_summary.csv")

        return summary_df