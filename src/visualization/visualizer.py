import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from pathlib import Path

from src.utils.logger import get_logger
from src.config.config import LOGS_DIR

class Visualizer:
    """
    Visualization tools for trading strategies and backtest results
    """
    def __init__(self, save_dir=None):
        """
        Initialize the visualizer
        
        Args:
            save_dir (str): Directory to save visualizations
        """
        self.logger = get_logger("Visualizer")
        self.save_dir = save_dir or os.path.join(LOGS_DIR, "visualizations")
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
    
    def plot_equity_curve(self, portfolio_history, trades=None, title="Portfolio Performance", 
                          show_trades=True, interactive=True, save_path=None):
        """
        Plot equity curve with optional trade markers
        
        Args:
            portfolio_history (pd.DataFrame): Portfolio history dataframe
            trades (pd.DataFrame): Trades dataframe
            title (str): Chart title
            show_trades (bool): Whether to show trades on the chart
            interactive (bool): Whether to use plotly for interactive charts
            save_path (str): Path to save the visualization
            
        Returns:
            plotly.graph_objects.Figure or matplotlib.figure.Figure: Figure object
        """
        if interactive:
            return self._plot_equity_curve_plotly(portfolio_history, trades, title, show_trades, save_path)
        else:
            return self._plot_equity_curve_matplotlib(portfolio_history, trades, title, show_trades, save_path)
    
    def _plot_equity_curve_plotly(self, portfolio_history, trades=None, title="Portfolio Performance", 
                                 show_trades=True, save_path=None):
        """Plot equity curve using plotly"""
        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True, 
                           vertical_spacing=0.1, 
                           subplot_titles=(title, 'Drawdown'),
                           row_heights=[0.7, 0.3])
        
        # Add equity curve
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'], 
                y=portfolio_history['equity'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Add trades if requested
        if show_trades and trades is not None and len(trades) > 0:
            # Add buy trades
            buy_trades = trades[trades['type'] == 'buy']
            if len(buy_trades) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_trades['timestamp'],
                        y=buy_trades['price'],
                        mode='markers',
                        name='Buy',
                        marker=dict(
                            symbol='triangle-up',
                            size=10,
                            color='green',
                            line=dict(width=1, color='darkgreen')
                        )
                    ),
                    row=1, col=1
                )
            
            # Add sell trades
            sell_trades = trades[trades['type'] == 'sell']
            if len(sell_trades) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_trades['timestamp'],
                        y=sell_trades['price'],
                        mode='markers',
                        name='Sell',
                        marker=dict(
                            symbol='triangle-down',
                            size=10,
                            color='red',
                            line=dict(width=1, color='darkred')
                        )
                    ),
                    row=1, col=1
                )
        
        # Calculate and plot drawdown
        portfolio_history['returns'] = portfolio_history['equity'].pct_change().fillna(0)
        cumulative_returns = (1 + portfolio_history['returns']).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_history['timestamp'],
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Portfolio Value", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", row=2, col=1)
        
        # Save figure if requested
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Interactive visualization saved to {save_path}")
        
        return fig
    
    def _plot_equity_curve_matplotlib(self, portfolio_history, trades=None, title="Portfolio Performance", 
                                     show_trades=True, save_path=None):
        """Plot equity curve using matplotlib"""
        # Set up the figure
        plt.figure(figsize=(14, 10))
        
        # Set style
        sns.set_style('whitegrid')
        
        # Plot equity curve
        plt.subplot(2, 1, 1)
        plt.plot(portfolio_history['timestamp'], portfolio_history['equity'], label='Portfolio Value')
        
        # Plot buy/sell signals if requested
        if show_trades and trades is not None and len(trades) > 0:
            # Plot buy signals
            buy_trades = trades[trades['type'] == 'buy']
            if len(buy_trades) > 0:
                plt.scatter(buy_trades['timestamp'], buy_trades['price'], 
                           marker='^', color='green', s=100, label='Buy')
            
            # Plot sell signals
            sell_trades = trades[trades['type'] == 'sell']
            if len(sell_trades) > 0:
                plt.scatter(sell_trades['timestamp'], sell_trades['price'], 
                           marker='v', color='red', s=100, label='Sell')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.legend()
        plt.grid(True)
        
        # Plot drawdown
        plt.subplot(2, 1, 2)
        portfolio_history['returns'] = portfolio_history['equity'].pct_change().fillna(0)
        cumulative_returns = (1 + portfolio_history['returns']).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        plt.fill_between(portfolio_history['timestamp'], drawdown, 0, color='red', alpha=0.3)
        plt.title('Drawdown')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save figure if requested
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Visualization saved to {save_path}")
        
        return plt.gcf()
    
    def plot_performance_metrics(self, metrics, title="Performance Metrics", save_path=None):
        """
        Plot performance metrics as a bar chart
        
        Args:
            metrics (dict): Dictionary of performance metrics
            title (str): Chart title
            save_path (str): Path to save the visualization
            
        Returns:
            plotly.graph_objects.Figure: Figure object
        """
        # Filter metrics for visualization
        viz_metrics = {
            'Total Return': metrics.get('total_return', 0) * 100,
            'Annual Return': metrics.get('annual_return', 0) * 100,
            'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
            'Max Drawdown': metrics.get('max_drawdown', 0) * 100 * -1,  # Convert to positive for visualization
            'Win Rate': metrics.get('win_rate', 0) * 100,
            'Profit Factor': metrics.get('profit_factor', 0)
        }
        
        # Create dataframe
        df = pd.DataFrame({
            'Metric': list(viz_metrics.keys()),
            'Value': list(viz_metrics.values())
        })
        
        # Create color map
        colors = ['green' if v > 0 else 'red' for v in df['Value']]
        colors[2] = 'blue'  # Sharpe Ratio
        colors[5] = 'purple'  # Profit Factor
        
        # Create figure
        fig = px.bar(
            df, 
            x='Metric', 
            y='Value',
            title=title,
            color='Metric',
            text='Value'
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title=None,
            yaxis_title='Value',
            template='plotly_white'
        )
        
        # Format text labels
        fig.update_traces(
            texttemplate='%{text:.2f}',
            textposition='outside'
        )
        
        # Save figure if requested
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Performance metrics visualization saved to {save_path}")
        
        return fig
    
    def plot_trade_analysis(self, trades, title="Trade Analysis", save_path=None):
        """
        Plot trade analysis
        
        Args:
            trades (pd.DataFrame): Trades dataframe
            title (str): Chart title
            save_path (str): Path to save the visualization
            
        Returns:
            plotly.graph_objects.Figure: Figure object
        """
        if trades is None or len(trades) == 0:
            self.logger.warning("No trades available for visualization")
            return None
        
        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Profit Distribution', 'Profit by Symbol', 
                           'Cumulative Profit', 'Trade Duration'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                  [{'type': 'scatter'}, {'type': 'histogram'}]]
        )
        
        # Prepare data
        trades['profit'] = trades['profit'].fillna(0)
        trades['duration'] = pd.to_timedelta(0)  # Initialize duration
        
        # Calculate trade duration and cumulative profit
        trades = trades.sort_values('timestamp')
        trades['cumulative_profit'] = trades['profit'].cumsum()
        
        # Group trades by symbol and type for buy/sell pairs
        for symbol in trades['symbol'].unique():
            symbol_trades = trades[trades['symbol'] == symbol].copy()
            
            # Match buy and sell trades
            buy_trades = symbol_trades[symbol_trades['type'] == 'buy'].copy()
            sell_trades = symbol_trades[symbol_trades['type'] == 'sell'].copy()
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Ensure we have matching pairs
                min_trades = min(len(buy_trades), len(sell_trades))
                buy_trades = buy_trades.iloc[:min_trades].reset_index(drop=True)
                sell_trades = sell_trades.iloc[:min_trades].reset_index(drop=True)
                
                # Calculate duration
                for i in range(min_trades):
                    buy_time = buy_trades.iloc[i]['timestamp']
                    sell_time = sell_trades.iloc[i]['timestamp']
                    duration = sell_time - buy_time
                    
                    # Update duration in the original dataframe
                    sell_idx = sell_trades.iloc[i].name
                    trades.loc[sell_idx, 'duration'] = duration
        
        # Convert duration to days
        trades['duration_days'] = trades['duration'].dt.total_seconds() / (24 * 60 * 60)
        
        # 1. Profit Distribution
        fig.add_trace(
            go.Histogram(
                x=trades['profit'],
                name='Profit Distribution',
                marker_color='blue',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # 2. Profit by Symbol
        symbol_profit = trades.groupby('symbol')['profit'].sum().reset_index()
        symbol_profit = symbol_profit.sort_values('profit', ascending=False)
        
        fig.add_trace(
            go.Bar(
                x=symbol_profit['symbol'],
                y=symbol_profit['profit'],
                name='Profit by Symbol',
                marker_color=['green' if x > 0 else 'red' for x in symbol_profit['profit']]
            ),
            row=1, col=2
        )
        
        # 3. Cumulative Profit
        fig.add_trace(
            go.Scatter(
                x=trades['timestamp'],
                y=trades['cumulative_profit'],
                mode='lines',
                name='Cumulative Profit',
                line=dict(color='green', width=2)
            ),
            row=2, col=1
        )
        
        # 4. Trade Duration
        fig.add_trace(
            go.Histogram(
                x=trades['duration_days'],
                name='Trade Duration (days)',
                marker_color='purple',
                opacity=0.7
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text=title,
            showlegend=False,
            template='plotly_white'
        )
        
        # Update axes
        fig.update_xaxes(title_text='Profit', row=1, col=1)
        fig.update_yaxes(title_text='Count', row=1, col=1)
        
        fig.update_xaxes(title_text='Symbol', row=1, col=2)
        fig.update_yaxes(title_text='Total Profit', row=1, col=2)
        
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Cumulative Profit', row=2, col=1)
        
        fig.update_xaxes(title_text='Duration (days)', row=2, col=2)
        fig.update_yaxes(title_text='Count', row=2, col=2)
        
        # Save figure if requested
        if save_path:
            fig.write_html(save_path)
            self.logger.info(f"Trade analysis visualization saved to {save_path}")
        
        return fig
    
    def create_dashboard(self, backtest_results, title="Backtest Dashboard", save_path=None):
        """
        Create a comprehensive dashboard for backtest results
        
        Args:
            backtest_results (dict): Dictionary containing backtest results
            title (str): Dashboard title
            save_path (str): Path to save the dashboard
            
        Returns:
            str: Path to saved dashboard
        """
        # Extract data from backtest results
        portfolio_history = backtest_results.get('portfolio_history')
        trades = backtest_results.get('trades')
        metrics = backtest_results.get('metrics')
        
        if portfolio_history is None:
            self.logger.warning("No portfolio history available for dashboard")
            return None
        
        # Convert to DataFrame if necessary
        if not isinstance(portfolio_history, pd.DataFrame):
            portfolio_history = pd.DataFrame(portfolio_history)
            portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
        
        if trades is not None and not isinstance(trades, pd.DataFrame):
            trades = pd.DataFrame(trades)
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        
        # Create dashboard HTML
        dashboard_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                }}
                .metrics-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin-bottom: 30px;
                }}
                .metric-box {{
                    width: 30%;
                    padding: 15px;
                    margin-bottom: 15px;
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    box-shadow: 0 0 5px rgba(0,0,0,0.05);
                }}
                .metric-title {{
                    font-size: 14px;
                    color: #666;
                    margin-bottom: 5px;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                }}
                .positive {{
                    color: green;
                }}
                .negative {{
                    color: red;
                }}
                .chart-container {{
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                </div>
                
                <div class="metrics-container">
        """
        
        # Add metrics
        if metrics:
            metrics_data = [
                {"name": "Total Return", "value": metrics.get('total_return', 0), "format": "percent"},
                {"name": "Annual Return", "value": metrics.get('annual_return', 0), "format": "percent"},
                {"name": "Sharpe Ratio", "value": metrics.get('sharpe_ratio', 0), "format": "decimal"},
                {"name": "Max Drawdown", "value": metrics.get('max_drawdown', 0), "format": "percent"},
                {"name": "Win Rate", "value": metrics.get('win_rate', 0), "format": "percent"},
                {"name": "Profit Factor", "value": metrics.get('profit_factor', 0), "format": "decimal"},
                {"name": "Total Trades", "value": metrics.get('total_trades', 0), "format": "integer"},
                {"name": "Final Equity", "value": metrics.get('final_equity', 0), "format": "currency"},
                {"name": "Volatility", "value": metrics.get('volatility', 0), "format": "percent"}
            ]
            
            for metric in metrics_data:
                value = metric["value"]
                formatted_value = ""
                
                if metric["format"] == "percent":
                    formatted_value = f"{value:.2%}"
                    class_name = "positive" if value > 0 else "negative"
                elif metric["format"] == "decimal":
                    formatted_value = f"{value:.2f}"
                    class_name = "positive" if value > 0 else "negative"
                elif metric["format"] == "integer":
                    formatted_value = f"{int(value)}"
                    class_name = ""
                elif metric["format"] == "currency":
                    formatted_value = f"${value:,.2f}"
                    class_name = "positive" if value > 0 else "negative"
                
                dashboard_html += f"""
                <div class="metric-box">
                    <div class="metric-title">{metric["name"]}</div>
                    <div class="metric-value {class_name}">{formatted_value}</div>
                </div>
                """
        
        dashboard_html += """
                </div>
                
                <div class="chart-container">
                    <h2>Portfolio Performance</h2>
                    <div id="equity-chart" style="width:100%; height:500px;"></div>
                </div>
                
                <div class="chart-container">
                    <h2>Drawdown</h2>
                    <div id="drawdown-chart" style="width:100%; height:300px;"></div>
                </div>
        """
        
        if trades is not None and len(trades) > 0:
            dashboard_html += """
                <div class="chart-container">
                    <h2>Trade Analysis</h2>
                    <div id="trade-chart" style="width:100%; height:400px;"></div>
                </div>
            """
        
        dashboard_html += """
            </div>
            
            <script>
        """
        
        # Add equity chart
        equity_data = {
            'x': portfolio_history['timestamp'].tolist(),
            'y': portfolio_history['equity'].tolist(),
            'type': 'scatter',
            'mode': 'lines',
            'name': 'Portfolio Value',
            'line': {'color': 'blue', 'width': 2}
        }
        
        dashboard_html += f"""
        var equityData = [{equity_data}];
        
        var equityLayout = {{
            title: 'Portfolio Equity Curve',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Portfolio Value'}},
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('equity-chart', equityData, equityLayout);
        """
        
        # Add drawdown chart
        portfolio_history['returns'] = portfolio_history['equity'].pct_change().fillna(0)
        cumulative_returns = (1 + portfolio_history['returns']).cumprod()
        drawdown = 1 - cumulative_returns / cumulative_returns.cummax()
        
        drawdown_data = {
            'x': portfolio_history['timestamp'].tolist(),
            'y': drawdown.tolist(),
            'type': 'scatter',
            'fill': 'tozeroy',
            'name': 'Drawdown',
            'line': {'color': 'red', 'width': 1}
        }
        
        dashboard_html += f"""
        var drawdownData = [{drawdown_data}];
        
        var drawdownLayout = {{
            title: 'Drawdown',
            xaxis: {{title: 'Date'}},
            yaxis: {{title: 'Drawdown'}},
            template: 'plotly_white'
        }};
        
        Plotly.newPlot('drawdown-chart', drawdownData, drawdownLayout);
        """
        
        # Add trade analysis if available
        if trades is not None and len(trades) > 0:
            trades['profit'] = trades['profit'].fillna(0)
            
            # Profit by symbol
            symbol_profit = trades.groupby('symbol')['profit'].sum().reset_index()
            
            trade_data = {
                'x': symbol_profit['symbol'].tolist(),
                'y': symbol_profit['profit'].tolist(),
                'type': 'bar',
                'name': 'Profit by Symbol',
                'marker': {
                    'color': ['green' if x > 0 else 'red' for x in symbol_profit['profit'].tolist()]
                }
            }
            
            dashboard_html += f"""
            var tradeData = [{trade_data}];
            
            var tradeLayout = {{
                title: 'Profit by Symbol',
                xaxis: {{title: 'Symbol'}},
                yaxis: {{title: 'Profit'}},
                template: 'plotly_white'
            }};
            
            Plotly.newPlot('trade-chart', tradeData, tradeLayout);
            """
        
        dashboard_html += """
            </script>
        </body>
        </html>
        """
        
        # Save dashboard
        if save_path is None:
            save_path = os.path.join(self.save_dir, "dashboard.html")
        
        with open(save_path, 'w') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Dashboard saved to {save_path}")
        return save_path
