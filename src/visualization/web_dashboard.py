#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Web dashboard for visualizing backtest results.
This module provides a Flask and Dash based web interface to display backtest results.
"""

import os
import json
import pandas as pd
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from flask import Flask, render_template, request, redirect, url_for
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from src.utils.logger import get_logger
from src.config.config import LOGS_DIR

logger = get_logger("WebDashboard")

# Initialize Flask app
server = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'))

# Initialize Dash app
app = dash.Dash(__name__, server=server, url_base_pathname='/dashboard/',
               external_stylesheets=[dbc.themes.BOOTSTRAP])

# Global variables to store backtest results
current_results = None
available_results = []


def scan_backtest_results():
    """Scan for available backtest result files"""
    global available_results
    
    results_dir = LOGS_DIR
    available_results = []
    
    try:
        for file in os.listdir(results_dir):
            if file.startswith('backtest_') and file.endswith('.json'):
                result_path = os.path.join(results_dir, file)
                try:
                    with open(result_path, 'r') as f:
                        result_data = json.load(f)
                    
                    # Extract metadata
                    strategy_name = result_data.get('parameters', {}).get('strategy', 'Unknown')
                    symbols = result_data.get('parameters', {}).get('symbols', [])
                    start_date = result_data.get('parameters', {}).get('start_date', '')
                    end_date = result_data.get('parameters', {}).get('end_date', '')
                    
                    # Format symbols for display
                    symbols_str = ', '.join(symbols) if isinstance(symbols, list) else str(symbols)
                    
                    available_results.append({
                        'filename': file,
                        'path': result_path,
                        'strategy': strategy_name,
                        'symbols': symbols_str,
                        'period': f"{start_date} to {end_date}",
                        'timestamp': os.path.getmtime(result_path)
                    })
                except Exception as e:
                    logger.error(f"Error loading result file {file}: {e}")
        
        # Sort by timestamp (newest first)
        available_results = sorted(available_results, key=lambda x: x['timestamp'], reverse=True)
        
    except Exception as e:
        logger.error(f"Error scanning backtest results: {e}")
    
    return available_results


def load_backtest_result(result_path):
    """Load backtest result from file"""
    global current_results
    
    try:
        with open(result_path, 'r') as f:
            result_data = json.load(f)
        
        # Convert portfolio history to DataFrame
        if 'portfolio_history' in result_data and result_data['portfolio_history']:
            portfolio_history = pd.DataFrame(result_data['portfolio_history'])
            portfolio_history['timestamp'] = pd.to_datetime(portfolio_history['timestamp'])
            result_data['portfolio_history'] = portfolio_history
        
        # Convert trades to DataFrame
        if 'trades' in result_data and result_data['trades']:
            trades = pd.DataFrame(result_data['trades'])
            trades['timestamp'] = pd.to_datetime(trades['timestamp'])
            result_data['trades'] = trades
        
        current_results = result_data
        return True
    
    except Exception as e:
        logger.error(f"Error loading backtest result: {e}")
        return False


def create_equity_curve():
    """Create equity curve figure"""
    if current_results is None or 'portfolio_history' not in current_results:
        return go.Figure()
    
    portfolio_history = current_results['portfolio_history']
    trades = current_results.get('trades')
    
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True, 
                       vertical_spacing=0.1, 
                       subplot_titles=('Portfolio Performance', 'Drawdown'),
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
    
    # Add trades if available
    if trades is not None and len(trades) > 0:
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
    if 'returns' not in portfolio_history.columns:
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
        height=600,
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
    
    return fig


def create_performance_metrics():
    """Create performance metrics figure"""
    if current_results is None or 'metrics' not in current_results:
        return go.Figure()
    
    metrics = current_results['metrics']
    
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
    
    # Create figure
    fig = px.bar(
        df, 
        x='Metric', 
        y='Value',
        title="Performance Metrics",
        color='Metric',
        text='Value'
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title=None,
        yaxis_title='Value',
        template='plotly_white'
    )
    
    # Format text labels
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside'
    )
    
    return fig


def create_trade_analysis():
    """Create trade analysis figure"""
    if current_results is None or 'trades' not in current_results or len(current_results['trades']) == 0:
        return go.Figure()
    
    trades = current_results['trades']
    
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
    
    # 4. Trade Duration (if available)
    if 'duration' in trades.columns:
        # Convert duration to days if it's a timedelta
        if pd.api.types.is_timedelta64_dtype(trades['duration']):
            trades['duration_days'] = trades['duration'].dt.total_seconds() / (24 * 60 * 60)
        else:
            trades['duration_days'] = trades['duration']
        
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
        height=600,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig


# Define the Dash layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("HKStock-Quant 回测结果分析", className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("选择回测结果"),
            dcc.Dropdown(
                id='backtest-selector',
                options=[],
                value=None,
                placeholder="选择回测结果文件"
            ),
            html.Button('刷新列表', id='refresh-button', className="btn btn-primary mt-2")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div(id='backtest-info', className="mt-3")
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("投资组合表现", className="mt-4"),
            dcc.Graph(id='equity-curve')
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("性能指标", className="mt-4"),
            dcc.Graph(id='performance-metrics')
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H4("交易分析", className="mt-4"),
            dcc.Graph(id='trade-analysis')
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Hr(),
            html.P("HKStock-Quant 量化交易系统 © 2025", className="text-center")
        ])
    ])
], fluid=True)


# Define callbacks
@app.callback(
    [Output('backtest-selector', 'options'),
     Output('backtest-selector', 'value')],
    [Input('refresh-button', 'n_clicks')]
)
def update_backtest_list(n_clicks):
    """Update the list of available backtest results"""
    results = scan_backtest_results()
    options = [{'label': f"{r['strategy']} - {r['symbols']} ({r['period']})", 'value': r['path']} for r in results]
    
    # If there are results, select the most recent one
    value = options[0]['value'] if options else None
    
    return options, value


@app.callback(
    [Output('backtest-info', 'children'),
     Output('equity-curve', 'figure'),
     Output('performance-metrics', 'figure'),
     Output('trade-analysis', 'figure')],
    [Input('backtest-selector', 'value')]
)
def update_dashboard(selected_result):
    """Update the dashboard with the selected backtest result"""
    if not selected_result:
        return html.Div("请选择回测结果文件"), go.Figure(), go.Figure(), go.Figure()
    
    # Load the selected result
    success = load_backtest_result(selected_result)
    if not success:
        return html.Div("加载回测结果失败，请检查文件格式"), go.Figure(), go.Figure(), go.Figure()
    
    # Create info card
    parameters = current_results.get('parameters', {})
    metrics = current_results.get('metrics', {})
    
    info_card = dbc.Card(
        dbc.CardBody([
            html.H5("回测信息", className="card-title"),
            html.Div([
                html.P(f"策略: {parameters.get('strategy', 'Unknown')}"),
                html.P(f"交易品种: {parameters.get('symbols', [])}"),
                html.P(f"时间段: {parameters.get('start_date', '')} 至 {parameters.get('end_date', '')}"),
                html.P(f"初始资金: {parameters.get('initial_capital', 0):,.2f}"),
                html.Hr(),
                html.P(f"最终资产: {metrics.get('final_equity', 0):,.2f}"),
                html.P(f"总收益率: {metrics.get('total_return', 0):.2%}"),
                html.P(f"年化收益率: {metrics.get('annual_return', 0):.2%}"),
                html.P(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}"),
                html.P(f"最大回撤: {metrics.get('max_drawdown', 0):.2%}"),
                html.P(f"交易次数: {metrics.get('total_trades', 0)}"),
            ])
        ]),
        className="mb-3"
    )
    
    # Create figures
    equity_curve = create_equity_curve()
    performance_metrics = create_performance_metrics()
    trade_analysis = create_trade_analysis()
    
    return info_card, equity_curve, performance_metrics, trade_analysis


# Flask routes
@server.route('/')
def index():
    """Main page that redirects to the Dash app"""
    return redirect('/dashboard/')


def run_server(host='0.0.0.0', port=8050, debug=False):
    """Run the web server"""
    logger.info(f"Starting web dashboard on http://{host}:{port}/")
    app.run_server(host=host, port=port, debug=debug)


if __name__ == '__main__':
    run_server(debug=True)
