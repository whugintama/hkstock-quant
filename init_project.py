#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Initialization script for the HKStock-Quant project.
This script sets up the project environment, creates necessary directories,
and verifies the installation of required dependencies.
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print(f"Error: Python 3.8 or higher is required. Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible.")
    return True

def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking dependencies...")
    
    required_packages = [
        "pandas", "numpy", "matplotlib", "seaborn", "plotly", "dash",
        "backtesting", "backtrader", "yfinance", "akshare", "pandas-datareader",
        "scikit-learn", "statsmodels", "tqdm", "pyyaml", "python-dotenv",
        "schedule", "pytest", "flask", "flask-restful"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package}")
    
    if missing_packages:
        print("\nMissing packages:")
        for package in missing_packages:
            print(f"  - {package}")
        
        install = input("\nDo you want to install missing packages? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
                print("Packages installed successfully.")
            except subprocess.CalledProcessError:
                print("Error installing packages. Please install them manually.")
                return False
        else:
            print("Please install the missing packages manually.")
            return False
    
    return True

def check_futu_api():
    """Check if Futu API is installed"""
    print("Checking Futu API...")
    
    try:
        importlib.import_module("futu")
        print("✓ Futu API is installed.")
        return True
    except ImportError:
        print("✗ Futu API is not installed.")
        
        install = input("Do you want to install Futu API? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "futu-api"])
                print("Futu API installed successfully.")
                return True
            except subprocess.CalledProcessError:
                print("Error installing Futu API. Please install it manually.")
                return False
        else:
            print("Please install Futu API manually if you plan to use it for live trading.")
            return True  # Return True anyway since Futu API is optional for backtesting

def create_env_file():
    """Create .env file from .env.example"""
    print("Creating .env file...")
    
    env_example_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.example")
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    if os.path.exists(env_path):
        overwrite = input(".env file already exists. Do you want to overwrite it? (y/n): ")
        if overwrite.lower() != 'y':
            print("Skipping .env file creation.")
            return True
    
    try:
        shutil.copy(env_example_path, env_path)
        print(".env file created successfully.")
        print("Please edit the .env file to set your API keys and other configuration.")
        return True
    except Exception as e:
        print(f"Error creating .env file: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("Creating directories...")
    
    directories = [
        "data/raw",
        "data/processed",
        "logs",
        "logs/visualizations",
        "logs/reports",
        "notebooks"
    ]
    
    for directory in directories:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), directory)
        os.makedirs(dir_path, exist_ok=True)
        print(f"✓ {directory}")
    
    return True

def setup_jupyter_notebook():
    """Setup Jupyter notebook"""
    print("Setting up Jupyter notebook...")
    
    try:
        importlib.import_module("notebook")
        print("✓ Jupyter notebook is installed.")
    except ImportError:
        print("✗ Jupyter notebook is not installed.")
        
        install = input("Do you want to install Jupyter notebook? (y/n): ")
        if install.lower() == 'y':
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "notebook"])
                print("Jupyter notebook installed successfully.")
            except subprocess.CalledProcessError:
                print("Error installing Jupyter notebook. Please install it manually.")
                return False
        else:
            print("Skipping Jupyter notebook installation.")
            return True
    
    # Create example notebook
    notebook_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
    notebook_path = os.path.join(notebook_dir, "getting_started.ipynb")
    
    if not os.path.exists(notebook_path):
        print("Creating example notebook...")
        
        example_notebook = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Getting Started with HKStock-Quant\n",
                        "\n",
                        "This notebook demonstrates how to use the HKStock-Quant framework for backtesting and analyzing trading strategies."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import os\n",
                        "import sys\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "# Add the project root directory to the Python path\n",
                        "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath('__file__'))))\n",
                        "\n",
                        "from src.data.data_fetcher import HKStockDataFetcher\n",
                        "from src.strategies.moving_average import MovingAverageStrategy\n",
                        "from src.backtest.engine import BacktestEngine\n",
                        "from src.visualization.visualizer import Visualizer"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Fetch Data\n",
                        "\n",
                        "First, let's fetch some historical data for Hong Kong stocks."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Initialize data fetcher\n",
                        "data_fetcher = HKStockDataFetcher()\n",
                        "\n",
                        "# Fetch data for Tencent\n",
                        "tencent_data = data_fetcher.get_data(\n",
                        "    symbol='0700.HK',\n",
                        "    start_date='2022-01-01',\n",
                        "    end_date='2023-01-01'\n",
                        ")\n",
                        "\n",
                        "# Display the first few rows\n",
                        "tencent_data.head()"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Backtest a Strategy\n",
                        "\n",
                        "Now, let's backtest a moving average strategy."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Initialize backtest engine\n",
                        "engine = BacktestEngine(\n",
                        "    strategy=MovingAverageStrategy,\n",
                        "    start_date='2022-01-01',\n",
                        "    end_date='2023-01-01',\n",
                        "    symbols=['0700.HK', '9988.HK'],\n",
                        "    initial_capital=100000\n",
                        ")\n",
                        "\n",
                        "# Run backtest\n",
                        "results = engine.run(strategy_params={\n",
                        "    'short_window': 20,\n",
                        "    'long_window': 50,\n",
                        "    'ma_type': 'EMA'\n",
                        "})\n",
                        "\n",
                        "# Display metrics\n",
                        "results['metrics']"
                    ]
                },
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## Visualize Results\n",
                        "\n",
                        "Let's visualize the backtest results."
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Initialize visualizer\n",
                        "visualizer = Visualizer()\n",
                        "\n",
                        "# Plot equity curve\n",
                        "fig = visualizer.plot_equity_curve(\n",
                        "    portfolio_history=results['portfolio_history'],\n",
                        "    trades=results['trades'],\n",
                        "    title=\"Moving Average Strategy Backtest\",\n",
                        "    show_trades=True,\n",
                        "    interactive=True\n",
                        ")\n",
                        "\n",
                        "fig"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "codemirror_mode": {
                        "name": "ipython",
                        "version": 3
                    },
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                    "nbconvert_exporter": "python",
                    "pygments_lexer": "ipython3",
                    "version": "3.8.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        try:
            import json
            with open(notebook_path, 'w') as f:
                json.dump(example_notebook, f)
            
            print(f"Example notebook created at {notebook_path}")
        except Exception as e:
            print(f"Error creating example notebook: {e}")
    
    return True

def main():
    """Main function"""
    print("=" * 50)
    print("HKStock-Quant Project Initialization")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check Futu API
    check_futu_api()
    
    # Create directories
    create_directories()
    
    # Create .env file
    create_env_file()
    
    # Setup Jupyter notebook
    setup_jupyter_notebook()
    
    print("\n" + "=" * 50)
    print("Initialization completed successfully!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Edit the .env file to set your API keys and other configuration")
    print("2. Run an example backtest: python examples/backtest_example.py")
    print("3. Explore the Jupyter notebooks in the notebooks directory")
    print("4. For live trading, configure the Futu API in src/config/futu_config.yaml")
    print("\nHappy trading!")

if __name__ == "__main__":
    main()
