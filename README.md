# Reinforcement Learning for Stocks

This repository is an early-stage project for exploring financial data and
reinforcement learning. At present, it contains a data-processing notebook and
an installable Python package namespace. It does not yet contain a working
trading environment, an RL agent, or a training pipeline.

> **Disclaimer:** This project is for research and educational purposes only.
> It is not financial advice. Historical data and notebook experiments do not
> guarantee future results.

## Repository contents

```text
.
├── notebooks/
│   └── Stocks Data processing.ipynb
├── rl_stocks/
│   ├── agents/
│   │   └── __init__.py
│   ├── environments/
│   │   └── __init__.py
│   └── __init__.py
├── environment.yml
├── pyproject.toml
└── setup.py
```

The `rl_stocks.agents` and `rl_stocks.environments` packages are currently
placeholders; no agent or environment classes are implemented yet. There is
also no test suite at this stage.

## Data-processing notebook

[`notebooks/Stocks Data processing.ipynb`](notebooks/Stocks%20Data%20processing.ipynb)
is the main runnable project component. It:

1. Downloads approximately three years of historical data from Yahoo Finance
   for EUR/USD, gold futures, crude oil futures, and the VIX.
2. Combines the downloaded series.
3. Calculates technical indicators for EUR/USD, including MACD, RSI, CCI, ADX,
   stochastic %K, Williams %R, Bollinger Bands, EMA, ATR, SAR, lagged closes,
   moving averages, and rolling standard deviations.
4. Applies log and Min-Max transformations to market price/volume columns,
   removes highly correlated indicator columns, standardizes the remaining
   indicators, and plots the EUR/USD closing price.

The download step requires an internet connection and availability of the
Yahoo Finance data source. Run the notebook with its working directory set to
`notebooks/`.

## Setup

The repository includes a Conda environment specification named `stocks`:

```bash
conda env create --file environment.yml
conda activate stocks
```

Install the local Python package in editable mode:

```bash
python -m pip install -e .
```

The package configuration currently declares no runtime dependencies. The
notebook imports pandas, NumPy, yfinance, pandas-datareader, TA-Lib,
scikit-learn, Matplotlib, and Plotly; install any missing notebook packages
into the active environment before running it. JupyterLab is also needed to
launch the notebook.

To open the notebook from the project root:

```bash
cd notebooks
jupyter lab "Stocks Data processing.ipynb"
```

## Current status

Data collection and indicator preparation are represented in the notebook.
Trading simulation, agent implementations, model training, backtesting, and
automated tests remain future work.
