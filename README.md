# Reinforcement Learning for Stocks

This repository explores historical stock-market data and reinforcement
learning. Its main Python component is an **offline data pipeline for DQN and
Double DQN experiments**: it downloads price data, computes features and
CVaR-based labels, and builds PyTorch transition datasets.

The package does not currently provide a trading simulator, implemented agent
classes, or a complete training/evaluation workflow. The notebooks contain
experiments and prototypes and may require additional code or cleanup to run
from start to finish.

> **Disclaimer:** This project is for research and educational purposes only.
> It is not financial advice. Historical data and experimental results do not
> guarantee future performance.

## Repository contents

```text
.
├── data/
│   ├── data_forex.csv
│   └── indicators_eurusd.csv
├── notebooks/
│   └── Stocks Data processing.ipynb
├── Environment_modeling.ipynb
├── rl_stocks/
│   ├── agents/                 # Placeholder
│   ├── environments/           # Placeholder
│   └── utils/
│       ├── data.py             # Data loading, reshaping, merging, scaling
│       ├── indicators.py       # Technical indicators, CVaR and action labels
│       ├── pipeline.py         # End-to-end data preparation
│       ├── transitions.py      # CVaR rewards and PyTorch Dataset
│       └── validation.py       # Data integrity checks
└── tests/
    └── test_pipeline.py
```

## Data pipeline

The pipeline in `rl_stocks.utils.pipeline` performs these steps:

1. Downloads OHLCV prices for the requested tickers and market indices from
   Yahoo Finance.
2. Computes daily returns and technical indicators with TA-Lib.
3. Calculates rolling CVaR and derives Hold, Buy, and Sell labels by comparing
   short- and long-horizon CVaR.
4. Aligns features, labels, and CVaR by date and ticker, checks the data, and
   scales feature columns.
5. Builds offline `(state, action, reward, next_state, done)` samples in a
   PyTorch `Dataset`.

This is an offline dataset builder, not a simulator: the actions are generated
from the CVaR comparison rather than chosen by an agent interacting with an
environment. The transition and reward assumptions are documented in
`rl_stocks.utils.transitions`.

### Example

Install the package and its runtime dependencies as described below, then:

```python
from datetime import date, timedelta
from torch.utils.data import DataLoader

from rl_stocks.utils.pipeline import prepare_training_data

tickers = ["AAPL", "MSFT", "NVDA"]
end = date.today()
start = end - timedelta(days=365 * 5)

prepared = prepare_training_data(
    tickers,
    start,
    end,
    sequence_length=10,
)

dataset = prepared["dataset"]
loader = DataLoader(dataset, batch_size=32, shuffle=True)
print("State shape:", dataset.state_shape)
print("Number of samples:", len(dataset))
```

`prepare_training_data` uses the S&P 500 (`^GSPC`) and VIX (`^VIX`) as market
tickers by default. It returns intermediate dataframes, the fitted scaler,
feature column names, action counts, and the final dataset.

For reliable out-of-sample evaluation, split data chronologically and fit
preprocessing only on training data. The current end-to-end helper scales all
rows it receives, so using it before a train/test split can leak information
from the test period.

## Setup

The package requires Python 3.10 or later. From the repository root, create
and activate an environment, then install the dependencies and this package:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn torch yfinance TA-Lib
python -m pip install -e .
```

The package metadata does not yet declare its runtime dependencies, so they
must be installed separately. TA-Lib may require platform-specific system
libraries. JupyterLab is needed to run the notebooks:

```bash
python -m pip install jupyterlab
jupyter lab
```

The repository also includes an `environment.yml` Conda specification; review
its platform-specific pins before using it on another machine.

## Notebooks and tests

- `Environment_modeling.ipynb` and
  `notebooks/Stocks Data processing.ipynb` contain data-processing and
  environment-related experiments.
- Some notebook cells refer to agent or environment code that is not currently
  implemented in the package.

Run the automated pipeline tests from the repository root with:

```bash
python -m pytest
```
