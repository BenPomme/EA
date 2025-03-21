# MetaTrader Backtesting Framework

A Python-based backtesting framework for MetaTrader Expert Advisors (EAs), supporting both backtest and forward test functionality.

## Setup

1. Make sure you have Python 3.8+ installed
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`
4. Install the required packages: `pip install pandas numpy matplotlib backtrader pytz`

## Data Preparation

To run backtests, you need historical data from MetaTrader in CSV format:

1. In MetaTrader, open the History Center (F2)
2. Select the symbol and timeframe you want to export
3. Right-click and select "Export"
4. Save the CSV file to the `data` directory in this project

### Data Format

MetaTrader 4 CSV format is expected to have the following columns:
- date (YYYY.MM.DD)
- time (HH:MM)
- open
- high
- low
- close
- volume

## Running Backtests

Use the `run_backtest.py` script to run backtests:

```bash
python run_backtest.py --data-file EURUSD_Daily.csv --strategy ma_crossover --start-date 2020-01-01 --end-date 2023-12-31
```

### Command-line Arguments

- `--data-file` (required): CSV file containing OHLCV data
- `--strategy` (default: `ma_crossover`): Trading strategy to backtest
  - Options: `ma_crossover`, `rsi_mean_reversion`, `bollinger_breakout`
- `--start-date` (optional): Start date for backtest (YYYY-MM-DD)
- `--end-date` (optional): End date for backtest (YYYY-MM-DD)
- `--timeframe` (default: `1D`): Timeframe for backtest (e.g., 1D, 4H, 1H)
- `--cash` (default: 10000.0): Initial cash amount
- `--commission` (default: 0.0001): Commission rate
- `--data-dir` (default: `data`): Directory containing data files
- `--output-dir` (default: `results`): Directory to save results
- `--no-plot`: Disable plotting
- `--forward-test`: Perform forward testing
- `--forward-ratio` (default: 0.25): Ratio of data to use for forward testing

## Forward Testing

Forward testing helps validate your strategy by training on one period and testing on unseen data:

```bash
python run_backtest.py --data-file EURUSD_Daily.csv --strategy rsi_mean_reversion --forward-test --forward-ratio 0.25
```

This will use 75% of the data for training and 25% for forward testing.

## Available Strategies

### 1. Moving Average Crossover (ma_crossover)

This strategy generates buy signals when the fast moving average crosses above the slow moving average, and sell signals when the fast moving average crosses below the slow moving average.

Parameters:
- `fast_period` (default: 10): Fast moving average period
- `slow_period` (default: 30): Slow moving average period
- `order_pct` (default: 0.95): Order percentage of portfolio
- `stop_loss_pct` (default: 0.02): Stop loss percentage
- `take_profit_pct` (default: 0.03): Take profit percentage

### 2. RSI Mean Reversion (rsi_mean_reversion)

This strategy buys when the RSI is oversold and sells when the RSI is overbought. It also uses ATR for position sizing.

Parameters:
- `rsi_period` (default: 14): RSI period
- `rsi_overbought` (default: 70): RSI overbought level
- `rsi_oversold` (default: 30): RSI oversold level
- `atr_period` (default: 14): ATR period
- `risk_pct` (default: 0.02): Risk percentage per trade
- `stop_loss_atr` (default: 2.0): Stop loss in ATR units
- `take_profit_atr` (default: 3.0): Take profit in ATR units

### 3. Bollinger Breakout (bollinger_breakout)

This strategy buys when price breaks above the upper Bollinger Band and sells when price breaks below the lower Bollinger Band. It uses a trailing stop for exit.

Parameters:
- `bb_period` (default: 20): Bollinger Bands period
- `bb_dev` (default: 2.0): Bollinger Bands standard deviation
- `order_pct` (default: 0.95): Order percentage of portfolio
- `trail_percent` (default: 0.05): Trailing stop percentage

## Creating Custom Strategies

To create a custom strategy:

1. Create a new Python file in the project directory
2. Implement your strategy by extending the `bt.Strategy` class
3. Register your strategy in the `strategy_map` in `run_backtest.py`

## Results

Backtest results are saved to the `results` directory and include:
- Portfolio value and profit/loss
- Sharpe ratio
- Maximum drawdown
- Number of trades and win rate
- System Quality Number (SQN)
- Annual return

A plot is also generated showing the equity curve and trades. 