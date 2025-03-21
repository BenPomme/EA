# LightningScalp EA

## Overview
LightningScalp is a high-frequency scalping Expert Advisor designed specifically for the EURUSD pair. It features advanced latency compensation mechanisms optimized for trading environments with 50ms ping. The EA implements a multi-signal approach that combines EMA crossovers, RSI, and market noise filtration to identify optimal trade entries and exits.

## Key Features

### Advanced Latency Compensation
- Built-in price projection algorithms that anticipate price movements during order transmission
- Adaptable to variable network conditions with customizable latency buffer
- Designed to perform optimally in 50ms ping environments

### Multi-Signal Trading System
- Fast/Slow EMA crossover strategy for trend identification
- RSI indicator for momentum and reversal signals
- Market noise filter to avoid choppy market conditions
- Trend strength analyzer to focus on high-probability trades

### Risk Management
- Per-trade risk percentage with dynamic position sizing
- Automatic leverage adjustment based on market volatility
- Maximum drawdown and daily loss protection
- Consecutive loss handling with progressive risk reduction
- End-of-day position closure

### Trade Management
- Dynamic trailing stops that activate after reaching profit targets
- Order rate limiting to prevent over-trading
- Session-based trading to focus on high-liquidity hours
- Real-time performance tracking and statistical dashboard

## Requirements
- MetaTrader 5 platform
- Recommended for EURUSD pair (can be adapted for other major pairs)
- Initial capital of $10,000 recommended
- Broker with tight spreads (3 points or less recommended)

## Installation
1. Copy the LightningScalp.mq5 file to your MetaTrader 5 Experts folder
2. Restart MetaTrader 5 or refresh the Navigator window
3. Drag and drop the LightningScalp EA onto your EURUSD M1 chart
4. Configure the input parameters as needed

## Parameters

### Trading Timeframe and Order Management
- **Trading_Timeframe**: Trading timeframe (default: M1)
- **OrdersPerMinute**: Maximum orders per minute (default: 3)
- **MaxSimultaneousOrders**: Maximum simultaneous orders (default: 5)
- **MaxSpreadPoints**: Maximum allowed spread in points (default: 3.0)
- **LatencyMs**: Your average network latency in milliseconds (default: 50)
- **LatencyBuffer**: Additional buffer for latency variation (default: 30)

### Strategy Parameters
- **Fast_EMA**: Fast EMA period (default: 12)
- **Slow_EMA**: Slow EMA period (default: 26)
- **RSI_Period**: RSI Period (default: 7)
- **RSI_OB**: RSI Overbought level (default: 70)
- **RSI_OS**: RSI Oversold level (default: 30)
- **UseMarketNoise**: Use Market Noise filter (default: true)
- **NoiseThreshold**: Market noise threshold (default: 0.4)
- **UseTrendStrength**: Use Trend Strength filter (default: true)
- **TrendThreshold**: Trend strength threshold (default: 0.6)

### Money Management
- **RiskPercent**: Risk percent per trade (default: 0.5%)
- **TP_Pips**: Take Profit in pips (default: 5.0)
- **SL_Pips**: Stop Loss in pips (default: 7.0)
- **TrailStart_Pips**: Start trailing after this many pips in profit (default: 3.0)
- **TrailStep_Pips**: Trailing step in pips (default: 0.5)
- **MaxLeverage**: Maximum leverage to use (default: 30.0)
- **DynamicLeverageAdjustment**: Adjust leverage based on volatility (default: true)

### Session & Risk Management
- **ManageTradingHours**: Only trade during specific hours (default: true)
- **ActiveHourStart**: Start hour of active trading in server time (default: 8)
- **ActiveHourEnd**: End hour of active trading in server time (default: 20)
- **CloseAllEOD**: Close all positions at end of day (default: true)
- **EOD_Hour**: Hour to close all positions (default: 23)
- **EOD_Minute**: Minute to close positions (default: 45)
- **MaxDailyLoss**: Maximum daily loss in % of account (default: 3.0%)
- **MaxDrawdown**: Maximum drawdown before stopping in % (default: 10.0%)

## Performance Expectation
LightningScalp is designed to make many small trades with tight profit targets. When properly configured, it can generate hundreds of trades per month with the following target performance metrics:

- Expected trades per day: 10-30 (market conditions dependent)
- Target win rate: 60-70%
- Target profit per month: 5-12% (account balance dependent)
- Maximum expected drawdown: 10-15%

## Optimization Tips
- For best results, optimize the EA parameters using MetaTrader's Strategy Tester with real tick data
- Focus optimization on these key parameters:
  - Fast/Slow EMA periods
  - RSI periods and levels
  - TP/SL pip settings
  - Noise and Trend thresholds
- Always use latency settings that match your actual trading environment

## Disclaimer
Trading foreign exchange carries a high level of risk and may not be suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade you should carefully consider your investment objectives, level of experience, and risk appetite. The possibility exists that you could sustain a loss of some or all of your initial investment and therefore you should not invest money that you cannot afford to lose.

## Version History
- 1.00 (2023-09-15): Initial release with latency compensation

## Contact and Support
- Creator: LightningScalp System
- Website: https://www.lightscalp.com
- Support Email: support@lightscalp.com 