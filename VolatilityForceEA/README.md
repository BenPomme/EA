# VolatilityForce v0.2 - High-Frequency Scalping EA

A high-frequency MQL5 Expert Advisor specialized for the EURUSD pair, designed to generate hundreds of trades per year using multiple scalping strategies.

## Overview

VolatilityForce v0.2 is a comprehensive scalping system that combines multiple trading strategies to identify frequent trading opportunities. The system uses volatility measurements, momentum indicators, breakout patterns, and range-bound conditions to enter trades with tight risk parameters.

Built for a $5000 initial balance, this EA is optimized to generate hundreds of trades per year while maintaining risk control through multiple protective mechanisms.

## Trading Strategies

The EA implements three complementary trading strategies that work together:

1. **Breakout Strategy**: Identifies price breakouts during periods of increasing volatility
   - Monitors volatility changes through ATR
   - Detects breaks above recent highs/lows
   - Uses Bollinger Band breakouts as additional triggers
   - Confirms entries with RSI levels

2. **Momentum Strategy**: Capitalizes on accelerating price momentum
   - Tracks RSI momentum and direction changes
   - Looks for RSI divergence conditions
   - Confirms with price position relative to SMA
   - Implements momentum acceleration rules

3. **Range-Bound Strategy**: Exploits mean reversion in consolidating markets
   - Identifies Bollinger Band contractions
   - Looks for oversold/overbought conditions near band edges
   - Uses mean reversion principles for entries
   - Implements distance-from-mean measurements

## Key Features

- **M5 Timeframe**: Optimized for frequent trading opportunities
- **Multiple Strategies**: Three complementary trading approaches
- **Scalping Focus**: Small profits, high frequency
- **Session-Based Trading**: Trade only during selected market hours
- **Risk Management**: Multiple protective measures to preserve capital
- **Performance Tracking**: Built-in trade statistics and win rate calculation
- **End-of-Day Closure**: Avoids overnight exposure
- **High-Leverage Compatible**: Works with leverage up to 500:1
- **Multiple Position Management**: Up to 8 simultaneous positions
- **Trade Metrics Display**: Real-time performance stats on chart

## Installation

1. Copy the EA files to your MetaTrader's MQL5/Experts directory
2. Restart MetaTrader or refresh the Navigator panel
3. Drag and drop the EA onto a EURUSD chart (M5 recommended)
4. Configure the input parameters as needed
5. Start the EA

## Parameters

### Trading Timeframe
- **Trading_Timeframe**: Main trading timeframe (default: M5)

### Strategy Parameters
- **ATR_Period**: Period for the ATR indicator (default: 5)
- **ATR_Multiplier**: Multiplier for ATR to define entry thresholds (default: 0.5)
- **RSI_Period**: Period for the RSI indicator (default: 5)
- **SMA_Period**: Period for the Simple Moving Average (default: 10)
- **UseBreakoutStrategy**: Enable breakout-based entries (default: true)
- **UseMomentumStrategy**: Enable momentum-based entries (default: true)
- **UseRangeStrategy**: Enable range-bound trading entries (default: true)

### Money Management
- **Risk_Percent**: Risk percentage per trade (default: 0.5%)
- **TP_ATR_Multiplier**: Take Profit ATR multiplier (default: 0.8)
- **SL_ATR_Multiplier**: Stop Loss ATR multiplier (default: 0.6)
- **MaxDrawdownPercent**: Maximum allowed drawdown (default: 15.0%)
- **DailyLossLimitPercent**: Daily loss limit (default: 5.0%)

### Trade Management
- **Slippage**: Maximum allowed slippage in points (default: 10)
- **UseTrailingStop**: Enable/disable trailing stop feature (default: true)
- **TrailingATR**: Trailing stop ATR multiplier (default: 0.3)
- **MaxLeverage**: Maximum leverage to use (default: 500)
- **MaxActivePositions**: Maximum number of active positions at once (default: 8)
- **TradeDelay**: Delay between trades in seconds (default: 60 - 1 minute)

### Session Management
- **ClosePositionsEndOfDay**: Close all positions at end of day (default: true)
- **EndOfDayHour**: Hour to close positions, 0-23 (default: 23)
- **EndOfDayMinute**: Minute to close positions, 0-59 (default: 0)
- **NoTradingFriday**: Stop opening new positions on Friday afternoon (default: true)
- **SessionTrading**: Only trade during specific sessions (default: true)
- **SessionStartHour**: Session start hour, 0-23 (default: 8)
- **SessionEndHour**: Session end hour, 0-23 (default: 20)

## Risk Management

The EA incorporates multiple layers of risk management to protect your account:

1. **Reduced Position Sizing**: Only 0.5% risk per trade with further 60% reduction
2. **Multiple Small Positions**: Spreads risk across up to 8 smaller positions
3. **Tight Stop-Losses**: Uses ATR-based stops with a 0.6 multiplier
4. **Tighter Trailing Stops**: 0.3 ATR for quick profit protection
5. **Drawdown Protection**: Automatically closes all positions if drawdown exceeds 15%
6. **Daily Loss Limit**: Stops trading for the day if daily loss exceeds 5%
7. **End-of-Day Closure**: Closes all positions at end of day to avoid overnight exposure
8. **Session Control**: Only trades during specified market hours
9. **Weekend Protection**: Avoids new trades on Friday afternoons
10. **Trade Delay**: Enforces a 1-minute cooldown between trades

## Performance Expectations

With the high-frequency scalping approach, you can expect:
- 100-300+ trades per month depending on market conditions
- Smaller profit per trade but higher overall frequency
- More consistent equity curve through diversified strategies
- Better statistical significance with high trade count
- Win rate tracking and performance metrics
- Improved profit expectancy through multiple entry methods
- Protection from excessive drawdowns

## Backtesting Recommendations

When backtesting this EA, it's recommended to:

1. Use "Every tick" mode for accurate scalping backtests
2. Start with a $5000 initial balance
3. Test each strategy individually by disabling others first
4. Pay attention to commission costs due to high trade frequency
5. Use at least 1 year of high-quality tick data
6. Be aware of the impact of spread on scalping performance
7. Enable real market hours simulation

## Potential Adjustments

Based on backtesting results, you might want to adjust:
1. Trade frequency via TradeDelay parameter
2. Risk exposure by changing Risk_Percent and position limits
3. Strategy selection by enabling/disabling individual strategies
4. Trading hours via SessionStartHour and SessionEndHour
5. Take profit and stop loss levels via ATR multipliers

## License

Copyright © 2023 VolatilityForce System 