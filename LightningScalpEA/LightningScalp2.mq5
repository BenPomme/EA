//+------------------------------------------------------------------+
//|                                                LightningScalp.mq5 |
//|                            Copyright 2023, LightningScalp System  |
//|                                       https://www.lightscalp.com  |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, LightningScalp System"
#property link      "https://www.lightscalp.com"
#property version   "1.03"
#property description "Ultra-Fast Scalping System for EURUSD with Latency Compensation"
#property description "Initial account: $10,000, variable leverage"
#property description "Optimized for 50ms ping environments"
#property description "FP Markets fee-optimized"

// Include necessary libraries
#include <Trade/Trade.mqh>
#include <Trade/SymbolInfo.mqh>
#include <Trade/PositionInfo.mqh>
#include <Trade/OrderInfo.mqh>

// Input parameters - Trading Timeframe and Order Management
input ENUM_TIMEFRAMES Trading_Timeframe = PERIOD_M1;  // Trading timeframe (M1 for high-frequency)
input int      OrdersPerMinute = 5;                  // Maximum orders per minute
input int      MaxSimultaneousOrders = 5;            // Maximum simultaneous orders
input double   MaxSpreadPoints = 7.0;                // Maximum spread in points to trade
input int      LatencyMs = 50;                       // Your average latency in milliseconds
input int      LatencyBuffer = 30;                   // Additional buffer for latency variation (ms)

// Input parameters - Strategy Parameters
input int      Fast_EMA = 5;                         // Fast EMA period (reduced for more signals)
input int      Slow_EMA = 15;                        // Slow EMA period (reduced for more signals)
input int      RSI_Period = 5;                       // RSI Period (reduced for faster response)
input int      RSI_OB = 70;                          // RSI Overbought level (adjusted)
input int      RSI_OS = 30;                          // RSI Oversold level (adjusted)
input bool     UseMarketNoise = false;               // Use Market Noise filter
input double   NoiseThreshold = 0.8;                 // Market noise threshold (increased)
input bool     UseTrendStrength = false;             // Use Trend Strength filter
input double   TrendThreshold = 0.3;                 // Trend strength threshold (reduced)
input bool     UseAlternativeSignals = true;         // Use alternative signals for more trades
input bool     UseAggressiveEntry = true;            // Use aggressive entry conditions

// Input parameters - Money Management
input double   RiskPercent = 0.5;                    // Risk percent per trade
input double   TP_Pips = 8.0;                        // Take Profit in pips
input double   SL_Pips = 10.0;                       // Stop Loss in pips
input double   TrailStart_Pips = 4.0;                // Start trailing after this many pips
input double   TrailStep_Pips = 0.5;                 // Trailing step in pips
input double   MaxLeverage = 30.0;                   // Maximum leverage to use
input double   DynamicLeverageAdjustment = true;     // Adjust leverage based on volatility

// Input parameters - Session & Risk Management
input bool     ManageTradingHours = false;           // Only trade during specific hours
input int      ActiveHourStart = 0;                  // Start hour of active trading (server time)
input int      ActiveHourEnd = 23;                   // End hour of active trading (server time)
input bool     CloseAllEOD = true;                   // Close all positions at end of day
input int      EOD_Hour = 23;                        // Hour to close all positions (0-23)
input int      EOD_Minute = 45;                      // Minute to close positions (0-59)
input double   MaxDailyLoss = 5.0;                   // Maximum daily loss (% of account)
input double   MaxDrawdown = 15.0;                   // Maximum drawdown before stopping (%)
input bool     EnableDebugMode = true;               // Enable detailed debug logging

// Input parameters - Broker Commission Handling
input double   CommissionPerLot = 5.5;               // Commission per lot (round trip) in account currency
input bool     AccountForCommission = true;          // Consider commission in profit calculations
input double   MinProfitTargetMultiplier = 1.5;      // Minimum profit must be this times the commission

// Global variables - Trading
CTrade         trade;                                // Trade object
CSymbolInfo    symbolInfo;                           // Symbol information
CPositionInfo  positionInfo;                         // Position information
COrderInfo     orderInfo;                            // Order information
int            fast_ema_handle;                      // Fast EMA indicator handle
int            slow_ema_handle;                      // Slow EMA indicator handle
int            rsi_handle;                           // RSI indicator handle
int            atr_handle;                           // ATR indicator handle
double         fast_ema_buffer[];                    // Fast EMA buffer
double         slow_ema_buffer[];                    // Slow EMA buffer
double         rsi_buffer[];                         // RSI buffer
double         atr_buffer[];                         // ATR buffer
double         price_buffer[];                       // Price buffer for calculations
double         high_buffer[];                        // High price buffer
double         low_buffer[];                         // Low price buffer
double         point;                                // Point value
double         pip_value;                            // Pip value
int            pip_digits;                           // Digits in a pip
datetime       last_order_time = 0;                  // Time of last order
int            orders_this_minute = 0;               // Orders placed in the current minute
datetime       current_minute = 0;                   // Current minute timestamp

// Global variables - Risk Management
double         starting_balance;                     // Account balance at EA start
double         daily_profit_loss = 0;                // Daily profit/loss
double         max_equity = 0;                       // Maximum equity reached
double         current_drawdown = 0;                 // Current drawdown
bool           daily_loss_reached = false;           // Flag for daily loss limit
bool           max_drawdown_reached = false;         // Flag for max drawdown
datetime       last_day = 0;                         // Last trading day
bool           eod_closure_done = false;             // Flag for end of day closure
int            total_trades = 0;                     // Total trades taken
int            winning_trades = 0;                   // Number of winning trades
int            losing_trades = 0;                    // Number of losing trades
int            consecutive_losses = 0;               // Consecutive losing trades

// Global variables - Debug
int            signal_check_count = 0;              // Counter for signal checks
int            conditions_not_met_count = 0;        // Counter for conditions not met
bool           logged_this_tick = false;            // Flag to prevent excessive logging

// Global variables - Commission Tracking
double         commission_per_pip = 0;              // Commission cost in terms of pips
double         min_profit_pips = 0;                 // Minimum profit required in pips to overcome commission

// Additional Global Variables
double         last_signal_price = 0;                // Last price where signal was generated
int            signal_cooldown_ticks = 0;            // Cooldown period between signals
const int      SIGNAL_COOLDOWN = 2;                  // Reduced from 5 to 2 ticks between signals

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize the symbol info object
   if(!symbolInfo.Name(_Symbol))
   {
      Print("Failed to initialize symbol info");
      return INIT_FAILED;
   }
   
   // Set trading parameters
   trade.SetExpertMagicNumber(987654);
   trade.SetMarginMode();
   trade.SetTypeFillingBySymbol(symbolInfo.Name());
   trade.SetDeviationInPoints(5);  // Increased slippage control
   
   // Calculate point, pip values and pip digits
   point = symbolInfo.Point();
   
   // Adjust for 5-digit brokers (standard is 4 digits for forex)
   if(symbolInfo.Digits() == 5 || symbolInfo.Digits() == 3)
   {
      pip_value = point * 10;
      pip_digits = 1;
   }
   else
   {
      pip_value = point;
      pip_digits = 0;
   }
   
   // Calculate commission in terms of pips
   double tick_value = symbolInfo.TickValue();
   if(symbolInfo.Digits() == 3 || symbolInfo.Digits() == 5)
      tick_value = tick_value * 10;
   
   double commission_per_lot_one_way = CommissionPerLot / 2.0;  // Split round trip commission
   double lot_size = 1.0;  // Standard lot
   commission_per_pip = commission_per_lot_one_way / (TP_Pips * tick_value / point);
   
   // Calculate minimum profit needed to overcome commission
   min_profit_pips = CommissionPerLot / (tick_value / point) * MinProfitTargetMultiplier;
   
   // Adjust take profit if necessary to account for commission
   if(AccountForCommission && TP_Pips < min_profit_pips)
   {
      Print("Warning: TP_Pips (", TP_Pips, ") is less than minimum required to overcome commission (", 
            DoubleToString(min_profit_pips, 1), "). Adjusted to ", DoubleToString(min_profit_pips, 1));
      // Note: We won't force this change, just log it
   }
   
   // Create indicator handles
   fast_ema_handle = iMA(_Symbol, Trading_Timeframe, Fast_EMA, 0, MODE_EMA, PRICE_CLOSE);
   slow_ema_handle = iMA(_Symbol, Trading_Timeframe, Slow_EMA, 0, MODE_EMA, PRICE_CLOSE);
   rsi_handle = iRSI(_Symbol, Trading_Timeframe, RSI_Period, PRICE_CLOSE);
   atr_handle = iATR(_Symbol, Trading_Timeframe, 14);
   
   if(fast_ema_handle == INVALID_HANDLE || slow_ema_handle == INVALID_HANDLE || 
      rsi_handle == INVALID_HANDLE || atr_handle == INVALID_HANDLE)
   {
      Print("Error creating indicator handles");
      return INIT_FAILED;
   }
   
   // Set arrays as series
   ArraySetAsSeries(fast_ema_buffer, true);
   ArraySetAsSeries(slow_ema_buffer, true);
   ArraySetAsSeries(rsi_buffer, true);
   ArraySetAsSeries(atr_buffer, true);
   ArraySetAsSeries(price_buffer, true);
   ArraySetAsSeries(high_buffer, true);
   ArraySetAsSeries(low_buffer, true);
   
   // Initialize risk management variables
   starting_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   max_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   last_day = 0;  // Will be set on first tick
   daily_profit_loss = 0;
   daily_loss_reached = false;
   max_drawdown_reached = false;
   eod_closure_done = false;
   total_trades = 0;
   winning_trades = 0;
   losing_trades = 0;
   consecutive_losses = 0;
   
   // Reset debug counters
   signal_check_count = 0;
   conditions_not_met_count = 0;
   
   // Set chart comments
   ChartSetString(0, CHART_COMMENT, "LightningScalp EA Running v1.03 - FP Markets Fee-Optimized");
   
   Print("LightningScalp EA v1.03 initialized. FP Markets fee structure integrated.");
   Print("Commission per standard lot (round trip): €", DoubleToString(CommissionPerLot, 2));
   Print("Min profit needed to overcome commission: ", DoubleToString(min_profit_pips, 1), " pips");
   Print("Current spread: ", DoubleToString(symbolInfo.Spread(), 1), " points");
   
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(fast_ema_handle != INVALID_HANDLE) IndicatorRelease(fast_ema_handle);
   if(slow_ema_handle != INVALID_HANDLE) IndicatorRelease(slow_ema_handle);
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
   
   // Log performance metrics
   if(total_trades > 0)
   {
      double win_rate = (double)winning_trades / (double)total_trades * 100.0;
      double avg_win_per_trade = total_trades > 0 ? (AccountInfoDouble(ACCOUNT_BALANCE) - starting_balance) / total_trades : 0;
      
      Print("LightningScalp Performance Summary:");
      Print("Total Trades: ", total_trades);
      Print("Winning Trades: ", winning_trades, " (", DoubleToString(win_rate, 2), "%)");
      Print("Losing Trades: ", losing_trades);
      Print("Average Profit Per Trade: $", DoubleToString(avg_win_per_trade, 2));
   }
   
   Print("LightningScalp EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Reset logging flag for this tick
   logged_this_tick = false;
   
   // Update symbol info
   if(!symbolInfo.RefreshRates())
   {
      if(EnableDebugMode) Print("Failed to refresh rates");
      return;
   }
   
   // Log current market state every 100 ticks
   if(EnableDebugMode && signal_check_count % 100 == 0)
   {
      Print("Current Spread: ", DoubleToString(symbolInfo.Spread(), 1), 
            " points (Max allowed: ", DoubleToString(MaxSpreadPoints, 1), ")");
      Print("Current time: ", TimeToString(TimeCurrent()), 
            ", Trading hours: ", (IsInTradingHours() ? "YES" : "NO"),
            ", EOD closure done: ", (eod_closure_done ? "YES" : "NO"));
      Print("Signal checks: ", signal_check_count, 
            ", Conditions not met: ", conditions_not_met_count,
            ", Signal/check ratio: ", 
            DoubleToString(signal_check_count > 0 ? (double)(signal_check_count - conditions_not_met_count) / signal_check_count * 100.0 : 0, 1), "%");
   }
   
   // Increment signal check counter
   signal_check_count++;
   
   // Check for new day - reset daily P/L tracking
   CheckNewDay();
   
   // Check if we need to close positions at end of day
   if(CloseAllEOD)
      CheckEndOfDay();
   
   // Update drawdown monitoring
   UpdateRiskMetrics();
   
   // Check if we can trade based on risk management
   if(daily_loss_reached || max_drawdown_reached)
   {
      if(EnableDebugMode && !logged_this_tick) 
      {
         Print("Trading blocked: ", (daily_loss_reached ? "Daily loss reached" : "Max drawdown reached"));
         logged_this_tick = true;
      }
      return;
   }
   
   // Check if we're in trading hours
   if(ManageTradingHours && !IsInTradingHours())
   {
      if(EnableDebugMode && !logged_this_tick) 
      {
         Print("Outside trading hours: ", TimeToString(TimeCurrent()));
         logged_this_tick = true;
      }
      return;
   }
   
   // Check if spread is too wide
   if(symbolInfo.Spread() > MaxSpreadPoints)
   {
      if(EnableDebugMode && !logged_this_tick) 
      {
         Print("Spread too wide: ", DoubleToString(symbolInfo.Spread(), 1), 
               " points (Max: ", DoubleToString(MaxSpreadPoints, 1), ")");
         logged_this_tick = true;
      }
      return;
   }
   
   // Check rate limiting
   if(!CheckOrderRateLimit())
   {
      if(EnableDebugMode && !logged_this_tick) 
      {
         Print("Rate limit reached: ", orders_this_minute, " orders this minute (Max: ", OrdersPerMinute, ")");
         logged_this_tick = true;
      }
      return;
   }
   
   // Copy indicator values
   if(CopyBuffer(fast_ema_handle, 0, 0, 3, fast_ema_buffer) <= 0 ||
      CopyBuffer(slow_ema_handle, 0, 0, 3, slow_ema_buffer) <= 0 ||
      CopyBuffer(rsi_handle, 0, 0, 3, rsi_buffer) <= 0 ||
      CopyBuffer(atr_handle, 0, 0, 3, atr_buffer) <= 0 ||
      CopyClose(_Symbol, Trading_Timeframe, 0, 10, price_buffer) <= 0 ||
      CopyHigh(_Symbol, Trading_Timeframe, 0, 10, high_buffer) <= 0 ||
      CopyLow(_Symbol, Trading_Timeframe, 0, 10, low_buffer) <= 0)
   {
      if(EnableDebugMode) Print("Error copying indicator values");
      return;
   }
   
   // Manage existing positions
   ManageOpenPositions();

   // Update current position count after management
   int position_count = CountOpenPositions();

   // Check if we can open new positions
   if(position_count >= MaxSimultaneousOrders)
   {
      if(EnableDebugMode && !logged_this_tick)
      {
         Print("Max positions reached: ", position_count, "/", MaxSimultaneousOrders);
         logged_this_tick = true;
      }
      return;
   }
   
   // Log indicator values if in debug mode
   if(EnableDebugMode && signal_check_count % 100 == 0)
   {
      Print("Fast EMA: ", DoubleToString(fast_ema_buffer[0], 5), 
            ", Slow EMA: ", DoubleToString(slow_ema_buffer[0], 5),
            ", RSI: ", DoubleToString(rsi_buffer[0], 2),
            ", ATR: ", DoubleToString(atr_buffer[0], 5));
   }
   
   // Analyze market for trade signals
   AnalyzeMarket();
   
   // Update display information
   UpdateDashboard();
}

//+------------------------------------------------------------------+
//| Analyze market for trade signals                                 |
//+------------------------------------------------------------------+
void AnalyzeMarket()
{
   // Get current price data
   double ask = symbolInfo.Ask();
   double bid = symbolInfo.Bid();
   double spread = symbolInfo.Spread() * point;
   
   // Decrement signal cooldown
   if(signal_cooldown_ticks > 0)
      signal_cooldown_ticks--;
   
   // Get indicator values
   double fast_ema = fast_ema_buffer[0];
   double slow_ema = slow_ema_buffer[0];
   double fast_ema_prev = fast_ema_buffer[1];
   double slow_ema_prev = slow_ema_buffer[1];
   double fast_ema_prev2 = fast_ema_buffer[2];
   double slow_ema_prev2 = slow_ema_buffer[2];
   double rsi = rsi_buffer[0];
   double rsi_prev = rsi_buffer[1];
   double rsi_prev2 = rsi_buffer[2];
   double atr = atr_buffer[0];
   
   // Calculate trend strength (EMA alignment and distance)
   double trend_strength = TrendStrengthCalculator(fast_ema, slow_ema, atr);
   
   // Calculate market noise level
   double noise_level = MarketNoiseCalculator();
   
   // EMA Cross Signal - Made more sensitive
   bool ema_cross_up = fast_ema > slow_ema && (fast_ema_prev <= slow_ema_prev || fast_ema_prev2 <= slow_ema_prev2);
   bool ema_cross_down = fast_ema < slow_ema && (fast_ema_prev >= slow_ema_prev || fast_ema_prev2 >= slow_ema_prev2);
   
   // RSI Signal - Made more sensitive
   bool rsi_signal_buy = rsi < RSI_OS || (rsi < 45 && rsi_prev < rsi && rsi < fast_ema);
   bool rsi_signal_sell = rsi > RSI_OB || (rsi > 55 && rsi_prev > rsi && rsi > fast_ema);
   
   // Alternative signals (more relaxed conditions)
   bool alt_ema_signal_buy = fast_ema > fast_ema_prev;
   bool alt_ema_signal_sell = fast_ema < fast_ema_prev;
   bool alt_rsi_signal_buy = rsi < 45 && rsi_prev < rsi;
   bool alt_rsi_signal_sell = rsi > 55 && rsi_prev > rsi;
   
   // Aggressive entry signals
   bool agg_signal_buy = false;
   bool agg_signal_sell = false;
   
   if(UseAggressiveEntry)
   {
      // Momentum-based entry - Made more sensitive
      double price_momentum = (price_buffer[0] - price_buffer[2]) / (2 * atr);
      agg_signal_buy = price_momentum > 0.1 && rsi < 65;  // Relaxed conditions
      agg_signal_sell = price_momentum < -0.1 && rsi > 35; // Relaxed conditions
      
      // Volatility breakout entry
      double recent_high = high_buffer[ArrayMaximum(high_buffer, 0, 3)];
      double recent_low = low_buffer[ArrayMinimum(low_buffer, 0, 3)];
      double volatility_threshold = atr * 0.3; // Reduced threshold
      
      agg_signal_buy = agg_signal_buy || (ask > recent_high - volatility_threshold);
      agg_signal_sell = agg_signal_sell || (bid < recent_low + volatility_threshold);
   }
   
   // Market Condition Filters - Made less restrictive
   bool noise_filter_pass = !UseMarketNoise || noise_level < NoiseThreshold;
   bool trend_filter_pass = !UseTrendStrength || trend_strength > TrendThreshold;
   
   // Price movement check - Made less restrictive
   bool price_moved = MathAbs(ask - last_signal_price) > (point * 1); // Reduced from 2 to 1 point
   
   // Signal combination logic
   bool buy_signal = ((ema_cross_up || rsi_signal_buy) || 
                     (UseAlternativeSignals && (alt_ema_signal_buy || alt_rsi_signal_buy)) ||
                     (UseAggressiveEntry && agg_signal_buy)) &&
                     noise_filter_pass && trend_filter_pass && 
                     (signal_cooldown_ticks == 0 || price_moved);
   
   bool sell_signal = ((ema_cross_down || rsi_signal_sell) || 
                      (UseAlternativeSignals && (alt_ema_signal_sell || alt_rsi_signal_sell)) ||
                      (UseAggressiveEntry && agg_signal_sell)) &&
                      noise_filter_pass && trend_filter_pass && 
                      (signal_cooldown_ticks == 0 || price_moved);
   
   // Debug logs for conditions
   if(EnableDebugMode && signal_check_count % 20 == 0)
   {
      Print("Signal conditions - Buy: EMA cross=", ema_cross_up, ", RSI=", rsi_signal_buy,
            ", Alt EMA=", alt_ema_signal_buy, ", Alt RSI=", alt_rsi_signal_buy,
            ", Agg=", (UseAggressiveEntry && agg_signal_buy));
      Print("Signal conditions - Sell: EMA cross=", ema_cross_down, ", RSI=", rsi_signal_sell,
            ", Alt EMA=", alt_ema_signal_sell, ", Alt RSI=", alt_rsi_signal_sell,
            ", Agg=", (UseAggressiveEntry && agg_signal_sell));
      Print("Filters: Noise=", noise_filter_pass, ", Trend=", trend_filter_pass,
            ", Cooldown=", (signal_cooldown_ticks == 0), ", Price moved=", price_moved);
   }
   
   // Execute trades based on signals
   if(buy_signal)
   {
      if(EnableDebugMode) Print("BUY SIGNAL TRIGGERED");
      last_signal_price = ask;
      signal_cooldown_ticks = SIGNAL_COOLDOWN;
      OpenPosition(ORDER_TYPE_BUY, ask);
   }
   else if(sell_signal)
   {
      if(EnableDebugMode) Print("SELL SIGNAL TRIGGERED");
      last_signal_price = bid;
      signal_cooldown_ticks = SIGNAL_COOLDOWN;
      OpenPosition(ORDER_TYPE_SELL, bid);
   }
   else
   {
      conditions_not_met_count++;
      
      if(EnableDebugMode && signal_check_count % 100 == 0)
      {
         Print("No signal triggered. Condition checks: ", signal_check_count, 
               ", Not met: ", conditions_not_met_count);
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate trend strength based on EMAs                           |
//+------------------------------------------------------------------+
double TrendStrengthCalculator(double fast_ema, double slow_ema, double atr)
{
   // Calculate distance between fast and slow EMAs
   double ema_distance = MathAbs(fast_ema - slow_ema);
   
   // Normalize by ATR to get relative strength
   double rel_strength = ema_distance / (atr * 2);
   
   // Apply sigmoid function to map to 0-1 range
   double trend_strength = 1.0 / (1.0 + MathExp(-5 * (rel_strength - 0.5)));
   
   return trend_strength;
}

//+------------------------------------------------------------------+
//| Calculate market noise level (0-1 range)                         |
//+------------------------------------------------------------------+
double MarketNoiseCalculator()
{
   // Use price buffer to calculate noise
   double price_sum = 0;
   double dir_movement = MathAbs(price_buffer[0] - price_buffer[9]);
   double total_movement = 0;
   
   // Calculate total price movement
   for(int i = 0; i < 9; i++)
   {
      total_movement += MathAbs(price_buffer[i] - price_buffer[i+1]);
   }
   
   // Noise ratio: 0 = perfect trend, 1 = maximum noise
   double noise_ratio = 0;
   if(total_movement > 0)
      noise_ratio = 1.0 - (dir_movement / total_movement);
   
   return noise_ratio;
}

//+------------------------------------------------------------------+
//| Open a new position with risk management                         |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE order_type, double entry_price)
{
   // Calculate TP and SL prices
   double tp_price = 0, sl_price = 0;
   
   if(order_type == ORDER_TYPE_BUY)
   {
      tp_price = entry_price + (TP_Pips * pip_value);
      sl_price = entry_price - (SL_Pips * pip_value);
   }
   else if(order_type == ORDER_TYPE_SELL)
   {
      tp_price = entry_price - (TP_Pips * pip_value);
      sl_price = entry_price + (SL_Pips * pip_value);
   }
   else
   {
      return; // Invalid order type
   }
   
   // Check if potential profit overcomes commission costs
   double risk_in_pips = MathAbs(entry_price - sl_price) / pip_value;
   double reward_in_pips = MathAbs(entry_price - tp_price) / pip_value;
   
   if(AccountForCommission && reward_in_pips < min_profit_pips)
   {
      if(EnableDebugMode) 
      {
         Print("Potential reward (", DoubleToString(reward_in_pips, 1), 
               " pips) is less than minimum needed (", DoubleToString(min_profit_pips, 1), 
               " pips) to overcome commission. Adjusting target.");
      }
      
      // Adjust TP to ensure minimum profit
      if(order_type == ORDER_TYPE_BUY)
      {
         tp_price = entry_price + (min_profit_pips * pip_value);
      }
      else // SELL
      {
         tp_price = entry_price - (min_profit_pips * pip_value);
      }
   }
   
   // Calculate position size based on risk management
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (RiskPercent / 100.0);
   
   // Risk adjustment based on consecutive losses
   if(consecutive_losses > 1)
   {
      risk_amount = risk_amount * (1.0 - (0.1 * consecutive_losses)); // Reduce risk by 10% per consecutive loss
      risk_amount = MathMax(risk_amount, account_balance * 0.0005); // Minimum risk 0.05%
   }
   
   // Calculate risk in price terms
   double risk_price_distance = MathAbs(entry_price - sl_price);
   
   // Calculate pip value in account currency
   double tick_value = symbolInfo.TickValue();
   if(symbolInfo.Digits() == 3 || symbolInfo.Digits() == 5)
      tick_value = tick_value * 10;
   
   // Calculate lot size based on risk
   double risk_lots = risk_amount / (risk_price_distance * tick_value / point);
   
   // Adjust for volatility if enabled
   if(DynamicLeverageAdjustment)
   {
      double atr_volatility = atr_buffer[0];
      double avg_atr = (atr_buffer[0] + atr_buffer[1] + atr_buffer[2]) / 3;
      double volatility_ratio = atr_volatility / avg_atr;
      
      // Reduce position size in higher volatility
      if(volatility_ratio > 1.5)
         risk_lots = risk_lots * 0.7;
      else if(volatility_ratio > 1.2)
         risk_lots = risk_lots * 0.85;
      // Increase position size in lower volatility
      else if(volatility_ratio < 0.7)
         risk_lots = risk_lots * 1.2;
      else if(volatility_ratio < 0.5)
         risk_lots = risk_lots * 1.3;
   }
   
   // Account for commission costs when opening multiple small positions
   // Ensure lot size is large enough for commission to make sense
   double commission_cost = CommissionPerLot * risk_lots;
   double expected_pip_value = risk_lots * (tick_value / point) * reward_in_pips;
   
   // If commission is too large compared to potential gain, increase lot size or skip trade
   if(AccountForCommission && commission_cost > expected_pip_value * 0.4) // If commission is more than 40% of potential profit
   {
      if(risk_lots < 0.1) // If lot size is very small
      {
         if(EnableDebugMode) 
            Print("Skipping trade: Commission cost too high relative to potential profit at small lot size");
         return; // Skip this trade
      }
      else
      {
         // Ensure minimum lot size for commission to make sense
         double min_lot_for_commission = (CommissionPerLot * 0.5) / (reward_in_pips * (tick_value / point));
         if(risk_lots < min_lot_for_commission)
         {
            risk_lots = min_lot_for_commission;
            if(EnableDebugMode)
               Print("Adjusting lot size to minimum required for commission efficiency: ", DoubleToString(risk_lots, 2));
         }
      }
   }
   
   // Calculate expected commission and profit for logging
   double expected_commission = CommissionPerLot * risk_lots;
   double expected_profit = reward_in_pips * risk_lots * (tick_value / point);
   double profit_after_commission = expected_profit - expected_commission;
   
   if(EnableDebugMode)
   {
      Print("Trade Analysis - Lot Size: ", DoubleToString(risk_lots, 2),
            ", Expected Profit: ", DoubleToString(expected_profit, 2),
            ", Commission: ", DoubleToString(expected_commission, 2),
            ", Net Profit: ", DoubleToString(profit_after_commission, 2));
   }
   
   // Execute the trade with market order
   bool result = false;
   
   if(order_type == ORDER_TYPE_BUY)
   {
      result = trade.Buy(risk_lots, _Symbol, 0, sl_price, tp_price, "LightningScalp Buy");
   }
   else // SELL
   {
      result = trade.Sell(risk_lots, _Symbol, 0, sl_price, tp_price, "LightningScalp Sell");
   }
   
   // Record trade info if successful
   if(result)
   {
      // Update rate limiting variables
      last_order_time = TimeCurrent();
      orders_this_minute++;
      total_trades++;
      
      // Log trade details
      Print("New position opened: ", EnumToString(order_type), 
            ", Lots: ", DoubleToString(risk_lots, 2),
            ", Entry: ", DoubleToString(entry_price, _Digits),
            ", SL: ", DoubleToString(sl_price, _Digits),
            ", TP: ", DoubleToString(tp_price, _Digits),
            ", Commission: ", DoubleToString(expected_commission, 2),
            ", Potential net profit: ", DoubleToString(profit_after_commission, 2));
   }
   else
   {
      Print("Error opening position: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Manage open positions (trailing stops, partial closes)           |
//+------------------------------------------------------------------+
void ManageOpenPositions()
{
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelect(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != 987654)
         continue;
      datetime pos_time = (datetime)PositionGetInteger(POSITION_TIME);
      // If position is open for more than 5 seconds, close it
      if(TimeCurrent() - pos_time > 5)
      {
         string symbol = PositionGetString(POSITION_SYMBOL);
         if(!trade.PositionClose(symbol))
         {
            Print("Failed to close position on ", symbol, ". Error: ", GetLastError());
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check order rate limits                                          |
//+------------------------------------------------------------------+
bool CheckOrderRateLimit()
{
   datetime current_time = TimeCurrent();
   
   // Check new minute
   MqlDateTime time_struct;
   TimeToStruct(current_time, time_struct);
   
   // Create time representing just the current minute
   datetime this_minute = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d:00", 
                           time_struct.year, time_struct.mon, time_struct.day, 
                           time_struct.hour, time_struct.min));
   
   // Reset counter on new minute
   if(this_minute != current_minute)
   {
      if(EnableDebugMode && orders_this_minute > 0) 
         Print("New minute: Resetting orders counter from ", orders_this_minute, " to 0");
      
      current_minute = this_minute;
      orders_this_minute = 0;
      last_order_time = 0; // Reset last order time on new minute
   }
   
   // Check if we've hit the rate limit
   if(orders_this_minute >= OrdersPerMinute)
   {
      return false;
   }
   
   // Reduced minimum delay between trades
   if(last_order_time > 0 && current_time - last_order_time < 1) // 1 second minimum between orders
   {
      return false;
   }
   
   return true;
}

//+------------------------------------------------------------------+
//| Check for new trading day                                        |
//+------------------------------------------------------------------+
void CheckNewDay()
{
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Convert to datetime for day comparison
   datetime current_day = StringToTime(StringFormat("%04d.%02d.%02d 00:00:00", 
                                        current_time.year, current_time.mon, current_time.day));
   
   // If it's a new day
   if(current_day > last_day)
   {
      if(last_day > 0) // Not the first run
      {
         // Log daily results
         Print("Daily Summary for ", TimeToString(last_day, TIME_DATE), 
               ": P/L = ", DoubleToString(daily_profit_loss, 2), 
               ", Trades: ", total_trades);
      }
      
      // Reset daily tracking
      daily_profit_loss = 0;
      daily_loss_reached = false;
      eod_closure_done = false;
      last_day = current_day;
   }
}

//+------------------------------------------------------------------+
//| Check if it's time to close positions for end of day             |
//+------------------------------------------------------------------+
void CheckEndOfDay()
{
   // Skip if already done for today
   if(eod_closure_done)
      return;
      
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Check if it's EOD time
   bool is_eod = (current_time.hour == EOD_Hour && 
                 current_time.min >= EOD_Minute);
   
   if(is_eod)
   {
      // Close all positions
      int closed_count = CloseAllPositions();
      if(closed_count > 0)
      {
         Print("End of day: Closed ", closed_count, " positions");
      }
      
      // Set flag to avoid repeated closures
      eod_closure_done = true;
   }
}

//+------------------------------------------------------------------+
//| Close all positions                                              |
//+------------------------------------------------------------------+
int CloseAllPositions()
{
   int closed_count = 0;
   
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == trade.RequestMagic())
         {
            if(trade.PositionClose(ticket))
            {
               closed_count++;
            }
            else
            {
               Print("Failed to close position ", ticket, ". Error: ", GetLastError());
            }
         }
      }
   }
   
   return closed_count;
}

//+------------------------------------------------------------------+
//| Check if we're in trading hours                                  |
//+------------------------------------------------------------------+
bool IsInTradingHours()
{
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Check if current hour is within active hours
   return (current_time.hour >= ActiveHourStart && current_time.hour < ActiveHourEnd);
}

//+------------------------------------------------------------------+
//| Update risk metrics                                              |
//+------------------------------------------------------------------+
void UpdateRiskMetrics()
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Update maximum equity if we've reached a new high
   if(account_equity > max_equity)
      max_equity = account_equity;
   
   // Calculate current drawdown from maximum equity
   if(max_equity > 0)
      current_drawdown = ((max_equity - account_equity) / max_equity) * 100.0;
   else
      current_drawdown = 0;
   
   // Update daily profit/loss
   daily_profit_loss = account_equity - account_balance;
   
   // Check if daily loss limit is reached
   if(!daily_loss_reached)
   {
      double daily_loss_percent = (daily_profit_loss / account_balance) * 100.0;
      
      if(daily_loss_percent < -MaxDailyLoss)
      {
         Print("Daily loss limit reached: ", DoubleToString(daily_loss_percent, 2), "%. Closing all positions.");
         CloseAllPositions();
         daily_loss_reached = true;
      }
   }
   
   // Check if maximum drawdown is reached
   if(!max_drawdown_reached && current_drawdown > MaxDrawdown)
   {
      Print("Maximum drawdown limit reached: ", DoubleToString(current_drawdown, 2), "%. Closing all positions.");
      CloseAllPositions();
      max_drawdown_reached = true;
   }
}

//+------------------------------------------------------------------+
//| OnTradeTransaction function to track closed positions            |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
{
   // Check for position close event
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      // Get the deal ticket to retrieve its profit
      ulong deal_ticket = trans.deal;
      
      if(deal_ticket > 0 && HistoryDealSelect(deal_ticket))
      {
         // Check if this deal closed a position
         if(HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
         {
            double deal_profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
            double deal_commission = HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);
            double deal_swap = HistoryDealGetDouble(deal_ticket, DEAL_SWAP);
            double total_profit = deal_profit + deal_commission + deal_swap;
            
            // Update win/loss counters
            if(total_profit > 0)
            {
               winning_trades++;
               consecutive_losses = 0;
            }
            else
            {
               losing_trades++;
               consecutive_losses++;
            }
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Update display information                                       |
//+------------------------------------------------------------------+
void UpdateDashboard()
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double profit_today = account_equity - account_balance + daily_profit_loss;
   
   // Calculate win rate
   double win_rate = (total_trades > 0) ? ((double)winning_trades / (double)total_trades) * 100.0 : 0;
   
   // Average latency calculation
   int orders_per_min_avg = (TimeCurrent() - last_order_time < 3600) ? orders_this_minute : 0;
   
   // Chart comment with statistics
   string comment = "LightningScalp EA - Latency: " + IntegerToString(LatencyMs) + "ms\n";
   comment += "Account: $" + DoubleToString(account_equity, 2) + " (Balance: $" + DoubleToString(account_balance, 2) + ")\n";
   comment += "Today's P/L: $" + DoubleToString(profit_today, 2) + "\n";
   comment += "Drawdown: " + DoubleToString(current_drawdown, 2) + "%\n";
   comment += "Open Positions: " + IntegerToString(CountOpenPositions()) + "/" + IntegerToString(MaxSimultaneousOrders) + "\n";
   comment += "Total Trades: " + IntegerToString(total_trades) + " (Win: " + DoubleToString(win_rate, 1) + "%)\n";
   comment += "Orders This Minute: " + IntegerToString(orders_this_minute) + "/" + IntegerToString(OrdersPerMinute) + "\n";
   comment += "Status: " + (daily_loss_reached ? "DAILY LOSS LIMIT" : 
                           (max_drawdown_reached ? "MAX DRAWDOWN" : "ACTIVE"));
   
   ChartSetString(0, CHART_COMMENT, comment);
}

//+------------------------------------------------------------------+
//| Count Open Positions for EA                                      |
//+------------------------------------------------------------------+
int CountOpenPositions()
{
   int count = 0;
   int total = PositionsTotal();
   for(int i = 0; i < total; i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0) continue;
      if(!PositionSelect(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) == 987654)
         count++;
   }
   return count;
}

//+------------------------------------------------------------------+ 