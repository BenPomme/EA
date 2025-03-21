//+------------------------------------------------------------------+
//|                                                VolatilityForce.mq5 |
//|                            Copyright 2023, VolatilityForce System  |
//|                                         https://www.volatility.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, VolatilityForce System"
#property link      "https://www.volatility.com"
#property version   "0.20"
#property description "High-Frequency Scalping System for EURUSD"
#property description "Initial account: $5000, max leverage: 500:1"
#property description "Target: Hundreds of trades per year with positive expectancy"

// Include necessary libraries
#include <Trade/Trade.mqh>
#include <Trade/SymbolInfo.mqh>
#include <Trade/PositionInfo.mqh>

// Input parameters - Trading Timeframe
input ENUM_TIMEFRAMES Trading_Timeframe = PERIOD_M5; // Main trading timeframe (lower for higher frequency)

// Input parameters - Strategy parameters
input int      ATR_Period = 5;            // ATR Period (shorter for faster response)
input double   ATR_Multiplier = 0.5;      // ATR Multiplier for entries (more sensitive)
input int      RSI_Period = 5;            // RSI Period for confirmation (shorter)
input int      SMA_Period = 10;           // SMA Period for trend direction (shorter)
input bool     UseBreakoutStrategy = true; // Use price breakout strategy
input bool     UseMomentumStrategy = true; // Use momentum strategy
input bool     UseRangeStrategy = true;    // Use range-bound strategy

// Input parameters - Money Management
input double   Risk_Percent = 0.5;        // Risk percent per trade (smaller for more trades)
input double   TP_ATR_Multiplier = 0.8;   // Take Profit ATR multiplier (smaller for quicker profits)
input double   SL_ATR_Multiplier = 0.6;   // Stop Loss ATR multiplier (tighter)
input double   MaxDrawdownPercent = 15.0; // Maximum allowed drawdown (%)
input double   DailyLossLimitPercent = 5.0; // Daily loss limit (% of account)

// Input parameters - Trade Management
input int      Slippage = 10;             // Maximum allowed slippage in points
input bool     UseTrailingStop = true;    // Use trailing stop feature
input double   TrailingATR = 0.3;         // Trailing stop ATR multiplier (tighter)
input double   MaxLeverage = 500;         // Maximum leverage to use
input int      MaxActivePositions = 8;    // Maximum number of active positions at once
input int      TradeDelay = 60;           // Delay between trades in seconds (1 minute)

// Input parameters - Session Management
input bool     ClosePositionsEndOfDay = true; // Close all positions at end of day
input int      EndOfDayHour = 23;         // Hour to close positions (0-23)
input int      EndOfDayMinute = 0;        // Minute to close positions (0-59)
input bool     NoTradingFriday = true;    // Stop opening new positions on Friday
input bool     SessionTrading = true;     // Only trade during specific sessions
input int      SessionStartHour = 8;      // Session start hour (0-23)
input int      SessionEndHour = 20;       // Session end hour (0-23)

// Global variables - Trade management
CTrade         trade;
CSymbolInfo    symbolInfo;
CPositionInfo  positionInfo;
int            atr_handle;
int            rsi_handle;
int            sma_handle;
int            bb_handle;
double         atr_buffer[];
double         rsi_buffer[];
double         sma_buffer[];
double         bb_upper[];
double         bb_lower[];
double         bb_middle[];
double         high_buffer[];
double         low_buffer[];
datetime       last_trade_time = 0;
ulong          last_ticket = 0;
int            trade_cooldown = 60;   // 1 minute cooldown between trades
int            trades_today = 0;      // Counter for today's trades

// Global variables - Risk management
double         starting_balance = 0;     // Account balance at EA start
double         daily_profit_loss = 0;    // Track daily P/L
double         max_equity = 0;           // Track maximum equity
double         current_drawdown = 0;     // Current drawdown percentage
datetime       last_day = 0;             // Last trading day
bool           daily_limit_reached = false; // Flag for daily loss limit
bool           max_drawdown_reached = false; // Flag for max drawdown
bool           eod_closure_done = false; // Flag for end-of-day closure
int            total_trades = 0;         // Total trades taken
int            winning_trades = 0;       // Number of winning trades
int            losing_trades = 0;        // Number of losing trades

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize symbol info
   if(!symbolInfo.Name(_Symbol))
   {
      Print("Failed to initialize symbol info");
      return INIT_FAILED;
   }
   
   // Set trading parameters
   trade.SetExpertMagicNumber(123456);
   trade.SetDeviationInPoints(Slippage);
   trade.SetMarginMode();
   trade.SetTypeFillingBySymbol(symbolInfo.Name());
   
   // Create indicator handles with the specified timeframe
   atr_handle = iATR(_Symbol, Trading_Timeframe, ATR_Period);
   rsi_handle = iRSI(_Symbol, Trading_Timeframe, RSI_Period, PRICE_CLOSE);
   sma_handle = iMA(_Symbol, Trading_Timeframe, SMA_Period, 0, MODE_SMA, PRICE_CLOSE);
   bb_handle = iBands(_Symbol, Trading_Timeframe, 20, 2.0, 0, PRICE_CLOSE);
   
   if(atr_handle == INVALID_HANDLE || rsi_handle == INVALID_HANDLE || 
      sma_handle == INVALID_HANDLE || bb_handle == INVALID_HANDLE)
   {
      Print("Error creating indicator handles");
      return INIT_FAILED;
   }
   
   // Initialize buffers
   ArraySetAsSeries(atr_buffer, true);
   ArraySetAsSeries(rsi_buffer, true);
   ArraySetAsSeries(sma_buffer, true);
   ArraySetAsSeries(bb_upper, true);
   ArraySetAsSeries(bb_lower, true);
   ArraySetAsSeries(bb_middle, true);
   ArraySetAsSeries(high_buffer, true);
   ArraySetAsSeries(low_buffer, true);
   
   // Initialize risk management variables
   starting_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   max_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   last_day = 0; // Will be set on first tick
   daily_profit_loss = 0;
   daily_limit_reached = false;
   max_drawdown_reached = false;
   eod_closure_done = false;
   trades_today = 0;
   total_trades = 0;
   winning_trades = 0;
   losing_trades = 0;
   
   // Update trade cooldown based on input
   trade_cooldown = TradeDelay;
   
   // Set chart name
   ChartSetString(0, CHART_COMMENT, "VolatilityForce v0.2 HF Scalping EA");
   
   Print("VolatilityForce v0.2 HF Scalping EA initialized. Targeting EURUSD with multi-strategy approach on ", 
          EnumToString(Trading_Timeframe), " timeframe");
   return INIT_SUCCEEDED;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   if(sma_handle != INVALID_HANDLE) IndicatorRelease(sma_handle);
   if(bb_handle != INVALID_HANDLE) IndicatorRelease(bb_handle);
   
   // Log performance metrics
   if(total_trades > 0)
   {
      double win_rate = (double)winning_trades / (double)total_trades * 100.0;
      Print("VolatilityForce v0.2 Performance Summary:");
      Print("Total Trades: ", total_trades);
      Print("Winning Trades: ", winning_trades, " (", DoubleToString(win_rate, 2), "%)");
      Print("Losing Trades: ", losing_trades);
   }
   
   Print("VolatilityForce v0.2 EA deinitialized. Reason: ", reason);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update symbol info
   if(!symbolInfo.RefreshRates())
      return;
   
   // Check for new day - reset daily P/L tracking
   CheckNewDay();
   
   // Check if we need to close positions at end of day
   if(ClosePositionsEndOfDay)
      CheckEndOfDay();
   
   // Update drawdown monitoring
   UpdateDrawdownStatus();
   
   // Check if we should stop trading due to risk management
   if(daily_limit_reached || max_drawdown_reached)
      return;
      
   // Check if we're in the specified trading session
   if(SessionTrading && !IsInTradingSession())
      return;
      
   // Copy indicator values
   if(CopyBuffer(atr_handle, 0, 0, 10, atr_buffer) <= 0 ||
      CopyBuffer(rsi_handle, 0, 0, 10, rsi_buffer) <= 0 ||
      CopyBuffer(sma_handle, 0, 0, 10, sma_buffer) <= 0 ||
      CopyBuffer(bb_handle, 0, 0, 10, bb_upper) <= 0 ||
      CopyBuffer(bb_handle, 1, 0, 10, bb_middle) <= 0 ||
      CopyBuffer(bb_handle, 2, 0, 10, bb_lower) <= 0 ||
      CopyHigh(_Symbol, Trading_Timeframe, 0, 10, high_buffer) <= 0 ||
      CopyLow(_Symbol, Trading_Timeframe, 0, 10, low_buffer) <= 0)
   {
      Print("Error copying indicator buffers");
      return;
   }
   
   // Check if we have enough data
   if(atr_buffer[0] == 0 || rsi_buffer[0] == 0 || sma_buffer[0] == 0)
      return;
   
   // Get current positions count for this symbol
   int active_positions = CountSymbolPositions();
   
   // Manage trailing stops for existing positions
   if(UseTrailingStop && active_positions > 0)
      ManageAllTrailingStops();
   
   // Check if we can open a new position
   bool can_trade = active_positions < MaxActivePositions && 
                    TimeCurrent() - last_trade_time > trade_cooldown && 
                    !IsFridayEvening() && 
                    !IsApproachingEndOfDay();
   
   if(can_trade)
   {
      // Use multiple strategies to find trading opportunities
      if(UseBreakoutStrategy)
         AnalyzeBreakoutStrategy();
         
      if(UseMomentumStrategy && active_positions < MaxActivePositions)
         AnalyzeMomentumStrategy();
         
      if(UseRangeStrategy && active_positions < MaxActivePositions)
         AnalyzeRangeStrategy();
   }
   
   // Calculate and display risk metrics periodically
   CalculateRisk();
}

//+------------------------------------------------------------------+
//| Check if current time is in the trading session                  |
//+------------------------------------------------------------------+
bool IsInTradingSession()
{
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Check if current hour is within session hours
   return (current_time.hour >= SessionStartHour && current_time.hour < SessionEndHour);
}

//+------------------------------------------------------------------+
//| Manage trailing stops for all positions                          |
//+------------------------------------------------------------------+
void ManageAllTrailingStops()
{
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == trade.RequestMagic())
         {
            ManageTrailingStop(ticket);
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check for a new trading day and reset daily variables            |
//+------------------------------------------------------------------+
void CheckNewDay()
{
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Convert current date to a datetime for comparison
   datetime current_day = StringToTime(StringFormat("%04d.%02d.%02d 00:00:00", 
                                        current_time.year, current_time.mon, current_time.day));
   
   // If we're in a new day
   if(current_day > last_day)
   {
      if(last_day > 0) // Not the first run
      {
         // Log the daily results
         Print("Daily Summary for ", TimeToString(last_day, TIME_DATE), 
               ": P/L = ", DoubleToString(daily_profit_loss, 2), 
               ", Trades = ", trades_today);
      }
      
      // Reset daily tracking
      daily_profit_loss = 0;
      daily_limit_reached = false;
      eod_closure_done = false;
      trades_today = 0;
      last_day = current_day;
   }
}

//+------------------------------------------------------------------+
//| Check if it's time to close all positions for end of day         |
//+------------------------------------------------------------------+
void CheckEndOfDay()
{
   // Only check if we haven't already closed positions today
   if(eod_closure_done)
      return;
      
   // Get current time
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Check if it's time to close positions
   bool is_end_of_day = (current_time.hour == EndOfDayHour && 
                         current_time.min >= EndOfDayMinute);
   
   if(is_end_of_day)
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
//| Close all open positions                                         |
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
//| Update drawdown status and check for limits                      |
//+------------------------------------------------------------------+
void UpdateDrawdownStatus()
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
   
   // Check if daily loss limit is reached (only if we have open positions)
   if(!daily_limit_reached && CountSymbolPositions() > 0)
   {
      double daily_loss_percent = (daily_profit_loss / account_balance) * 100.0;
      
      if(daily_loss_percent < -DailyLossLimitPercent)
      {
         Print("Daily loss limit reached: ", DoubleToString(daily_loss_percent, 2), "%. Closing all positions.");
         CloseAllPositions();
         daily_limit_reached = true;
      }
   }
   
   // Check if maximum drawdown is reached
   if(!max_drawdown_reached && current_drawdown > MaxDrawdownPercent)
   {
      Print("Maximum drawdown limit reached: ", DoubleToString(current_drawdown, 2), "%. Closing all positions.");
      CloseAllPositions();
      max_drawdown_reached = true;
   }
}

//+------------------------------------------------------------------+
//| Check if it's Friday evening - no new positions                  |
//+------------------------------------------------------------------+
bool IsFridayEvening()
{
   if(!NoTradingFriday)
      return false;
      
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Don't open new positions on Friday after 12:00
   return (current_time.day_of_week == 5 && current_time.hour >= 12);
}

//+------------------------------------------------------------------+
//| Check if we're approaching end of day                            |
//+------------------------------------------------------------------+
bool IsApproachingEndOfDay()
{
   if(!ClosePositionsEndOfDay)
      return false;
      
   MqlDateTime current_time;
   TimeToStruct(TimeCurrent(), current_time);
   
   // Don't open new positions within 30 minutes of end-of-day closure
   if(current_time.hour == EndOfDayHour)
   {
      if(EndOfDayMinute >= 30 && current_time.min >= EndOfDayMinute - 30)
         return true;
      else if(EndOfDayMinute < 30 && current_time.min >= EndOfDayMinute + 30)
         return true;
   }
   else if(current_time.hour == EndOfDayHour - 1 && EndOfDayMinute < 30)
   {
      if(current_time.min >= 60 - (30 - EndOfDayMinute))
         return true;
   }
   
   return false;
}

//+------------------------------------------------------------------+
//| STRATEGY 1: Breakout strategy based on volatility                |
//+------------------------------------------------------------------+
void AnalyzeBreakoutStrategy()
{
   // Volatility measurement
   double current_atr = atr_buffer[0];
   double prev_atr = atr_buffer[1];
   double atr_change = current_atr / prev_atr;
   
   // Price data
   double current_price = symbolInfo.Ask();
   double recent_high = high_buffer[ArrayMaximum(high_buffer, 0, 5)];
   double recent_low = low_buffer[ArrayMinimum(low_buffer, 0, 5)];
   
   // Bollinger Band data
   double upper_band = bb_upper[0];
   double lower_band = bb_lower[0];
   
   // Breakout conditions - volatility increasing
   bool volatility_increasing = atr_change > 1.02; // 2% increase in volatility
   
   // Bullish Breakout: Price breaks above recent high with increasing volatility
   if(volatility_increasing && current_price > recent_high)
   {
      double rsi = rsi_buffer[0];
      // Confirm with RSI (not extremely overbought)
      if(rsi < 70)
      {
         OpenPosition(ORDER_TYPE_BUY);
         return;
      }
   }
   
   // Bearish Breakout: Price breaks below recent low with increasing volatility
   if(volatility_increasing && symbolInfo.Bid() < recent_low)
   {
      double rsi = rsi_buffer[0];
      // Confirm with RSI (not extremely oversold)
      if(rsi > 30)
      {
         OpenPosition(ORDER_TYPE_SELL);
         return;
      }
   }
   
   // Bollinger Band Breakouts (additional trigger)
   if(current_price > upper_band + (current_atr * 0.1))
   {
      OpenPosition(ORDER_TYPE_BUY);
      return;
   }
   else if(symbolInfo.Bid() < lower_band - (current_atr * 0.1))
   {
      OpenPosition(ORDER_TYPE_SELL);
      return;
   }
}

//+------------------------------------------------------------------+
//| STRATEGY 2: Momentum-based strategy                              |
//+------------------------------------------------------------------+
void AnalyzeMomentumStrategy()
{
   // RSI values
   double current_rsi = rsi_buffer[0];
   double prev_rsi = rsi_buffer[1];
   double prev_prev_rsi = rsi_buffer[2];
   
   // Price relative to SMA
   double current_price = symbolInfo.Ask();
   double sma_value = sma_buffer[0];
   bool price_above_sma = current_price > sma_value;
   
   // RSI Momentum: Measure rate of change in RSI
   double rsi_momentum = current_rsi - prev_rsi;
   double prev_rsi_momentum = prev_rsi - prev_prev_rsi;
   
   // Bullish Momentum: RSI rising with accelerating momentum
   if(rsi_momentum > 0 && rsi_momentum > prev_rsi_momentum)
   {
      // RSI in optimal range for buy (not overbought, showing strength)
      if(current_rsi > 45 && current_rsi < 70)
      {
         // Price confirms by being above SMA
         if(price_above_sma)
         {
            OpenPosition(ORDER_TYPE_BUY);
            return;
         }
      }
   }
   
   // Bearish Momentum: RSI falling with accelerating momentum
   if(rsi_momentum < 0 && rsi_momentum < prev_rsi_momentum)
   {
      // RSI in optimal range for sell (not oversold, showing weakness)
      if(current_rsi < 55 && current_rsi > 30)
      {
         // Price confirms by being below SMA
         if(!price_above_sma)
         {
            OpenPosition(ORDER_TYPE_SELL);
            return;
         }
      }
   }
   
   // RSI Divergence (very simplified implementation)
   // Classic bullish divergence: Price makes lower low but RSI makes higher low
   if(!price_above_sma && low_buffer[1] < low_buffer[3] && rsi_buffer[1] > rsi_buffer[3] && current_rsi > prev_rsi)
   {
      OpenPosition(ORDER_TYPE_BUY);
      return;
   }
   
   // Classic bearish divergence: Price makes higher high but RSI makes lower high
   if(price_above_sma && high_buffer[1] > high_buffer[3] && rsi_buffer[1] < rsi_buffer[3] && current_rsi < prev_rsi)
   {
      OpenPosition(ORDER_TYPE_SELL);
      return;
   }
}

//+------------------------------------------------------------------+
//| STRATEGY 3: Range-bound strategy                                 |
//+------------------------------------------------------------------+
void AnalyzeRangeStrategy()
{
   // Bollinger Band data
   double upper_band = bb_upper[0];
   double middle_band = bb_middle[0];
   double lower_band = bb_lower[0];
   double band_width = upper_band - lower_band;
   
   // Price data
   double current_price = symbolInfo.Ask();
   double current_bid = symbolInfo.Bid();
   
   // Calculate percent of price within bands
   double price_position = (current_price - lower_band) / band_width;
   
   // RSI data
   double current_rsi = rsi_buffer[0];
   
   // Band Contraction: Looking for narrow bands indicating low volatility
   double prev_band_width = bb_upper[3] - bb_lower[3];
   bool bands_contracting = band_width < prev_band_width;
   
   // Oversold near lower band
   if(price_position < 0.2 && current_rsi < 40 && bands_contracting)
   {
      // Price near lower band and RSI oversold but not extremely
      if(current_rsi > 30 && current_rsi < 40)
      {
         OpenPosition(ORDER_TYPE_BUY);
         return;
      }
   }
   
   // Overbought near upper band
   if(price_position > 0.8 && current_rsi > 60 && bands_contracting)
   {
      // Price near upper band and RSI overbought but not extremely
      if(current_rsi < 70 && current_rsi > 60)
      {
         OpenPosition(ORDER_TYPE_SELL);
         return;
      }
   }
   
   // Mean reversion: Price distance from middle band with RSI confirmation
   double middle_distance = MathAbs(current_price - middle_band) / band_width;
   
   if(middle_distance > 0.4) // Price significantly away from middle band
   {
      // Price above middle, RSI turning down from overbought zone
      if(current_price > middle_band && current_rsi > 60 && current_rsi < rsi_buffer[1])
      {
         OpenPosition(ORDER_TYPE_SELL);
         return;
      }
      
      // Price below middle, RSI turning up from oversold zone
      if(current_bid < middle_band && current_rsi < 40 && current_rsi > rsi_buffer[1])
      {
         OpenPosition(ORDER_TYPE_BUY);
         return;
      }
   }
}

//+------------------------------------------------------------------+
//| Open a new position                                             |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE order_type)
{
   double entry_price = (order_type == ORDER_TYPE_BUY) ? symbolInfo.Ask() : symbolInfo.Bid();
   
   // Calculate stop loss and take profit levels based on ATR
   double stop_loss = 0, take_profit = 0;
   double atr_value = atr_buffer[0];
   
   if(order_type == ORDER_TYPE_BUY)
   {
      stop_loss = entry_price - (atr_value * SL_ATR_Multiplier);
      take_profit = entry_price + (atr_value * TP_ATR_Multiplier);
   }
   else
   {
      stop_loss = entry_price + (atr_value * SL_ATR_Multiplier);
      take_profit = entry_price - (atr_value * TP_ATR_Multiplier);
   }
   
   // Calculate position size based on risk percentage
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (Risk_Percent / 100.0);
   double point_value = symbolInfo.TickValue() * symbolInfo.Point();
   
   // Calculate stop loss in points
   double sl_points = MathAbs(entry_price - stop_loss) / symbolInfo.Point();
   
   // Calculate lot size based on risk
   double lot_size = NormalizeDouble(risk_amount / (sl_points * point_value), 2);
   double min_lot = symbolInfo.LotsMin();
   double max_lot = symbolInfo.LotsMax();
   double lot_step = symbolInfo.LotsStep();
   
   // Normalize lot size to broker's requirements
   lot_size = MathMin(max_lot, MathMax(min_lot, NormalizeDouble(lot_size / lot_step, 0) * lot_step));

   // Check leverage constraint
   double max_allowed_lot = (account_balance * MaxLeverage) / (entry_price * 100000);
   lot_size = MathMin(lot_size, max_allowed_lot);
   
   // Scale down lot size for high-frequency trading to control overall exposure
   lot_size = lot_size * 0.6; // Using only 60% of calculated risk
   lot_size = NormalizeDouble(lot_size, 2);
   
   // Execute the trade
   bool result = false;
   if(order_type == ORDER_TYPE_BUY)
   {
      result = trade.Buy(lot_size, _Symbol, 0, stop_loss, take_profit, "VolatilityForce Buy");
   }
   else
   {
      result = trade.Sell(lot_size, _Symbol, 0, stop_loss, take_profit, "VolatilityForce Sell");
   }
   
   // Record trade time if successful
   if(result)
   {
      last_trade_time = TimeCurrent();
      last_ticket = trade.ResultOrder();
      trades_today++;
      total_trades++;
      
      Print("Order opened: ", EnumToString(order_type), " Lot Size: ", lot_size, 
            " SL: ", stop_loss, " TP: ", take_profit, 
            " ATR: ", DoubleToString(atr_value, 5));
   }
   else
   {
      Print("Error opening position: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Manage trailing stops for open positions                         |
//+------------------------------------------------------------------+
void ManageTrailingStop(ulong ticket)
{
   if(!PositionSelectByTicket(ticket))
      return;
      
   double position_sl = PositionGetDouble(POSITION_SL);
   double position_tp = PositionGetDouble(POSITION_TP);
   double position_price_open = PositionGetDouble(POSITION_PRICE_OPEN);
   ENUM_POSITION_TYPE position_type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   
   double current_price = (position_type == POSITION_TYPE_BUY) ? symbolInfo.Bid() : symbolInfo.Ask();
   double new_sl = 0;
   
   double atr_value = atr_buffer[0];
   double trailing_distance = atr_value * TrailingATR;
   
   // Calculate new stop loss based on position type
   if(position_type == POSITION_TYPE_BUY)
   {
      // For long positions, move stop loss up if price increases
      new_sl = current_price - trailing_distance;
      
      // Only move stop loss if it's above the current stop loss and we're in profit
      if(new_sl > position_sl && current_price > position_price_open)
      {
         if(trade.PositionModify(ticket, new_sl, position_tp))
         {
            Print("Trailing stop updated for Buy position. New SL: ", new_sl);
         }
      }
   }
   else // POSITION_TYPE_SELL
   {
      // For short positions, move stop loss down if price decreases
      new_sl = current_price + trailing_distance;
      
      // Only move stop loss if it's below the current stop loss and we're in profit
      if((new_sl < position_sl || position_sl == 0) && current_price < position_price_open)
      {
         if(trade.PositionModify(ticket, new_sl, position_tp))
         {
            Print("Trailing stop updated for Sell position. New SL: ", new_sl);
         }
      }
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
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD && 
      trans.deal_type == DEAL_TYPE_SELL &&
      trans.order_type == ORDER_TYPE_CLOSE_BY)
   {
      // Position has been closed, update stats
      double deal_profit = 0;
      
      // Get the deal ticket to retrieve its profit
      ulong deal_ticket = trans.deal;
      if(deal_ticket > 0 && HistoryDealSelect(deal_ticket))
      {
         deal_profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
         
         // Update win/loss counters
         if(deal_profit > 0)
            winning_trades++;
         else
            losing_trades++;
      }
   }
}

//+------------------------------------------------------------------+
//| Calculate account risk metrics                                   |
//+------------------------------------------------------------------+
void CalculateRisk()
{
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double account_margin = AccountInfoDouble(ACCOUNT_MARGIN);
   
   // Calculate margin level
   double margin_level = (account_margin > 0) ? (account_equity / account_margin) * 100.0 : 0;
   
   // Calculate drawdown
   double drawdown = (account_balance > 0) ? ((account_balance - account_equity) / account_balance) * 100.0 : 0;
   
   // Log risk metrics less frequently
   if(TimeCurrent() % 3600 == 0) // Log once per hour
   {
      Print("Risk Metrics - Balance: ", account_balance, 
            " Equity: ", account_equity, 
            " Margin: ", account_margin,
            " Margin Level: ", NormalizeDouble(margin_level, 2), "%",
            " Drawdown: ", NormalizeDouble(drawdown, 2), "%",
            " Daily P/L: ", DoubleToString(daily_profit_loss, 2),
            " Today's Trades: ", trades_today);
   }
   
   // Calculate win rate
   double win_rate = (total_trades > 0) ? (double)winning_trades / (double)total_trades * 100.0 : 0;
   
   // Update chart comment with basic stats
   string comment = "VolatilityForce v0.2 HF Scalping EA\n";
   comment += "Timeframe: " + EnumToString(Trading_Timeframe) + "\n";
   comment += "Active positions: " + IntegerToString(CountSymbolPositions()) + "/" + IntegerToString(MaxActivePositions) + "\n";
   comment += "Today's Trades: " + IntegerToString(trades_today) + "\n";
   comment += "Total Trades: " + IntegerToString(total_trades) + "\n";
   comment += "Win Rate: " + DoubleToString(win_rate, 1) + "%\n";
   comment += "Balance: $" + DoubleToString(account_balance, 2) + "\n";
   comment += "Daily P/L: $" + DoubleToString(daily_profit_loss, 2) + "\n";
   comment += "Current Drawdown: " + DoubleToString(current_drawdown, 2) + "%\n";
   comment += "Status: " + (daily_limit_reached ? "DAILY LIMIT REACHED" : 
                          (max_drawdown_reached ? "MAX DRAWDOWN REACHED" : "ACTIVE"));
   
   ChartSetString(0, CHART_COMMENT, comment);
}

//+------------------------------------------------------------------+
//| Count positions for the current symbol                          |
//+------------------------------------------------------------------+
int CountSymbolPositions()
{
   int count = 0;
   for(int i = 0; i < PositionsTotal(); i++)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol)
         {
            count++;
         }
      }
   }
   return count;
}

//+------------------------------------------------------------------+ 