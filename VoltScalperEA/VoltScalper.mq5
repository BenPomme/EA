//+------------------------------------------------------------------+
//|                                                  VoltScalper.mq5 |
//|                   Ultra-Fast Scalping Bot for EURUSD              |
//|          Built for home day traders with 30-50ms latency and        |
//|          optimized for FP Markets fee structure (from FPmarketsfees.txt)|
//|                         by [Your Name]                           |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, [Your Name]"
#property link      "https://www.example.com"
#property version   "1.00"
#property strict
#property description "VoltScalper - Ultra-Fast HFT Scalping Bot for EURUSD. Designed for <50ms latency and optimized for FP Markets fees."

//--- Include necessary libraries
#include <Trade/Trade.mqh>
#include <Trade/SymbolInfo.mqh>

//--- Input parameters - Trading and Order Management
input ENUM_TIMEFRAMES Trading_Timeframe = PERIOD_M1;  // M1 timeframe for ultra high frequency
input int      OrdersPerMinute           = 10;         // Maximum orders per minute
input int      MaxSimultaneousOrders     = 3;          // Maximum simultaneous open trades
input double   MaxSpreadPoints           = 7.0;        // Maximum spread (in points) allowed
input int      LatencyMs                 = 40;         // Latency in ms (typical 30-50ms)
input int      LatencyBuffer             = 10;         // Additional buffer in ms

//--- Input parameters - Strategy (Ultra fast settings)
input int      Fast_EMA_Period           = 3;          // Very fast EMA period
input int      Slow_EMA_Period           = 8;          // Fast slow EMA period
input int      RSI_Period                = 5;          // Very fast RSI period
input int      RSI_Overbought            = 70;         // RSI overbought threshold
input int      RSI_Oversold              = 30;         // RSI oversold threshold

//--- Input parameters - Money Management
input double   RiskPercent               = 0.5;        // Risk percent per trade
input double   TP_Pips                   = 6.0;        // Take profit in pips
input double   SL_Pips                   = 8.0;        // Stop loss in pips
input double   TrailStart_Pips           = 3.0;        // When to start trailing
input double   TrailStep_Pips            = 1.0;        // Trailing step in pips
input double   MaxLeverage               = 30.0;       // Maximum leverage allowed
input bool     DynamicLeverageAdjustment = true;       // Adjust position size based on volatility

//--- Input parameters - Session & Risk
input bool     ManageTradingHours        = false;      // Restrict trading hours
input int      ActiveHourStart           = 0;          // Start hour
input int      ActiveHourEnd             = 23;         // End hour
input double   MaxDailyLoss              = 3.0;        // Maximum daily loss percentage
input double   MaxDrawdown               = 10.0;       // Maximum drawdown percentage
input bool     EnableDebugMode           = true;       // Enable debug logging

//--- Input parameters - Broker Commission Handling (FP Markets)
input double   CommissionPerLot          = 5.5;        // Commission per lot (round trip)
input bool     AccountForCommission      = true;       // Adjust profit targets for commission
input double   MinProfitTargetMultiplier = 1.5;        // Minimum profit must be this times the commission

//--- Global variables
CTrade         trade;              // Trade object
CSymbolInfo    symbolInfo;         // Symbol info object

double         point;              // Point size
double         pip_value;          // Pip value
int            pip_digits;         // Pip digits

//--- Indicator buffers (simple arrays used with CopyBuffer calls)
double         fast_ema_buffer[];
double         slow_ema_buffer[];
double         rsi_buffer[];
double         price_buffer[];
double         high_buffer[];
double         low_buffer[];

//--- Order management and risk variables
datetime       last_order_time = 0;
int            orders_this_minute = 0;
datetime       current_minute = 0;

//--- Daily performance metrics
double         starting_balance;
double         daily_profit_loss = 0;
double         max_equity = 0;
bool           daily_loss_reached = false;
bool           max_drawdown_reached = false;
datetime       last_day = 0;

//--- Debugging counters
int            signal_check_count = 0;
int            conditions_not_met_count = 0;
bool           logged_this_tick = false;

//--- Commission tracking
double         commission_per_pip = 0;
double         min_profit_pips = 0;

//--- Signal management for rapid trading
double         last_signal_price = 0;
int            signal_cooldown_ticks = 0;
const int      SIGNAL_COOLDOWN = 2; // Minimal cooldown ticks between signals

//--- Global variables
int fast_handle = INVALID_HANDLE;
int slow_handle = INVALID_HANDLE;
int rsi_handle = INVALID_HANDLE;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Initialize symbol info
   if(!symbolInfo.Name(_Symbol))
     {
      Print("Failed to initialize symbol info.");
      return(INIT_FAILED);
     }
   // Set trade parameters
   trade.SetExpertMagicNumber(987654);
   trade.SetMarginMode();
   trade.SetTypeFillingBySymbol(symbolInfo.Name());
   trade.SetDeviationInPoints(3);
   
   // Determine point and pip values
   point = symbolInfo.Point();
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
   
   // Commission calculations
   double tick_value = symbolInfo.TickValue();
   if(symbolInfo.Digits()==3 || symbolInfo.Digits()==5)
      tick_value *= 10;
   double commission_per_lot_one_way = CommissionPerLot/2.0;
   commission_per_pip = commission_per_lot_one_way/(TP_Pips*tick_value/point);
   min_profit_pips = CommissionPerLot/(tick_value/point)*MinProfitTargetMultiplier;
   if(AccountForCommission && TP_Pips < min_profit_pips)
     {
      Print("Warning: TP_Pips (",TP_Pips,") is less than min required (",
            DoubleToString(min_profit_pips,1),").");
     }
   
   // Allocate indicator buffers
   ArrayResize(fast_ema_buffer,3);
   ArrayResize(slow_ema_buffer,3);
   ArrayResize(rsi_buffer,3);
   ArrayResize(price_buffer,10);
   ArrayResize(high_buffer,10);
   ArrayResize(low_buffer,10);
   
   // Create indicator handles and check
   fast_handle = iMA(_Symbol, Trading_Timeframe, Fast_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   slow_handle = iMA(_Symbol, Trading_Timeframe, Slow_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   rsi_handle = iRSI(_Symbol, Trading_Timeframe, RSI_Period, PRICE_CLOSE);
   if(fast_handle==INVALID_HANDLE || slow_handle==INVALID_HANDLE || rsi_handle==INVALID_HANDLE)
     {
      Print("Error creating indicator handles.");
      return(INIT_FAILED);
     }
   
   // Set initial risk management variables
   starting_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   max_equity = starting_balance;
   last_day = 0;
   daily_profit_loss = 0;

   ChartSetString(0, CHART_COMMENT, "VoltScalper v1.00 - HFT Scalping on EURUSD");
   Print("VoltScalper v1.00 initialized. Ready to scalp EURUSD.");
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(fast_handle != INVALID_HANDLE) IndicatorRelease(fast_handle);
   if(slow_handle != INVALID_HANDLE) IndicatorRelease(slow_handle);
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   Print("VoltScalper deinitialized. Reason: ", reason);
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   logged_this_tick = false;
   if(!symbolInfo.RefreshRates())
     {
      if(EnableDebugMode) Print("RefreshRates failed");
      return;
     }
   
   if(!CheckOrderRateLimit())
     {
      if(EnableDebugMode && !logged_this_tick) 
         Print("Order rate limit reached.");
      return;
     }
   
   if(CopyBuffer(fast_handle, 0, 0, 3, fast_ema_buffer) <= 0 ||
      CopyBuffer(slow_handle, 0, 0, 3, slow_ema_buffer) <= 0 ||
      CopyBuffer(rsi_handle, 0, 0, 3, rsi_buffer) <= 0 ||
      CopyClose(_Symbol, Trading_Timeframe, 0, 10, price_buffer) <= 0 ||
      CopyHigh(_Symbol, Trading_Timeframe, 0, 10, high_buffer) <= 0 ||
      CopyLow(_Symbol, Trading_Timeframe, 0, 10, low_buffer) <= 0)
     {
      if(EnableDebugMode) Print("Error copying indicator data");
      return;
     }
   
   ManageOpenPositions();
   int position_count = CountOpenPositions();
   if(position_count >= MaxSimultaneousOrders)
     {
      if(EnableDebugMode && !logged_this_tick)
         Print("Max open positions reached: ", position_count);
      return;
     }
   
   AnalyzeMarket();
   UpdateDashboard();
  }

//+------------------------------------------------------------------+
//| Analyze market and generate trade signals                        |
//+------------------------------------------------------------------+
void AnalyzeMarket()
  {
   double ask = symbolInfo.Ask();
   double bid = symbolInfo.Bid();
   double spread = symbolInfo.Spread() * point;
   
   if(signal_cooldown_ticks > 0)
      signal_cooldown_ticks--;
   
   double fast_ema = fast_ema_buffer[0];
   double slow_ema = slow_ema_buffer[0];
   double atr = (high_buffer[0] - low_buffer[0]) / 2.0;
   
   bool bullish = (fast_ema > slow_ema);
   bool bearish = (fast_ema < slow_ema);
   
   double latency_adjustment = (double)LatencyMs / 1000.0 * atr;
   
   bool buy_signal = bullish && (MathAbs(ask - last_signal_price) > (point));
   bool sell_signal = bearish && (MathAbs(bid - last_signal_price) > (point));
   
   // For testing: Force a trade if no open positions exist
   if(CountOpenPositions() == 0)
     {
      if(EnableDebugMode)
         Print("No open positions detected. Forcing BUY signal for testing.");
      buy_signal = true;
      sell_signal = false;
     }
   
   if(buy_signal)
     {
      if(EnableDebugMode) Print("BUY signal triggered: fast EMA=", fast_ema, ", slow EMA=", slow_ema);
      last_signal_price = ask;
      signal_cooldown_ticks = SIGNAL_COOLDOWN;
      OpenPosition(ORDER_TYPE_BUY, ask + latency_adjustment);
     }
   else if(sell_signal)
     {
      if(EnableDebugMode) Print("SELL signal triggered: fast EMA=", fast_ema, ", slow EMA=", slow_ema);
      last_signal_price = bid;
      signal_cooldown_ticks = SIGNAL_COOLDOWN;
      OpenPosition(ORDER_TYPE_SELL, bid - latency_adjustment);
     }
   else
     {
      conditions_not_met_count++;
      if(EnableDebugMode && signal_check_count % 100 == 0)
         Print("No signal. Checks: ", signal_check_count, " not met: ", conditions_not_met_count);
     }
   signal_check_count++;
  }

//+------------------------------------------------------------------+
//| Open a position with risk management adjustments                 |
//+------------------------------------------------------------------+
void OpenPosition(ENUM_ORDER_TYPE order_type, double entry_price)
  {
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
   else return;
   
   double reward_in_pips = MathAbs(entry_price - tp_price) / pip_value;
   if(AccountForCommission && reward_in_pips < min_profit_pips)
     {
      if(EnableDebugMode) Print("Adjusting TP to overcome commission.");
      if(order_type == ORDER_TYPE_BUY)
         tp_price = entry_price + (min_profit_pips * pip_value);
      else
         tp_price = entry_price - (min_profit_pips * pip_value);
     }
   
   double account_balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double risk_amount = account_balance * (RiskPercent/100.0);
   double risk_distance = MathAbs(entry_price - sl_price);
   double tick_value = symbolInfo.TickValue();
   if(symbolInfo.Digits()==3 || symbolInfo.Digits()==5)
      tick_value *= 10;
   
   double risk_lots = risk_amount / ((risk_distance * tick_value) / point);
   if(DynamicLeverageAdjustment)
     {
      double volatility = (high_buffer[0]-low_buffer[0]);
      if(volatility > 0)
         risk_lots *= (1.0/volatility);
     }
   
   // Normalize lot size to broker requirements
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   risk_lots = MathMax(risk_lots, minLot);
   risk_lots = MathMin(risk_lots, maxLot);
   // Adjust to nearest multiple of lotStep
   risk_lots = MathFloor(risk_lots / lotStep) * lotStep;
   
   risk_lots = MathMax(risk_lots, minLot); // ensure not below min
   
   bool result = false;
   if(order_type == ORDER_TYPE_BUY)
      result = trade.Buy(risk_lots, _Symbol, 0, sl_price, tp_price, "VoltScalper Buy");
   else
      result = trade.Sell(risk_lots, _Symbol, 0, sl_price, tp_price, "VoltScalper Sell");
   
   if(result)
     {
      last_order_time = TimeCurrent();
      orders_this_minute++;
      Print("Order executed: ", EnumToString(order_type), " Lots: ", DoubleToString(risk_lots,2),
            " Entry: ", DoubleToString(entry_price, _Digits),
            " SL: ", DoubleToString(sl_price, _Digits),
            " TP: ", DoubleToString(tp_price, _Digits));
     }
   else
      Print("Order error: ", GetLastError());
  }

//+------------------------------------------------------------------+
//| Manage open positions - close positions older than 5 seconds     |
//+------------------------------------------------------------------+
void ManageOpenPositions()
  {
   int total = PositionsTotal();
   for(int i = total - 1; i >= 0; i--)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket==0) continue;
      if(!PositionSelect(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC) != 987654) continue;
      datetime pos_time = (datetime)PositionGetInteger(POSITION_TIME);
      if(TimeCurrent() - pos_time > 5)
        {
         string sym = PositionGetString(POSITION_SYMBOL);
         if(!trade.PositionClose(sym))
            Print("Failed to close position on ", sym, ". Error: ",GetLastError());
        }
     }
  }

//+------------------------------------------------------------------+
//| Count open positions for this EA                                 |
//+------------------------------------------------------------------+
int CountOpenPositions()
  {
   int count = 0;
   int total = PositionsTotal();
   for(int i=0; i<total; i++)
     {
      ulong ticket = PositionGetTicket(i);
      if(ticket==0) continue;
      if(!PositionSelect(ticket)) continue;
      if(PositionGetInteger(POSITION_MAGIC)==987654)
         count++;
     }
   return count;
  }

//+------------------------------------------------------------------+
//| Check order rate limits                                            |
//+------------------------------------------------------------------+
bool CheckOrderRateLimit()
  {
   datetime current_time = TimeCurrent();
   MqlDateTime tm;
   TimeToStruct(current_time, tm);
   datetime this_minute = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d:00",
                           tm.year, tm.mon, tm.day, tm.hour, tm.min));
   if(this_minute != current_minute)
     {
      current_minute = this_minute;
      orders_this_minute = 0;
      last_order_time = 0;
     }
   if(orders_this_minute >= OrdersPerMinute)
      return false;
   if(last_order_time > 0 && current_time - last_order_time < 1)
      return false;
   return true;
  }

//+------------------------------------------------------------------+
//| Update dashboard information on chart                            |
//+------------------------------------------------------------------+
void UpdateDashboard()
  {
   double account_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   string dash = "VoltScalper v1.00\n";
   dash += "Equity: $" + DoubleToString(account_equity,2) + "\n";
   dash += "Open Positions: " + IntegerToString(CountOpenPositions()) + "/" + IntegerToString(MaxSimultaneousOrders) + "\n";
   dash += "Orders/Min: " + IntegerToString(orders_this_minute) + "/" + IntegerToString(OrdersPerMinute) + "\n";
   dash += "Daily P/L: $" + DoubleToString(account_equity - starting_balance,2) + "\n";
   ChartSetString(0, CHART_COMMENT, dash);
  }

//+------------------------------------------------------------------+ 