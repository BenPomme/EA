//+------------------------------------------------------------------+
//|    Swing Intraday Trading EA (Corrected for MQL5)                |
//+------------------------------------------------------------------+
#property strict
#property copyright "ChatGPT"
#property version   "1.0"

#include <Trade/Trade.mqh>
CTrade trade;

//--- Input parameters
input int    FastMAPeriod = 50;      // Fast MA period
input int    SlowMAPeriod = 100;     // Slow MA period
input double FixedLot     = 0.1;      // Fixed lot size
input double StopLossPips = 100;      // Stop-loss in pips (optional, not used in trade orders here)

//--- Global indicator handles
int handleFast = INVALID_HANDLE;
int handleSlow = INVALID_HANDLE;

//--- Global variable for new bar detection
datetime lastBarTime = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   // Create indicator handles for fast and slow moving averages
   handleFast = iMA(_Symbol, _Period, FastMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   handleSlow = iMA(_Symbol, _Period, SlowMAPeriod, 0, MODE_SMA, PRICE_CLOSE);
   
   if(handleFast == INVALID_HANDLE || handleSlow == INVALID_HANDLE)
   {
      Print("Failed to create one or both MA handles");
      return(INIT_FAILED);
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(handleFast != INVALID_HANDLE)
      IndicatorRelease(handleFast);
   if(handleSlow != INVALID_HANDLE)
      IndicatorRelease(handleSlow);
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Only evaluate on a new bar
   datetime currentTime = iTime(_Symbol, _Period, 0);
   if(currentTime == lastBarTime)
      return;
   lastBarTime = currentTime;
   
   // Arrays to hold MA values; we need two values: current (index 0) and previous (index 1)
   double fastMAArray[2], slowMAArray[2];
   
   if(CopyBuffer(handleFast, 0, 0, 2, fastMAArray) <= 0)
   {
      Print("Failed to copy fast MA buffer");
      return;
   }
   if(CopyBuffer(handleSlow, 0, 0, 2, slowMAArray) <= 0)
   {
      Print("Failed to copy slow MA buffer");
      return;
   }
   
   // Extract current and previous values
   double fastMA_current = fastMAArray[0];
   double fastMA_prev    = fastMAArray[1];
   double slowMA_current = slowMAArray[0];
   double slowMA_prev    = slowMAArray[1];
   
   double lot = FixedLot; // Using fixed lot size
   
   // If no open position exists, then check for new entry signals:
   if(!PositionSelect(_Symbol))
   {
      // Bullish crossover: previous fast <= previous slow and current fast > current slow
      if(fastMA_prev <= slowMA_prev && fastMA_current > slowMA_current)
      {
         trade.Buy(lot, _Symbol);
      }
      // Bearish crossover: previous fast >= previous slow and current fast < current slow
      else if(fastMA_prev >= slowMA_prev && fastMA_current < slowMA_current)
      {
         trade.Sell(lot, _Symbol);
      }
   }
   else
   {
      // If a position exists, check for a reversal signal:
      int posType = PositionGetInteger(POSITION_TYPE);
      
      // For a long position, if a bearish crossover occurs, close and open short
      if(posType == POSITION_TYPE_BUY && (fastMA_prev >= slowMA_prev && fastMA_current < slowMA_current))
      {
         trade.PositionClose(_Symbol);
         trade.Sell(lot, _Symbol);
      }
      // For a short position, if a bullish crossover occurs, close and open long
      else if(posType == POSITION_TYPE_SELL && (fastMA_prev <= slowMA_prev && fastMA_current > slowMA_current))
      {
         trade.PositionClose(_Symbol);
         trade.Buy(lot, _Symbol);
      }
   }
}