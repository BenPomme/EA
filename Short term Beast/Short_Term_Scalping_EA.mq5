
//--- Short-Term Scalping EA (MQL5)
#include <Trade\Trade.mqh>
CTrade trade;

input double FixedLot=0.1;
input int LookbackBars=10;
input double StopLossPips=10;
input double TakeProfitPips=5;

void OnTick()
{
   if(PositionSelect(_Symbol)) return;

   double ask=SymbolInfoDouble(_Symbol,SYMBOL_ASK);
   double bid=SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
   double pip=(SymbolInfoInteger(_Symbol,SYMBOL_DIGITS)==3 || SymbolInfoInteger(_Symbol,SYMBOL_DIGITS)==5)?point*10:point;

   double High=iHigh(_Symbol,_Period,iHighest(_Symbol,_Period,MODE_HIGH,LookbackBars,1));
   double Low=iLow(_Symbol,_Period,iLowest(_Symbol,_Period,MODE_LOW,LookbackBars,LookbackBars));

   if(!PositionSelect(_Symbol)){
      if(Ask>High){
         trade.Buy(FixedLot,NULL,Ask,(Ask-StopLossPips*pip),(Ask+pip*5));
      }
      if(Bid<Low){
         trade.Sell(FixedLot,NULL,(Bid+pip*StopLossPips),(Bid-pip*5));
      }
   }
}
