/+------------------------------------------------------------------+
//|  Quantum_MeanReversion_Arb.mq5                                  |
//|  Conceptual EA for Mean Reversion & Simple Pair Arbitrage       |
//+------------------------------------------------------------------+
#property copyright "Copyright 2025"
#property version   "1.00"
#property strict

// Include required files in correct order
#include <Object.mqh>
#include <Trade\Trade.mqh>

// Create trade object
CTrade Trade;

// --- OpenCL Function Imports
#import "OpenCL.dll"
   int CLContextCreate(int device_type);
   int CLProgramCreate(int context, string program_source);
   int CLKernelCreate(int program, string kernel_name);
   int CLBufferCreate(int context, int size, int flags);
   bool CLBufferWrite(int buffer, float &data[]);
   bool CLBufferRead(int buffer, float &data[]);
   bool CLKernelSetArgument(int kernel, int arg_index, int value);
   bool CLKernelSetArgument(int kernel, int arg_index, float value);
   bool CLExecute(int kernel, int dimensions, uint &global_work_size[], uint &local_work_size[]);
   void CLBufferFree(int buffer);
   void CLKernelFree(int kernel);
   void CLProgramFree(int program);
   void CLContextFree(int context);
#import

// --- OpenCL Constants
#define CL_USE_GPU 1
#define CL_MEM_READ_WRITE 1

//--- External inputs
input double Lots            = 0.10;   // Default fixed lot size
input double RiskPerTrade    = 1.0;    // % risk for single-instrument trades
input bool   UseStopLoss     = true;
input int    SL_Pips         = 75;     // For single-instrument trades
input int    TP_Pips         = 150;    // For single-instrument trades (2:1 ratio)
input bool   UseGPUAcceleration   = true;    // Enable GPU acceleration
input int    GPUWorkGroupSize   = 256;     // GPU work group size
input int    GPUBufferSize      = 1024;    // GPU buffer size
input int    MaxScalpIterations = 100;     // Max iterations
input double TimeStep           = 0.01;    // Time step
input double QuantumReversionProb = 0.55; // Probability threshold for reversion (lowered for more trades)
input int    MagicNumber     = 20002; // For trade identification

//--- Fee and Cost Parameters
input double CommissionPerLot = 5.50;    // Commission per lot in EUR (FP Markets standard)
input double MaxSpreadPips    = 2.0;     // Maximum allowed spread in pips
input double SlippagePips     = 1.0;     // Expected slippage in pips
input bool   UseSwap          = true;    // Consider swap rates in calculations
input string CommissionCurrency = "EUR";  // Commission currency (FP Markets standard)

//--- Arbitrage pairs (for demonstration)
input string ArbSymbol1      = "EURUSD";
input string ArbSymbol2      = "GBPUSD";
input double ArbSpreadThreshold = 0.0010; // Minimal difference for opening an arb trade (increased for more opportunities)

//--- Global variables
string gSymbol;       // main symbol for single-instrument reversion
bool   gGPUSupported; // GPU usage flag

// GPU-related variables
int gpuContext;        // OpenCL context
int gpuProgram;        // OpenCL program
int gpuKernel;         // OpenCL kernel
int gpuPriceBuffer;    // Buffer for price history
int gpuRSIBuffer;      // Buffer for RSI values
int gpuSpreadBuffer;   // Buffer for spread values

//+------------------------------------------------------------------+
//| Initialize GPU acceleration                                       |
//+------------------------------------------------------------------+
bool InitializeGPU()
{
   if(!UseGPUAcceleration) return false;
   
   Print("Initializing GPU acceleration...");
   
   // Create OpenCL context
   gpuContext = CLContextCreate(CL_USE_GPU);
   if(gpuContext == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL context");
      return false;
   }
   Print("OpenCL context created successfully");
   
   // Create and build OpenCL program
   string programSource = LoadGPUProgram();
   Print("Loading OpenCL program...");
   gpuProgram = CLProgramCreate(gpuContext, programSource);
   if(gpuProgram == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL program");
      CLContextFree(gpuContext);
      return false;
   }
   Print("OpenCL program created successfully");
   
   // Create kernel
   Print("Creating OpenCL kernel...");
   gpuKernel = CLKernelCreate(gpuProgram, "CalculateQuantumReversion");
   if(gpuKernel == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL kernel");
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
      return false;
   }
   Print("OpenCL kernel created successfully");
   
   // Create buffers
   Print("Creating OpenCL buffers...");
   gpuPriceBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuRSIBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuSpreadBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   
   if(gpuPriceBuffer == INVALID_HANDLE || gpuRSIBuffer == INVALID_HANDLE || gpuSpreadBuffer == INVALID_HANDLE)
   {
      Print("Failed to create one or more OpenCL buffers");
      CLBufferFree(gpuPriceBuffer);
      CLBufferFree(gpuRSIBuffer);
      CLBufferFree(gpuSpreadBuffer);
      CLKernelFree(gpuKernel);
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
      return false;
   }
   Print("OpenCL buffers created successfully");
   
   Print("GPU acceleration initialized successfully");
   return true;
}

//+------------------------------------------------------------------+
//| Load GPU program source code                                      |
//+------------------------------------------------------------------+
string LoadGPUProgram()
{
   string program = 
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n" +
   "\n" +
   "__kernel void CalculateQuantumReversion(\n" +
   "   __global float* priceHistory,\n" +
   "   __global float* rsiValues,\n" +
   "   __global float* spreadValues,\n" +
   "   const float volatility,\n" +
   "   const float meanPrice,\n" +
   "   const float meanRSI,\n" +
   "   const float meanSpread\n" +
   ")\n" +
   "{\n" +
   "   int i = get_global_id(0);\n" +
   "   int size = get_global_size(0);\n" +
   "   \n" +
   "   if(i >= size) return;\n" +
   "   \n" +
   "   // Calculate quantum reversion probability\n" +
   "   float priceDeviation = fabs(priceHistory[i] - meanPrice) / volatility;\n" +
   "   float rsiDeviation = fabs(rsiValues[i] - meanRSI) / 30.0f;\n" +
   "   float spreadDeviation = fabs(spreadValues[i] - meanSpread) / volatility;\n" +
   "   \n" +
   "   // Combine deviations into a single probability\n" +
   "   float probability = exp(-(priceDeviation + rsiDeviation + spreadDeviation) / 3.0f);\n" +
   "   \n" +
   "   // Store result back in priceHistory buffer\n" +
   "   priceHistory[i] = probability;\n" +
   "}\n";
   
   Print("OpenCL program source loaded, length: ", StringLen(program));
   return program;
}

//+------------------------------------------------------------------+
//| GPU implementation of quantum reversion calculation              |
//+------------------------------------------------------------------+
double GPU_QuantumReversion(string sym, bool isBuy)
{
   if(!gGPUSupported) return CPU_QuantumReversion(sym, isBuy);
   
   // Prepare data for GPU
   float priceHistory[];
   float rsiValues[];
   float spreadValues[];
   ArrayResize(priceHistory, GPUBufferSize);
   ArrayResize(rsiValues, GPUBufferSize);
   ArrayResize(spreadValues, GPUBufferSize);
   
   // Fill price history
   for(int i=0; i<GPUBufferSize; i++)
   {
      priceHistory[i] = (float)iClose(sym, PERIOD_H1, i);
      rsiValues[i] = (float)iRSI(sym, PERIOD_H1, 14, i);
      spreadValues[i] = (float)(iClose(ArbSymbol1, PERIOD_H1, i) - iClose(ArbSymbol2, PERIOD_H1, i));
   }
   
   // Calculate means and volatility
   float meanPrice = 0, meanRSI = 0, meanSpread = 0;
   float volatility = 0;
   for(int i=0; i<GPUBufferSize; i++)
   {
      meanPrice += priceHistory[i];
      meanRSI += rsiValues[i];
      meanSpread += spreadValues[i];
   }
   meanPrice /= (float)GPUBufferSize;
   meanRSI /= (float)GPUBufferSize;
   meanSpread /= (float)GPUBufferSize;
   
   for(int i=0; i<GPUBufferSize; i++)
   {
      volatility += MathPow(priceHistory[i] - meanPrice, 2);
   }
   volatility = (float)MathSqrt(volatility / GPUBufferSize);
   
   // Copy data to GPU
   if(!CLBufferWrite(gpuPriceBuffer, priceHistory))
   {
      Print("Failed to write price history to GPU");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLBufferWrite(gpuRSIBuffer, rsiValues))
   {
      Print("Failed to write RSI values to GPU");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLBufferWrite(gpuSpreadBuffer, spreadValues))
   {
      Print("Failed to write spread values to GPU");
      return CPU_QuantumReversion(sym, isBuy);
   }
   
   // Set kernel arguments
   if(!CLKernelSetArgument(gpuKernel, 0, gpuPriceBuffer))
   {
      Print("Failed to set price buffer argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLKernelSetArgument(gpuKernel, 1, gpuRSIBuffer))
   {
      Print("Failed to set RSI buffer argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLKernelSetArgument(gpuKernel, 2, gpuSpreadBuffer))
   {
      Print("Failed to set spread buffer argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLKernelSetArgument(gpuKernel, 3, volatility))
   {
      Print("Failed to set volatility argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLKernelSetArgument(gpuKernel, 4, meanPrice))
   {
      Print("Failed to set mean price argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLKernelSetArgument(gpuKernel, 5, meanRSI))
   {
      Print("Failed to set mean RSI argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   if(!CLKernelSetArgument(gpuKernel, 6, meanSpread))
   {
      Print("Failed to set mean spread argument");
      return CPU_QuantumReversion(sym, isBuy);
   }
   
   // Execute kernel
   uint globalWorkSize[] = {GPUBufferSize};
   uint localWorkSize[] = {GPUWorkGroupSize};
   if(!CLExecute(gpuKernel, 1, globalWorkSize, localWorkSize))
   {
      Print("Failed to execute OpenCL kernel");
      return CPU_QuantumReversion(sym, isBuy);
   }
   
   // Read results back
   if(!CLBufferRead(gpuPriceBuffer, priceHistory))
   {
      Print("Failed to read results from GPU");
      return CPU_QuantumReversion(sym, isBuy);
   }
   
   // Calculate final probability
   float totalProb = 0;
   for(int i=0; i<GPUBufferSize; i++)
   {
      totalProb += priceHistory[i];
   }
   return totalProb / GPUBufferSize;
}

//+------------------------------------------------------------------+
//| CPU implementation of quantum reversion calculation              |
//+------------------------------------------------------------------+
double CPU_QuantumReversion(string sym, bool isBuy)
{
   // Simple CPU implementation for fallback
   double rsiVal = iRSI(sym, PERIOD_H1, 14, 0);
   double meanRSI = 50.0;
   double deviation = MathAbs(rsiVal - meanRSI) / 30.0;
   return MathExp(-deviation);
}

//+------------------------------------------------------------------+
//| GPU implementation of quantum spread calculation                 |
//+------------------------------------------------------------------+
double GPU_QuantumSpread(string sym1, string sym2, double spread)
{
   if(!gGPUSupported) return CPU_QuantumSpread(sym1, sym2, spread);
   
   // Use the same GPU calculation as reversion but with spread data
   return GPU_QuantumReversion(sym1, spread > 0);
}

//+------------------------------------------------------------------+
//| CPU implementation of quantum spread calculation                 |
//+------------------------------------------------------------------+
double CPU_QuantumSpread(string sym1, string sym2, double spread)
{
   // Simple CPU implementation for fallback
   double meanSpread = 0.0;
   double deviation = MathAbs(spread - meanSpread) / ArbSpreadThreshold;
   return MathExp(-deviation);
}

//+------------------------------------------------------------------+
//| OnInit                                                          |
//+------------------------------------------------------------------+
int OnInit()
{
   gSymbol = _Symbol;
   gGPUSupported = InitializeGPU();
   
   if(!gGPUSupported && UseGPUAcceleration)
   {
      Print("GPU acceleration failed to initialize. Falling back to CPU mode.");
   }
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnDeinit                                                        |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   if(gGPUSupported)
   {
      // Clean up GPU resources
      CLBufferFree(gpuPriceBuffer);
      CLBufferFree(gpuRSIBuffer);
      CLBufferFree(gpuSpreadBuffer);
      CLKernelFree(gpuKernel);
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
   }
}

//+------------------------------------------------------------------+
//| OnTick                                                          |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1) Mean Reversion on main symbol
   if(!CheckExistingPosition(gSymbol))
   {
      bool entryBuy=false, entrySell=false;
      if(CheckMeanReversionSignal(gSymbol, entryBuy, entrySell))
      {
         if(entryBuy)
         {
            double revertProb = (gGPUSupported)
                                ? GPU_QuantumReversion(gSymbol,true)
                                : CPU_QuantumReversion(gSymbol,true);

            if(revertProb >= QuantumReversionProb)
               PlaceSingleInstrumentTrade(gSymbol, true);
         }
         if(entrySell)
         {
            double revertProb = (gGPUSupported)
                                ? GPU_QuantumReversion(gSymbol,false)
                                : CPU_QuantumReversion(gSymbol,false);

            if(revertProb >= QuantumReversionProb)
               PlaceSingleInstrumentTrade(gSymbol, false);
         }
      }
   }

   // 2) Spread Arbitrage Check (ArbSymbol1 & ArbSymbol2)
   if(!CheckExistingPosition(ArbSymbol1) && !CheckExistingPosition(ArbSymbol2))
   {
      double bid1 = SymbolInfoDouble(ArbSymbol1, SYMBOL_BID);
      double bid2 = SymbolInfoDouble(ArbSymbol2, SYMBOL_BID);
      if(bid1 <= 0.0 || bid2 <= 0.0) return; // invalid quotes

      double spread = bid1 - bid2;
      // if spread > threshold => short overvalued, long undervalued
      // if spread < -threshold => the opposite
      if(spread > ArbSpreadThreshold)
      {
         double revertProb = (gGPUSupported)
                             ? GPU_QuantumSpread(ArbSymbol1,ArbSymbol2,spread)
                             : CPU_QuantumSpread(ArbSymbol1,ArbSymbol2,spread);

         if(revertProb >= QuantumReversionProb)
         {
            // SELL the overpriced (ArbSymbol1) & BUY the underpriced (ArbSymbol2)
            PlaceArbitrageTrade(ArbSymbol1,false); // SELL
            PlaceArbitrageTrade(ArbSymbol2,true);  // BUY
         }
      }
      else if(spread < -ArbSpreadThreshold)
      {
         double revertProb = (gGPUSupported)
                             ? GPU_QuantumSpread(ArbSymbol1,ArbSymbol2,spread)
                             : CPU_QuantumSpread(ArbSymbol1,ArbSymbol2,spread);

         if(revertProb >= QuantumReversionProb)
         {
            // BUY the underpriced (ArbSymbol1) & SELL the overpriced (ArbSymbol2)
            PlaceArbitrageTrade(ArbSymbol1,true);   // BUY
            PlaceArbitrageTrade(ArbSymbol2,false);  // SELL
         }
      }
   }
}

//+------------------------------------------------------------------+
//| Check if we already have position in given symbol               |
//+------------------------------------------------------------------+
bool CheckExistingPosition(string sym)
{
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == sym)
            return(true);
      }
   }
   return(false);
}

//+------------------------------------------------------------------+
//| CheckMeanReversionSignal                                        |
//| Example: RSI-based threshold. If RSI<35 => buy, RSI>65 => sell. |
//+------------------------------------------------------------------+
bool CheckMeanReversionSignal(string sym, bool &outBuy, bool &outSell)
{
   outBuy  = false;
   outSell = false;

   double rsiVal = iRSI(sym, PERIOD_H1, 14, 0);
   if(rsiVal <= 35.0)  // Less extreme buy condition
   {
      outBuy = true;
      return true;
   }
   else if(rsiVal >= 65.0)  // Less extreme sell condition
   {
      outSell = true;
      return true;
   }
   return false;
}

//+------------------------------------------------------------------+
//| Calculate total trading costs for a position                     |
//+------------------------------------------------------------------+
double CalculateTradingCosts(string sym, double lots, bool isBuy)
{
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   
   // Calculate spread cost
   double spread = SymbolInfoInteger(sym, SYMBOL_SPREAD) * point;
   double spreadCost = spread * lots * tickValue / tickSize;
   
   // Calculate commission in account currency
   double commission = CommissionPerLot * lots;
   if(CommissionCurrency != AccountInfoString(ACCOUNT_CURRENCY))
   {
      // Convert commission from EUR to account currency if needed
      double eurRate = SymbolInfoDouble("EURUSD", SYMBOL_BID);
      if(AccountInfoString(ACCOUNT_CURRENCY) == "USD")
         commission *= eurRate;
      else if(AccountInfoString(ACCOUNT_CURRENCY) == "GBP")
         commission *= eurRate / SymbolInfoDouble("GBPUSD", SYMBOL_BID);
   }
   
   // Calculate slippage cost
   double slippageCost = SlippagePips * point * lots * tickValue / tickSize;
   
   // Calculate swap if applicable
   double swapCost = 0;
   if(UseSwap)
   {
      double swapRate = isBuy ? 
         SymbolInfoDouble(sym, SYMBOL_SWAP_LONG) : 
         SymbolInfoDouble(sym, SYMBOL_SWAP_SHORT);
      swapCost = swapRate * lots;
   }
   
   double totalCosts = spreadCost + commission + slippageCost + swapCost;
   Print("Trading costs breakdown for ", sym, ":");
   Print("Spread cost: ", spreadCost, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Commission: ", commission, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Slippage: ", slippageCost, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Swap: ", swapCost, " ", AccountInfoString(ACCOUNT_CURRENCY));
   Print("Total costs: ", totalCosts, " ", AccountInfoString(ACCOUNT_CURRENCY));
   
   return totalCosts;
}

//+------------------------------------------------------------------+
//| Check if current spread is acceptable                            |
//+------------------------------------------------------------------+
bool IsSpreadAcceptable(string sym)
{
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);
   double currentSpread = SymbolInfoInteger(sym, SYMBOL_SPREAD) * point;
   double maxSpread = MaxSpreadPips * point;
   
   return currentSpread <= maxSpread;
}

//+------------------------------------------------------------------+
//| PlaceSingleInstrumentTrade                                      |
//+------------------------------------------------------------------+
void PlaceSingleInstrumentTrade(string sym, bool isBuy)
{
   // Check spread first
   if(!IsSpreadAcceptable(sym))
   {
      Print("Spread too high for ", sym, ": ", SymbolInfoInteger(sym, SYMBOL_SPREAD), " pips");
      return;
   }
   
   double price = (isBuy)
                  ? SymbolInfoDouble(sym, SYMBOL_ASK)
                  : SymbolInfoDouble(sym, SYMBOL_BID);
   if(price <= 0.0) return;

   double lotSize = CalculateLotSize(sym);
   
   // Calculate trading costs
   double tradingCosts = CalculateTradingCosts(sym, lotSize, isBuy);
   Print("Trading costs for ", sym, ": ", tradingCosts, " ", AccountInfoString(ACCOUNT_CURRENCY));

   double sl=0, tp=0;
   double point = SymbolInfoDouble(sym, SYMBOL_POINT);

   if(UseStopLoss)
   {
      if(isBuy) sl = price - SL_Pips*point;
      else      sl = price + SL_Pips*point;
   }
   if(TP_Pips > 0)
   {
      if(isBuy) tp = price + TP_Pips*point;
      else      tp = price - TP_Pips*point;
   }

   Trade.SetExpertMagicNumber(MagicNumber);
   if(isBuy)
      Trade.Buy(lotSize, sym, price, sl, tp, "QuantumMeanRevBuy");
   else
      Trade.Sell(lotSize, sym, price, sl, tp, "QuantumMeanRevSell");
}

//+------------------------------------------------------------------+
//| PlaceArbitrageTrade (market-neutral leg)                        |
//+------------------------------------------------------------------+
void PlaceArbitrageTrade(string sym, bool isBuy)
{
   // Check spread first
   if(!IsSpreadAcceptable(sym))
   {
      Print("Spread too high for ", sym, ": ", SymbolInfoInteger(sym, SYMBOL_SPREAD), " pips");
      return;
   }
   
   double price = (isBuy)
                  ? SymbolInfoDouble(sym, SYMBOL_ASK)
                  : SymbolInfoDouble(sym, SYMBOL_BID);
   if(price <= 0.0) return;

   // For arbitrage, we'll just use a fixed lot for both legs
   double lotSize = Lots;
   
   // Calculate trading costs
   double tradingCosts = CalculateTradingCosts(sym, lotSize, isBuy);
   Print("Trading costs for ", sym, ": ", tradingCosts, " ", AccountInfoString(ACCOUNT_CURRENCY));

   Trade.SetExpertMagicNumber(MagicNumber);
   if(isBuy)
      Trade.Buy(lotSize, sym, price, 0, 0, "QuantumArbBuy");
   else
      Trade.Sell(lotSize, sym, price, 0, 0, "QuantumArbSell");
}

//+------------------------------------------------------------------+
//| CalculateLotSize for single-instrument trades                   |
//+------------------------------------------------------------------+
double CalculateLotSize(string sym)
{
   if(RiskPerTrade <= 0.0) return(Lots);

   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskValue = balance * (RiskPerTrade/100.0);
   double tickValue = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(sym, SYMBOL_TRADE_TICK_SIZE);
   double point     = SymbolInfoDouble(sym, SYMBOL_POINT);

   // Include trading costs in risk calculation
   double estimatedCosts = CalculateTradingCosts(sym, Lots, true); // Use default lot for estimation
   riskValue -= estimatedCosts;

   double costPerLot = (SL_Pips*point / tickSize) * tickValue;
   if(costPerLot <= 0.0) return(Lots);

   double lotsNeeded = riskValue / costPerLot;
   return NormalizeLotSize(sym, lotsNeeded);
}

//+------------------------------------------------------------------+
//| NormalizeLotSize                                                |
//+------------------------------------------------------------------+
double NormalizeLotSize(string sym, double lots)
{
   double lotStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
   double lotMin  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
   double lotMax  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MAX);

   lots = MathFloor(lots/lotStep)*lotStep;
   if(lots < lotMin) lots = lotMin;
   if(lots > lotMax) lots = lotMax;
   return(lots);
}
//+------------------------------------------------------------------+