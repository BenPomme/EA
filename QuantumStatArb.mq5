//+------------------------------------------------------------------+
//|                                                  QuantumStatArb.mq5 |
//|                              Copyright 2024, Quantum Trading Systems|
//+------------------------------------------------------------------+
#property copyright "Copyright 2024, Quantum Trading Systems"
#property link      ""
#property version   "1.00"
#property strict
#property tester_indicator "Examples\\ATR.ex5"
#property tester_library "OpenCL.dll"

// Include required files
#include <Object.mqh>
#include <Arrays\Array.mqh>
#include <Arrays\ArrayObj.mqh>
#include <Trade\Trade.mqh>
#include <Arrays\ArrayDouble.mqh>

// OpenCL imports with proper conditional compilation
#ifdef __MQL5__
#import "OpenCL64.dll" // Change to OpenCL64.dll for 64-bit platform
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
#endif

// OpenCL Constants
#define CL_USE_GPU 1
#define CL_MEM_READ_WRITE 1

// Trading Parameters
input group "Trading Parameters"
input double RiskPercent = 1.0;          // Risk per trade (%)
input double MaxDrawdown = 30.0;         // Maximum drawdown (%)
input int    MaxPositions = 3;           // Maximum concurrent positions
input bool   UseGPUAcceleration = true;  // Use GPU acceleration

// Kalman Filter Parameters
input group "Kalman Filter Parameters"
input int    KalmanWindow = 100;         // Kalman filter window
input double ProcessVariance = 0.0001;   // Process variance
input double MeasurementVariance = 0.1;  // Measurement variance

// Market Microstructure Parameters
input group "Market Microstructure"
input int    OrderFlowDepth = 5;         // Order flow analysis depth
input int    MicroStructureWindow = 50;  // Market microstructure window
input double ImbalanceThreshold = 0.7;   // Order imbalance threshold

// Mean Reversion Parameters
input group "Mean Reversion"
input int    MeanReversionPeriod = 20;   // Mean reversion period
input double DeviationThreshold = 2.0;   // Standard deviation threshold
input int    HoldingPeriod = 4;         // Holding period in hours

// GPU Acceleration Parameters
input group "GPU Settings"
input int    GPUWorkGroupSize = 256;     // GPU work group size
input int    GPUBufferSize = 1024;      // GPU buffer size

// Fee Structure (FP Markets)
input group "Commission Settings"
input double CommissionPerLot = 5.50;    // Commission per lot in EUR
input string CommissionCurrency = "EUR"; // Commission currency

// Global Variables
CTrade Trade;
int gpuContext;
int gpuProgram;
int gpuKernel;
int gpuKalmanBuffer;
int gpuPriceBuffer;
int gpuStateBuffer;

// Market state arrays
double priceHistory[];
double kalmanStates[];
double microStructure[];
double regimeStates[];

//+------------------------------------------------------------------+
//| Custom indicator buffers                                         |
//+------------------------------------------------------------------+
double KalmanFilterBuffer[];
double MicroStructureBuffer[];
double RegimeBuffer[];

// Indicator handles
int atrHandle;
int ma20Handle;
int ma50Handle;

//+------------------------------------------------------------------+
//| Expert initialization function                                    |
//+------------------------------------------------------------------+
int OnInit()
{
   // Initialize arrays
   ArrayResize(priceHistory, GPUBufferSize);
   ArrayResize(kalmanStates, GPUBufferSize);
   ArrayResize(microStructure, GPUBufferSize);
   ArrayResize(regimeStates, GPUBufferSize);
   
   // Initialize indicator handles
   atrHandle = iATR(_Symbol, PERIOD_H1, 14);
   ma20Handle = iMA(_Symbol, PERIOD_H1, 20, 0, MODE_SMA, PRICE_CLOSE);
   ma50Handle = iMA(_Symbol, PERIOD_H1, 50, 0, MODE_SMA, PRICE_CLOSE);
   
   if(atrHandle == INVALID_HANDLE || ma20Handle == INVALID_HANDLE || ma50Handle == INVALID_HANDLE)
   {
      Print("Failed to create indicator handles");
      return INIT_FAILED;
   }
   
   // Initialize GPU only if not in testing mode
   if(UseGPUAcceleration && !MQLInfoInteger(MQL_TESTER))
   {
      if(!InitializeGPU())
      {
         Print("Failed to initialize GPU. Falling back to CPU calculations.");
      }
   }
   
   // Set up trading parameters
   Trade.SetExpertMagicNumber(123456);
   Trade.SetMarginMode();
   Trade.SetTypeFillingBySymbol(_Symbol);
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   // Release indicator handles
   IndicatorRelease(atrHandle);
   IndicatorRelease(ma20Handle);
   IndicatorRelease(ma50Handle);
   
   if(UseGPUAcceleration)
   {
      // Clean up GPU resources
      CLBufferFree(gpuKalmanBuffer);
      CLBufferFree(gpuPriceBuffer);
      CLBufferFree(gpuStateBuffer);
      CLKernelFree(gpuKernel);
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
   }
}

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
{
   // Update market state
   UpdateMarketState();
   
   // Check for maximum positions
   if(PositionsTotal() >= MaxPositions) return;
   
   // Check drawdown
   if(CheckDrawdown()) return;
   
   // Generate trading signals
   double signal = GenerateSignal();
   
   // Execute trades based on signal
   if(MathAbs(signal) >= ImbalanceThreshold)
   {
      ExecuteTrade(signal > 0);
   }
}

//+------------------------------------------------------------------+
//| Initialize GPU acceleration                                      |
//+------------------------------------------------------------------+
bool InitializeGPU()
{
   Print("Initializing GPU acceleration...");
   
   // Create OpenCL context
   gpuContext = CLContextCreate(CL_USE_GPU);
   if(gpuContext == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL context");
      return false;
   }
   
   // Create and build OpenCL program
   string programSource = LoadGPUProgram();
   gpuProgram = CLProgramCreate(gpuContext, programSource);
   if(gpuProgram == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL program");
      CLContextFree(gpuContext);
      return false;
   }
   
   // Create kernel
   gpuKernel = CLKernelCreate(gpuProgram, "UpdateKalmanFilter");
   if(gpuKernel == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL kernel");
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
      return false;
   }
   
   // Create buffers
   gpuKalmanBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuPriceBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuStateBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   
   if(gpuKalmanBuffer == INVALID_HANDLE || gpuPriceBuffer == INVALID_HANDLE || 
      gpuStateBuffer == INVALID_HANDLE)
   {
      Print("Failed to create OpenCL buffers");
      CLBufferFree(gpuKalmanBuffer);
      CLBufferFree(gpuPriceBuffer);
      CLBufferFree(gpuStateBuffer);
      CLKernelFree(gpuKernel);
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
      return false;
   }
   
   Print("GPU acceleration initialized successfully");
   return true;
}

//+------------------------------------------------------------------+
//| Load GPU program source                                          |
//+------------------------------------------------------------------+
string LoadGPUProgram()
{
   string program = 
   "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
   "\n"
   "__kernel void UpdateKalmanFilter(\n"
   "    __global float* kalmanStates,\n"
   "    __global float* priceHistory,\n"
   "    __global float* stateBuffer,\n"
   "    const float processVariance,\n"
   "    const float measurementVariance\n"
   ")\n"
   "{\n"
   "    int i = get_global_id(0);\n"
   "    int size = get_global_size(0);\n"
   "    \n"
   "    if(i >= size) return;\n"
   "    \n"
   "    // Kalman filter prediction step\n"
   "    float prediction = kalmanStates[i];\n"
   "    float predictionError = sqrt(stateBuffer[i] + processVariance);\n"
   "    \n"
   "    // Kalman filter update step\n"
   "    float measurement = priceHistory[i];\n"
   "    float kalmanGain = predictionError * predictionError / \n"
   "                       (predictionError * predictionError + measurementVariance);\n"
   "    \n"
   "    // Update state estimate and error covariance\n"
   "    kalmanStates[i] = prediction + kalmanGain * (measurement - prediction);\n"
   "    stateBuffer[i] = (1.0f - kalmanGain) * predictionError * predictionError;\n"
   "}\n";
   
   return program;
}

//+------------------------------------------------------------------+
//| Update market state using GPU acceleration                       |
//+------------------------------------------------------------------+
void UpdateMarketState()
{
   // Update price history
   for(int i=GPUBufferSize-1; i>0; i--)
   {
      priceHistory[i] = priceHistory[i-1];
   }
   priceHistory[0] = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   if(UseGPUAcceleration)
   {
      UpdateStateGPU();
   }
   else
   {
      UpdateStateCPU();
   }
}

//+------------------------------------------------------------------+
//| Update market state using GPU                                    |
//+------------------------------------------------------------------+
void UpdateStateGPU()
{
   // Write data to GPU buffers
   if(!CLBufferWrite(gpuKalmanBuffer, kalmanStates) ||
      !CLBufferWrite(gpuPriceBuffer, priceHistory) ||
      !CLBufferWrite(gpuStateBuffer, regimeStates))
   {
      Print("Failed to write data to GPU buffers");
      UpdateStateCPU();
      return;
   }
   
   // Set kernel arguments
   if(!CLKernelSetArgument(gpuKernel, 0, gpuKalmanBuffer) ||
      !CLKernelSetArgument(gpuKernel, 1, gpuPriceBuffer) ||
      !CLKernelSetArgument(gpuKernel, 2, gpuStateBuffer) ||
      !CLKernelSetArgument(gpuKernel, 3, (float)ProcessVariance) ||
      !CLKernelSetArgument(gpuKernel, 4, (float)MeasurementVariance))
   {
      Print("Failed to set kernel arguments");
      UpdateStateCPU();
      return;
   }
   
   // Execute kernel
   uint globalWorkSize[] = {(uint)GPUBufferSize};
   uint localWorkSize[] = {(uint)GPUWorkGroupSize};
   if(!CLExecute(gpuKernel, 1, globalWorkSize, localWorkSize))
   {
      Print("Failed to execute kernel");
      UpdateStateCPU();
      return;
   }
   
   // Read results back
   if(!CLBufferRead(gpuKalmanBuffer, kalmanStates) ||
      !CLBufferRead(gpuStateBuffer, regimeStates))
   {
      Print("Failed to read results from GPU");
      UpdateStateCPU();
      return;
   }
}

//+------------------------------------------------------------------+
//| Update market state using CPU                                    |
//+------------------------------------------------------------------+
void UpdateStateCPU()
{
   // Implement Kalman filter on CPU
   for(int i=0; i<GPUBufferSize; i++)
   {
      // Prediction step
      double prediction = kalmanStates[i];
      double predictionError = MathSqrt(regimeStates[i] + ProcessVariance);
      
      // Update step
      double measurement = priceHistory[i];
      double kalmanGain = predictionError * predictionError / 
                         (predictionError * predictionError + MeasurementVariance);
      
      // Update state estimate and error covariance
      kalmanStates[i] = prediction + kalmanGain * (measurement - prediction);
      regimeStates[i] = (1.0 - kalmanGain) * predictionError * predictionError;
   }
}

//+------------------------------------------------------------------+
//| Generate trading signal                                          |
//+------------------------------------------------------------------+
double GenerateSignal()
{
   // Combine multiple signal sources
   double kalmanSignal = CalculateKalmanSignal();
   double microStructureSignal = CalculateMicroStructureSignal();
   double regimeSignal = CalculateRegimeSignal();
   
   // Weight the signals based on market conditions
   double signal = 0.4 * kalmanSignal + 
                  0.3 * microStructureSignal + 
                  0.3 * regimeSignal;
                  
   return signal;
}

//+------------------------------------------------------------------+
//| Calculate signal based on Kalman filter                          |
//+------------------------------------------------------------------+
double CalculateKalmanSignal()
{
   double currentState = kalmanStates[0];
   double currentPrice = priceHistory[0];
   
   return (currentState - currentPrice) / Point();
}

//+------------------------------------------------------------------+
//| Calculate signal based on market microstructure                  |
//+------------------------------------------------------------------+
double CalculateMicroStructureSignal()
{
   double buyVolume = 0.0, sellVolume = 0.0;
   
   MqlTick tick[];
   if(CopyTicks(_Symbol, tick, COPY_TICKS_ALL, 0, OrderFlowDepth) > 0)
   {
      for(int i=0; i<OrderFlowDepth && i<ArraySize(tick); i++)
      {
         if((tick[i].flags & TICK_FLAG_BUY) != 0)
            buyVolume += (double)tick[i].volume;
         if((tick[i].flags & TICK_FLAG_SELL) != 0)
            sellVolume += (double)tick[i].volume;
      }
   }
   
   if(buyVolume + sellVolume < 0.000001) return 0.0;
   return (buyVolume - sellVolume) / (buyVolume + sellVolume);
}

//+------------------------------------------------------------------+
//| Calculate signal based on regime detection                       |
//+------------------------------------------------------------------+
double CalculateRegimeSignal()
{
   double volatility = CalculateVolatility();
   double trend = CalculateTrend();
   
   return trend * (1.0 + volatility);
}

//+------------------------------------------------------------------+
//| Execute trade based on signal                                    |
//+------------------------------------------------------------------+
void ExecuteTrade(bool isBuy)
{
   double lotSize = CalculateLotSize();
   double stopLoss = CalculateStopLoss(isBuy);
   double takeProfit = CalculateTakeProfit(isBuy);
   
   // Calculate trading costs
   double tradingCosts = CalculateTradingCosts(lotSize, isBuy);
   
   // Adjust position size based on costs
   lotSize = AdjustLotSizeForCosts(lotSize, tradingCosts);
   
   if(isBuy)
   {
      Trade.Buy(lotSize, _Symbol, 0, stopLoss, takeProfit, "QuantumStatArb");
   }
   else
   {
      Trade.Sell(lotSize, _Symbol, 0, stopLoss, takeProfit, "QuantumStatArb");
   }
}

//+------------------------------------------------------------------+
//| Calculate trading costs                                          |
//+------------------------------------------------------------------+
double CalculateTradingCosts(double lots, bool isBuy)
{
   double commission = CommissionPerLot * lots;
   
   // Convert commission to account currency if needed
   if(CommissionCurrency != AccountInfoString(ACCOUNT_CURRENCY))
   {
      double conversionRate = 1.0;
      if(CommissionCurrency == "EUR" && AccountInfoString(ACCOUNT_CURRENCY) == "USD")
         conversionRate = SymbolInfoDouble("EURUSD", SYMBOL_BID);
      commission *= conversionRate;
   }
   
   // Calculate spread cost
   double spread = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD) * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double spreadCost = spread * lots * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   
   return commission + spreadCost;
}

//+------------------------------------------------------------------+
//| Calculate position size based on risk                            |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * RiskPercent / 100.0;
   
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double stopLossPoints = MathAbs(CalculateStopLoss(true) - SymbolInfoDouble(_Symbol, SYMBOL_ASK));
   
   double lotSize = riskAmount / (stopLossPoints * tickValue);
   return NormalizeLotSize(lotSize);
}

//+------------------------------------------------------------------+
//| Normalize lot size to broker requirements                        |
//+------------------------------------------------------------------+
double NormalizeLotSize(double lots)
{
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double lotStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lots = MathRound(lots / lotStep) * lotStep;
   lots = MathMax(lots, minLot);
   lots = MathMin(lots, maxLot);
   
   return lots;
}

//+------------------------------------------------------------------+
//| Calculate stop loss level                                        |
//+------------------------------------------------------------------+
double CalculateStopLoss(bool isBuy)
{
   double atrBuffer[];
   ArraySetAsSeries(atrBuffer, true);
   
   if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0)
      return 0.0;
      
   double multiplier = 1.5;
   
   if(isBuy)
      return SymbolInfoDouble(_Symbol, SYMBOL_ASK) - atrBuffer[0] * multiplier;
   else
      return SymbolInfoDouble(_Symbol, SYMBOL_BID) + atrBuffer[0] * multiplier;
}

//+------------------------------------------------------------------+
//| Calculate take profit level                                      |
//+------------------------------------------------------------------+
double CalculateTakeProfit(bool isBuy)
{
   double atrBuffer[];
   ArraySetAsSeries(atrBuffer, true);
   
   if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0)
      return 0.0;
      
   double multiplier = 2.5;
   
   if(isBuy)
      return SymbolInfoDouble(_Symbol, SYMBOL_ASK) + atrBuffer[0] * multiplier;
   else
      return SymbolInfoDouble(_Symbol, SYMBOL_BID) - atrBuffer[0] * multiplier;
}

//+------------------------------------------------------------------+
//| Check if maximum drawdown is exceeded                            |
//+------------------------------------------------------------------+
bool CheckDrawdown()
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double drawdown = (balance - equity) / balance * 100;
   
   return drawdown > MaxDrawdown;
}

//+------------------------------------------------------------------+
//| Calculate current volatility                                     |
//+------------------------------------------------------------------+
double CalculateVolatility()
{
   double atrBuffer[];
   ArraySetAsSeries(atrBuffer, true);
   
   if(CopyBuffer(atrHandle, 0, 0, 1, atrBuffer) <= 0)
      return 0.0;
      
   return atrBuffer[0] / Point();
}

//+------------------------------------------------------------------+
//| Calculate current trend                                          |
//+------------------------------------------------------------------+
double CalculateTrend()
{
   double ma20Buffer[], ma50Buffer[];
   ArraySetAsSeries(ma20Buffer, true);
   ArraySetAsSeries(ma50Buffer, true);
   
   if(CopyBuffer(ma20Handle, 0, 0, 1, ma20Buffer) <= 0 ||
      CopyBuffer(ma50Handle, 0, 0, 1, ma50Buffer) <= 0)
      return 0.0;
   
   return (ma20Buffer[0] - ma50Buffer[0]) / Point();
}

//+------------------------------------------------------------------+
//| Adjust lot size based on trading costs                           |
//+------------------------------------------------------------------+
double AdjustLotSizeForCosts(double lots, double costs)
{
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double maxRisk = balance * RiskPercent / 100.0;
   
   // Reduce position size if costs exceed 10% of risk amount
   if(costs > maxRisk * 0.1)
   {
      lots *= (maxRisk * 0.1) / costs;
   }
   
   return NormalizeLotSize(lots);
}

//+------------------------------------------------------------------+
//| Tester function                                                    |
//+------------------------------------------------------------------+
double OnTester()
{
   // Basic statistics that are guaranteed to be available
   double initial_deposit = TesterStatistics(STAT_INITIAL_DEPOSIT);
   double profit = TesterStatistics(STAT_PROFIT);
   double gross_profit = TesterStatistics(STAT_GROSS_PROFIT);
   double gross_loss = TesterStatistics(STAT_GROSS_LOSS);
   double max_drawdown = TesterStatistics(STAT_BALANCE_DD);
   double trades = TesterStatistics(STAT_TRADES);
   
   // Early exit for insufficient trades
   if(trades < 10) return 0.0;
   
   // Calculate metrics
   double profit_factor = (gross_loss != 0.0) ? MathAbs(gross_profit / gross_loss) : 0.0;
   double recovery_factor = (max_drawdown != 0.0) ? profit / max_drawdown : 0.0;
   double roi = (initial_deposit != 0.0) ? (profit / initial_deposit) * 100.0 : 0.0;
   
   // Calculate fitness score
   double fitness = 0.0;
   
   // Base metrics (80% weight)
   if(profit_factor > 0.0) fitness += 0.3 * MathMin(profit_factor / 1.5, 1.0);
   if(recovery_factor > 0.0) fitness += 0.2 * MathMin(recovery_factor, 1.0);
   if(roi > 0.0) fitness += 0.3 * MathMin(roi / 20.0, 1.0);
   
   // Trade frequency (20% weight)
   fitness += 0.2 * MathMin(trades / 100.0, 1.0);
   
   // Apply penalties
   if(max_drawdown > MaxDrawdown) fitness *= 0.5;
   if(trades < 30) fitness *= 0.7;
   if(profit_factor < 1.2) fitness *= 0.8;
   
   return fitness;
} 