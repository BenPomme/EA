//+------------------------------------------------------------------+
//|  Quantum_Trend_Follower.mq5                                     |
//|  Minimal version WITHOUT using <Trade\Trade.mqh>                |
//+------------------------------------------------------------------+
#property copyright ""
#property version   "1.00"
#property strict
// --- Minimal trade definitions inserted for compilation without <Trade\Trade.mqh>
#define TRADE_ACTION_DEAL 1
#define ORDER_TYPE_BUY 0
#define ORDER_TYPE_SELL 1
#define TRADE_RETCODE_DONE 10009

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

// --- GPU Acceleration Settings
input bool   UseGPUAcceleration = true;    // Enable GPU acceleration for quantum calculations
input int    GPUWorkGroupSize   = 256;     // GPU work group size
input int    GPUBufferSize      = 1024;    // GPU buffer size for calculations
input int    MaxScalpIterations = 100;     // Maximum iterations for quantum calculations
input double TimeStep           = 0.01;    // Time step for quantum calculations

// --- Quantum Physics Constants
#define PLANCK_CONSTANT 6.62607015e-34
#define BOLTZMANN_CONSTANT 1.380649e-23
#define SPEED_OF_LIGHT 299792458.0

// --- Quantum State Parameters
input int    QuantumStates     = 10;    // Number of quantum states to consider
input double EnergyThreshold   = 0.7;   // Energy threshold for state transitions
input double WaveFunctionDecay = 0.95;  // Decay factor for wave function
input double EntanglementFactor = 0.3;  // Factor for quantum entanglement effects

// --- Market Quantum Parameters
input int    PriceHistorySize  = 100;   // Size of price history for quantum analysis
input double VolatilityFactor  = 0.5;   // Factor for volatility in quantum calculations
input double MomentumFactor    = 0.3;   // Factor for momentum in quantum calculations

// --- External Inputs
input double Lots            = 0.10;   // Default lot size
input double RiskPerTrade    = 1.0;    // % risk per trade
input bool   UseStopLoss     = true;   // Enable stop-loss
input int    SL_Pips         = 100;    // Default Stop-Loss (pips)
input int    TP_Pips         = 300;    // Default Take-Profit (pips)
input double QuantumThreshold = 0.70;  // Probability threshold (0..1)

//--- Global variables
string gSymbol;         // current symbol
double gPoint;          // symbol point size
bool   gTradeInProgress = false; // to track if a trade is open

// Quantum state arrays
double waveFunction[];    // Wave function for price prediction
double energyLevels[];    // Energy levels for different states
double momentumStates[];  // Momentum states
double priceHistory[];    // Price history for quantum analysis

// GPU-related variables
int gpuContext;        // OpenCL context
int gpuProgram;        // OpenCL program
int gpuKernel;         // OpenCL kernel
int gpuWaveBuffer;     // Buffer for wave function
int gpuEnergyBuffer;   // Buffer for energy levels
int gpuMomentumBuffer; // Buffer for momentum states
int gpuPriceBuffer;    // Buffer for price history

//+------------------------------------------------------------------+
//| GPU Acceleration Functions                                        |
//+------------------------------------------------------------------+
#ifndef USE_GPU_ACCELERATION_LIB
// Fallback dummy implementations if GPUOpenCL.dylib is not available
int GPU_SolveSchrodingerEquation(const double &gpuHamiltonian[], double &gpuPsiWave[], int lookback, int iterations, double timeStep)
{
   // Return error code (-1) to indicate GPU function is not available.
   return -1;
}

void GPU_InitializeOpenCL()
{
   // Dummy implementation
   Print("GPU_InitializeOpenCL skipped: GPU module not available.");
}

void GPU_ReleaseOpenCL()
{
   // Dummy implementation
   Print("GPU_ReleaseOpenCL skipped: GPU module not available.");
}
#else
#import "GPUOpenCL.dylib"
   int   GPU_SolveSchrodingerEquation(const double &gpuHamiltonian[], double &gpuPsiWave[], int lookback, int iterations, double timeStep);
   void  GPU_InitializeOpenCL();
   void  GPU_ReleaseOpenCL();
#import
#endif

//+------------------------------------------------------------------+
//| Initialize trade request structure                                |
//+------------------------------------------------------------------+
void InitializeTradeRequest(MqlTradeRequest &request)
{
   request.action = TRADE_ACTION_DEAL;
   request.symbol = "";
   request.magic = 0;
   request.type = 0;
   request.volume = 0;
   request.price = 0;
   request.sl = 0;
   request.tp = 0;
}

//+------------------------------------------------------------------+
//| Initialize trade result structure                                 |
//+------------------------------------------------------------------+
void InitializeTradeResult(MqlTradeResult &result)
{
   result.retcode = 0;
   result.order = 0;
}

//+------------------------------------------------------------------+
//| Initialize quantum state arrays                                   |
//+------------------------------------------------------------------+
void InitializeQuantumStates()
{
   Print("Initializing quantum states...");
   Print("Quantum States: ", QuantumStates);
   Print("Price History Size: ", PriceHistorySize);
   
   ArrayResize(waveFunction, QuantumStates);
   ArrayResize(energyLevels, QuantumStates);
   ArrayResize(momentumStates, QuantumStates);
   ArrayResize(priceHistory, PriceHistorySize);
   
   // Initialize wave function as a Gaussian distribution
   float sigma = (float)QuantumStates / 6.0f;
   float norm_factor = 0;
   for(int i=0; i<QuantumStates; i++)
   {
      waveFunction[i] = (float)MathExp(-0.5f * MathPow(i - QuantumStates/2, 2) / (sigma*sigma));
      norm_factor += MathPow(waveFunction[i], 2);
   }
   norm_factor = (float)MathSqrt(norm_factor);
   for(int i=0; i<QuantumStates; i++)
      waveFunction[i] /= norm_factor;
      
   // Initialize energy levels
   for(int i=0; i<QuantumStates; i++)
   {
      energyLevels[i] = (float)((i + 0.5) * PLANCK_CONSTANT);
      momentumStates[i] = 0;
   }
   
   Print("Initial wave function sum: ", ArraySum(waveFunction));
   Print("Initial energy levels sum: ", ArraySum(energyLevels));
   Print("Initial momentum states sum: ", ArraySum(momentumStates));
}

//+------------------------------------------------------------------+
//| Initialize GPU acceleration                                       |
//+------------------------------------------------------------------+
bool InitializeGPU()
{
   bool gpuEnabled = UseGPUAcceleration;
   if(!gpuEnabled) return true;
   
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
      Print("Failed to create OpenCL program. Program source: ", programSource);
      CLContextFree(gpuContext);
      return false;
   }
   Print("OpenCL program created successfully");
   
   // Create kernel
   Print("Creating OpenCL kernel...");
   gpuKernel = CLKernelCreate(gpuProgram, "UpdateQuantumStates");
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
   gpuWaveBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuEnergyBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuMomentumBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   gpuPriceBuffer = CLBufferCreate(gpuContext, GPUBufferSize, CL_MEM_READ_WRITE);
   
   if(gpuWaveBuffer == INVALID_HANDLE || gpuEnergyBuffer == INVALID_HANDLE ||
      gpuMomentumBuffer == INVALID_HANDLE || gpuPriceBuffer == INVALID_HANDLE)
   {
      Print("Failed to create one or more OpenCL buffers");
      CLBufferFree(gpuWaveBuffer);
      CLBufferFree(gpuEnergyBuffer);
      CLBufferFree(gpuMomentumBuffer);
      CLBufferFree(gpuPriceBuffer);
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
   "__kernel void UpdateQuantumStates(\n" +
   "   __global float* waveFunction,\n" +
   "   __global float* energyLevels,\n" +
   "   __global float* momentumStates,\n" +
   "   __global float* priceHistory,\n" +
   "   const float volatility,\n" +
   "   const float momentum,\n" +
   "   const float entanglementFactor\n" +
   ")\n" +
   "{\n" +
   "   int i = get_global_id(0);\n" +
   "   int size = get_global_size(0);\n" +
   "   \n" +
   "   if(i >= size) return;\n" +
   "   \n" +
   "   // Update energy levels based on market conditions\n" +
   "   energyLevels[i] *= (1.0f + volatility);\n" +
   "   momentumStates[i] = momentum;\n" +
   "   \n" +
   "   // Apply quantum entanglement effects\n" +
   "   for(int j = 0; j < size; j++)\n" +
   "   {\n" +
   "      if(i != j)\n" +
   "      {\n" +
   "         float entanglement = exp(-fabs((float)(i-j)) * entanglementFactor);\n" +
   "         energyLevels[i] *= (1.0f + entanglement * energyLevels[j] / energyLevels[i]);\n" +
   "      }\n" +
   "   }\n" +
   "   \n" +
   "   // Update wave function\n" +
   "   waveFunction[i] *= exp(-energyLevels[i]);\n" +
   "}\n";
   
   Print("OpenCL program source loaded, length: ", StringLen(program));
   return program;
}

//+------------------------------------------------------------------+
//| Update quantum states using GPU acceleration                     |
//+------------------------------------------------------------------+
void UpdateQuantumStates()
{
   bool gpuEnabled = UseGPUAcceleration;
   if(gpuEnabled)
   {
      // Update price history
      for(int i=PriceHistorySize-1; i>0; i--)
         priceHistory[i] = (float)priceHistory[i-1];
      priceHistory[0] = (float)SymbolInfoDouble(gSymbol, SYMBOL_BID);
      
      // Calculate market conditions
      float volatility = (float)CalculateVolatility();
      float momentum = (float)CalculateMomentum();
      
      // Copy data to GPU buffers
      if(!CLBufferWrite(gpuWaveBuffer, waveFunction))
      {
         Print("Failed to write wave function to GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLBufferWrite(gpuEnergyBuffer, energyLevels))
      {
         Print("Failed to write energy levels to GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLBufferWrite(gpuMomentumBuffer, momentumStates))
      {
         Print("Failed to write momentum states to GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLBufferWrite(gpuPriceBuffer, priceHistory))
      {
         Print("Failed to write price history to GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
      
      // Set kernel arguments
      if(!CLKernelSetArgument(gpuKernel, 0, gpuWaveBuffer))
      {
         Print("Failed to set wave function argument");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLKernelSetArgument(gpuKernel, 1, gpuEnergyBuffer))
      {
         Print("Failed to set energy levels argument");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLKernelSetArgument(gpuKernel, 2, gpuMomentumBuffer))
      {
         Print("Failed to set momentum states argument");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLKernelSetArgument(gpuKernel, 3, gpuPriceBuffer))
      {
         Print("Failed to set price history argument");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLKernelSetArgument(gpuKernel, 4, volatility))
      {
         Print("Failed to set volatility argument");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLKernelSetArgument(gpuKernel, 5, momentum))
      {
         Print("Failed to set momentum argument");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLKernelSetArgument(gpuKernel, 6, (float)EntanglementFactor))
      {
         Print("Failed to set entanglement factor argument");
         UpdateQuantumStatesCPU();
         return;
      }
      
      // Execute kernel
      uint globalWorkSize[] = {QuantumStates};
      uint localWorkSize[] = {GPUWorkGroupSize};
      if(!CLExecute(gpuKernel, 1, globalWorkSize, localWorkSize))
      {
         Print("Failed to execute OpenCL kernel");
         UpdateQuantumStatesCPU();
         return;
      }
      
      // Read results back
      if(!CLBufferRead(gpuWaveBuffer, waveFunction))
      {
         Print("Failed to read wave function from GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLBufferRead(gpuEnergyBuffer, energyLevels))
      {
         Print("Failed to read energy levels from GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
      if(!CLBufferRead(gpuMomentumBuffer, momentumStates))
      {
         Print("Failed to read momentum states from GPU buffer");
         UpdateQuantumStatesCPU();
         return;
      }
   }
   else
   {
      UpdateQuantumStatesCPU();
   }
}

//+------------------------------------------------------------------+
//| CPU implementation of quantum state updates                      |
//+------------------------------------------------------------------+
void UpdateQuantumStatesCPU()
{
   // Update price history
   for(int i=PriceHistorySize-1; i>0; i--)
      priceHistory[i] = priceHistory[i-1];
   priceHistory[0] = SymbolInfoDouble(gSymbol, SYMBOL_BID);
   
   // Calculate volatility and momentum
   double volatility = CalculateVolatility();
   double momentum = CalculateMomentum();
   
   // Update energy levels based on market conditions
   for(int i=0; i<QuantumStates; i++)
   {
      energyLevels[i] *= (1.0 + volatility * VolatilityFactor);
      momentumStates[i] = momentum * MomentumFactor;
   }
   
   // Apply quantum entanglement effects
   ApplyQuantumEntanglement();
   
   // Update wave function
   UpdateWaveFunction();
}

//+------------------------------------------------------------------+
//| Calculate market volatility                                      |
//+------------------------------------------------------------------+
double CalculateVolatility()
{
   double sum = 0;
   for(int i=0; i<PriceHistorySize-1; i++)
   {
      double diff = priceHistory[i] - priceHistory[i+1];
      sum += diff * diff;
   }
   return MathSqrt(sum / (PriceHistorySize-1)) / gPoint;
}

//+------------------------------------------------------------------+
//| Calculate market momentum                                        |
//+------------------------------------------------------------------+
double CalculateMomentum()
{
   if(PriceHistorySize < 2) return 0;
   return (priceHistory[0] - priceHistory[1]) / gPoint;
}

//+------------------------------------------------------------------+
//| Apply quantum entanglement effects                               |
//+------------------------------------------------------------------+
void ApplyQuantumEntanglement()
{
   for(int i=0; i<QuantumStates; i++)
   {
      for(int j=i+1; j<QuantumStates; j++)
      {
         double entanglement = MathExp(-MathAbs(i-j) * EntanglementFactor);
         energyLevels[i] *= (1.0 + entanglement * energyLevels[j] / energyLevels[i]);
         energyLevels[j] *= (1.0 + entanglement * energyLevels[i] / energyLevels[j]);
      }
   }
}

//+------------------------------------------------------------------+
//| Update wave function based on energy levels                      |
//+------------------------------------------------------------------+
void UpdateWaveFunction()
{
   double sum = 0;
   for(int i=0; i<QuantumStates; i++)
   {
      waveFunction[i] *= MathExp(-energyLevels[i] * WaveFunctionDecay);
      sum += MathPow(waveFunction[i], 2);
   }
   
   // Normalize wave function
   sum = MathSqrt(sum);
   if(sum > 0)
   {
      for(int i=0; i<QuantumStates; i++)
         waveFunction[i] /= sum;
   }
}

//+------------------------------------------------------------------+
//| Calculate quantum forecast probability                           |
//+------------------------------------------------------------------+
double CalculateQuantumForecast(bool bullish)
{
   UpdateQuantumStates();
   
   // Calculate probability based on wave function and energy levels
   double probability = 0;
   for(int i=0; i<QuantumStates; i++)
   {
      if(bullish)
         probability += waveFunction[i] * (i > QuantumStates/2 ? 1 : 0);
      else
         probability += waveFunction[i] * (i < QuantumStates/2 ? 1 : 0);
   }
   
   Print("Quantum Forecast for ", (bullish ? "Buy" : "Sell"), " signal:");
   Print("Probability: ", probability);
   Print("Wave Function Sum: ", ArraySum(waveFunction));
   Print("Energy Levels Sum: ", ArraySum(energyLevels));
   
   return probability;
}

//+------------------------------------------------------------------+
//| Helper function to sum array elements                             |
//+------------------------------------------------------------------+
double ArraySum(const double &array[])
{
   double sum = 0;
   for(int i=0; i<ArraySize(array); i++)
      sum += array[i];
   return sum;
}

//+------------------------------------------------------------------+
//| OnInit                                                          |
//+------------------------------------------------------------------+
int OnInit()
{
   gSymbol = _Symbol;
   gPoint  = SymbolInfoDouble(gSymbol, SYMBOL_POINT);
   
   // Initialize quantum states
   InitializeQuantumStates();
   
   // Initialize GPU acceleration
   bool gpuEnabled = UseGPUAcceleration;
   if(!InitializeGPU())
   {
      Print("Failed to initialize GPU acceleration. Falling back to CPU calculations.");
      gpuEnabled = false;
   }

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| OnTick                                                          |
//+------------------------------------------------------------------+
void OnTick()
{
   // 1) Check if we already have an open position
   if(CheckExistingPosition())
   {
      gTradeInProgress = true;
      return;
   }
   else gTradeInProgress = false;

   // 2) Check breakout signals
   bool breakoutBuySignal  = CheckBreakoutSignal(true);
   bool breakoutSellSignal = CheckBreakoutSignal(false);

   // 3) If a signal is confirmed, run quantum forecast
   if(breakoutBuySignal)
   {
      double probOfContinuation = CalculateQuantumForecast(true);
      Print("Buy Signal - Quantum Probability: ", probOfContinuation);
      if(probOfContinuation >= QuantumThreshold)
      {
         Print("Placing Buy Order - Probability: ", probOfContinuation);
         PlaceOrder(true);
      }
   }
   else if(breakoutSellSignal)
   {
      double probOfContinuation = CalculateQuantumForecast(false);
      Print("Sell Signal - Quantum Probability: ", probOfContinuation);
      if(probOfContinuation >= QuantumThreshold)
      {
         Print("Placing Sell Order - Probability: ", probOfContinuation);
         PlaceOrder(false);
      }
   }
}

//+------------------------------------------------------------------+
//| Check breakout signal (simple Donchian)                         |
//+------------------------------------------------------------------+
bool CheckBreakoutSignal(bool bullish)
{
   int    period     = 20;
   double donchianHigh = iHigh(gSymbol, PERIOD_H1, 0);
   double donchianLow  = iLow(gSymbol, PERIOD_H1, 0);

   Print("Checking ", (bullish ? "Buy" : "Sell"), " breakout signal...");
   Print("Initial Donchian High: ", donchianHigh, " Low: ", donchianLow);

   for(int i=1; i<period; i++)
   {
      double tmpHigh = iHigh(gSymbol, PERIOD_H1, i);
      double tmpLow  = iLow(gSymbol, PERIOD_H1, i);
      if(tmpHigh > donchianHigh) donchianHigh = tmpHigh;
      if(tmpLow  < donchianLow ) donchianLow  = tmpLow;
   }

   double lastPrice = iClose(gSymbol, PERIOD_H1, 0);
   double breakoutLevel = (bullish) ? donchianHigh + 2*gPoint : donchianLow - 2*gPoint;

   Print("Final Donchian High: ", donchianHigh, " Low: ", donchianLow);
   Print("Last Price: ", lastPrice, " Breakout Level: ", breakoutLevel);

   if(bullish)
   {
      if(lastPrice > breakoutLevel)
      {
         Print("Buy Breakout Signal - Price: ", lastPrice, " Breakout Level: ", breakoutLevel);
         return(true);
      }
   }
   else
   {
      if(lastPrice < breakoutLevel)
      {
         Print("Sell Breakout Signal - Price: ", lastPrice, " Breakout Level: ", breakoutLevel);
         return(true);
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Place an order using OrderSend (no standard lib)                |
//+------------------------------------------------------------------+
void PlaceOrder(bool isBuy)
{
   MqlTradeRequest  request;
   MqlTradeResult   result;
   InitializeTradeRequest(request);
   InitializeTradeResult(result);

   //--- fill request
   request.action   = TRADE_ACTION_DEAL;
   request.symbol   = gSymbol;
   request.magic    = 10001; // arbitrary magic number
   request.type     = (isBuy ? ORDER_TYPE_BUY : ORDER_TYPE_SELL);
   request.volume   = CalculateLotSize();

   // price & SL/TP
   double price = (isBuy)
                  ? NormalizeDouble(SymbolInfoDouble(gSymbol, SYMBOL_ASK), _Digits)
                  : NormalizeDouble(SymbolInfoDouble(gSymbol, SYMBOL_BID), _Digits);

   request.price = price;

   // StopLoss & TakeProfit
   if(UseStopLoss)
   {
      double sl = (isBuy) ? price - SL_Pips*gPoint : price + SL_Pips*gPoint;
      request.sl = NormalizeDouble(sl, _Digits);
   }
   if(TP_Pips > 0)
   {
      double tp = (isBuy) ? price + TP_Pips*gPoint : price - TP_Pips*gPoint;
      request.tp = NormalizeDouble(tp, _Digits);
   }

   Print("Placing ", (isBuy ? "Buy" : "Sell"), " Order - Price: ", price, " SL: ", request.sl, " TP: ", request.tp);

   //--- send order
   if(!OrderSend(request, result) || result.retcode != TRADE_RETCODE_DONE)
   {
      Print("OrderSend failed: retcode=", result.retcode);
   }
   else
   {
      Print("Order placed successfully: ticket=", result.order);
   }
}

//+------------------------------------------------------------------+
//| CalculateLotSize                                                |
//+------------------------------------------------------------------+
double CalculateLotSize()
{
   // If user doesn't want risk-based sizing, just use fixed.
   if(RiskPerTrade <= 0.0) return(Lots);

   double balance   = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskValue = balance * (RiskPerTrade/100.0);
   double tickValue = SymbolInfoDouble(gSymbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize  = SymbolInfoDouble(gSymbol, SYMBOL_TRADE_TICK_SIZE);

   // costPerLot = money lost if SL is hit on 1 lot
   double costPerLot = (SL_Pips*gPoint / tickSize) * tickValue;
   if(costPerLot <= 0) return(Lots);

   double lotsNeeded = riskValue / costPerLot;
   return NormalizeLotSize(lotsNeeded);
}

//+------------------------------------------------------------------+
//| NormalizeLotSize                                                |
//+------------------------------------------------------------------+
double NormalizeLotSize(double lots)
{
   double lotStep = SymbolInfoDouble(gSymbol, SYMBOL_VOLUME_STEP);
   double lotMin  = SymbolInfoDouble(gSymbol, SYMBOL_VOLUME_MIN);
   double lotMax  = SymbolInfoDouble(gSymbol, SYMBOL_VOLUME_MAX);

   // Round down to nearest step
   lots = MathFloor(lots / lotStep) * lotStep;
   if(lots < lotMin) lots = lotMin;
   if(lots > lotMax) lots = lotMax;
   return lots;
}

//+------------------------------------------------------------------+
//| CheckExistingPosition                                           |
//+------------------------------------------------------------------+
bool CheckExistingPosition()
{
   // We'll do a simple check for open positions in this symbol
   for(int i=PositionsTotal()-1; i>=0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket > 0)
      {
         string posSymbol = PositionGetString(POSITION_SYMBOL);
         if(posSymbol == gSymbol)
            return(true);
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| FakeQuantumForecast (stub) -> returns random probability        |
//+------------------------------------------------------------------+
double FakeQuantumForecast(bool bullish)
{
   // Instead of referencing standard library or GPU code,
   // we just return a random number for demonstration
   return (double)MathRand() / 32767.0;
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   bool gpuEnabled = UseGPUAcceleration;
   if(gpuEnabled)
   {
      // Clean up GPU resources
      CLBufferFree(gpuWaveBuffer);
      CLBufferFree(gpuEnergyBuffer);
      CLBufferFree(gpuMomentumBuffer);
      CLBufferFree(gpuPriceBuffer);
      CLKernelFree(gpuKernel);
      CLProgramFree(gpuProgram);
      CLContextFree(gpuContext);
   }
}