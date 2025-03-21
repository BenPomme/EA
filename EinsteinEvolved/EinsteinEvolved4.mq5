//+------------------------------------------------------------------+
//|                                                   EinsteinEvolved4.mq5               |
//|                Einstein Evolved: GPU Accelerated Quantum & ML EA   |
//|                                                                  |
//|   This version offloads heavy Schrödinger equation calculations to  |
//|   the GPU using OpenCL (via an external library) to improve speed.  |
//+------------------------------------------------------------------+

#property copyright "Copyright 2023, Benjamin Pommeraud"
#property link      "https://www.quantumtradinglab.com"
#property version   "4.00"
#property strict
#property description "EinsteinEvolved4: GPU Accelerated Quantum EA"

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>

//--- Input parameters - Performance Optimization
input int    CalculationFrequency     = 5;               // Frequency of quantum calculations (in ticks)
input bool   UseCachedCalculations    = true;            // Use cached calculations to reduce CPU load
input bool   UseSimplifiedQuantum     = false;           // Use simplified quantum calculations (faster but less accurate)
input int    MaxIterations            = 5;               // Maximum iterations for Schrödinger solution (lower = faster)

//--- New Input: GPU Acceleration
input bool   UseGPUAcceleration       = true;            // Offload heavy calculations to GPU via OpenCL

//--- Import external GPU acceleration functions
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

//--- Input parameters - Quantum Constants (Modified for more active trading)
input double PlankConstant            = 6.62607015e-24;  // Planck constant for scaling
input double QuantumDecayRate         = 0.65;            // Quantum decay factor
input int    UncertaintyPeriod        = 12;              // Uncertainty window period
input double EntanglementThreshold    = 0.2;             // Entanglement threshold
input double WaveFunctionCollapse     = 1.5;             // Wave function collapse multiplier

//--- Input parameters - Trading Execution (Adjusted for $5000 capital with 1:500 leverage)
input double RiskPercent              = 0.8;             // Risk percent per trade
input double QuantumSL                = 20.0;            // Stop Loss in pips
input double QuantumTP                = 40.0;            // Take Profit in pips
input int    MaxOpenPositions         = 2;               // Maximum open positions

//--- Input parameters - Machine Learning and Quantum Module
input double InpInitialLotSize        = 0.2;             // Initial lot size for trades
input int    InpSignalPeriod          = 10;              // Base signal period for indicator
input int    InpQuantumStates         = 12;              // Number of quantum states
input double InpQuantumDecay          = 0.90;            // Decay factor
input bool   EnableDebugMode          = false;           // Enable detailed debug logging

//--- Input parameters - Indicator Settings
input int    ATRPeriod                = 12;              // Period for ATR indicator
input int    RSI_Period               = 10;              // RSI period
input int    MAMethodPeriod           = 40;              // Period for Moving Average

//--- Input parameters - Advanced Schrödinger Settings
input int    SchrodingerLookback      = 40;              // Lookback period (reduced for better performance)
input int    PotentialWellDepth       = 30;              // Potential well depth factor
input double QuantumPotentialScaling  = 0.005;           // Scaling factor
input double WaveFunctionDiffusion    = 0.2;             // Diffusion coefficient

//--- Input parameters - Commission and Fees
input double CommissionPerLot         = 5.5;             // Commission per lot in base currency
input double SwapLong                 = -2.5;            // Swap points for long positions
input double SwapShort                = -2.5;            // Swap points for short positions
input double SpreadAverage            = 1.0;             // Average spread in pips
input string SymbolGroup              = "Forex Majors Raw"; // Symbol group for commission calculation

//--- Global Variables
CTrade trade;
double point;
double pip_value;
int pip_digits;

double OptimizedSignalPeriod;         // Adaptive signal period optimized via ML
double lastTradeProfit = 0.0;         // Profit from the last trade
int tradeCounter = 0;                 // Counter for executed trades
int calculationCounter = 0;           // Counter for limiting calculations

//--- Quantum Module Variables
int NUM_Q_STATES;
double QuantumState[];

//--- Advanced Schrödinger Equation Variables
double psi_wave[];                    // Wave function array
double potential_array[];             // Potential energy array
double hamiltonian[];                 // Hamiltonian operator matrix (flattened)
double market_momentum[];             // Market momentum buffer

//--- Caching Variables for Performance Optimization
double cached_prediction = 0;         // Cached Schrödinger prediction
double cached_threshold = 0;          // Cached quantum indicator threshold
datetime last_calculation_time = 0;   // Time of last full calculation
bool calculation_valid = false;       // Flag indicating if cache is valid

//--- Indicator Handles and Buffers
int atr_handle;
int rsi_handle;
int ma_handle;
double atr_buffer[];
double rsi_buffer[];
double ma_buffer[];

//--- Price history buffers
double close_buffer[];
double high_buffer[];
double low_buffer[];

// Account variables for capital management
double initialCapital = 5000.0;
double maxLeverage = 500.0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Set up trade parameters
   trade.SetExpertMagicNumber(98767);  // New magic number for version 4
   trade.SetDeviationInPoints(5);
   
   // Get market info for pip calculation
   point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   if(digits == 5 || digits == 3)
     {
      pip_value = point * 10;
      pip_digits = 1;
     }
   else
     {
      pip_value = point;
      pip_digits = 0;
     }
      
   // Initialize machine learning parameter
   OptimizedSignalPeriod = InpSignalPeriod;
   
   // Create indicator handles
   atr_handle = iATR(_Symbol, PERIOD_CURRENT, ATRPeriod);
   rsi_handle = iRSI(_Symbol, PERIOD_CURRENT, RSI_Period, PRICE_CLOSE);
   ma_handle  = iMA(_Symbol, PERIOD_CURRENT, MAMethodPeriod, 0, MODE_SMA, PRICE_CLOSE);
   if(atr_handle == INVALID_HANDLE || rsi_handle == INVALID_HANDLE || ma_handle == INVALID_HANDLE)
     {
      Print("Error creating indicator handles.");
      return(INIT_FAILED);
     }
     
   // Pre-allocate memory for arrays to avoid runtime resizing
   ArrayResize(atr_buffer, 3, 10);
   ArrayResize(rsi_buffer, 3, 10);
   ArrayResize(ma_buffer, 3, 10);
   ArrayResize(close_buffer, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(high_buffer, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(low_buffer, SchrodingerLookback, SchrodingerLookback+10);
   
   // Initialize Quantum Module
   InitializeQuantumModule();
   
   // Initialize Schrödinger equation components
   InitializeSchrodingerEquation();
   
   // Initialize GPU if enabled
   if(UseGPUAcceleration)
      GPU_InitializeOpenCL();
   
   Print("EinsteinEvolved4 initialized with GPU acceleration = ", UseGPUAcceleration);
   Print("Calculation frequency: Every ", CalculationFrequency, " ticks, Max iterations: ", MaxIterations);
   
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   if(ma_handle  != INVALID_HANDLE) IndicatorRelease(ma_handle);
   
   // Release GPU resources if used
   if(UseGPUAcceleration)
      GPU_ReleaseOpenCL();
   
   Print("EinsteinEvolved4 deinitialized.");
  }

//+------------------------------------------------------------------+
//| Initialize Quantum Module                                        |
//+------------------------------------------------------------------+
void InitializeQuantumModule()
  {
   NUM_Q_STATES = InpQuantumStates;
   ArrayResize(QuantumState, NUM_Q_STATES, NUM_Q_STATES+5);
   double equalProb = 1.0 / NUM_Q_STATES;
   for(int i=0; i<NUM_Q_STATES; i++)
      QuantumState[i] = equalProb;
   if(EnableDebugMode)
      Print("Quantum Module initialized with ", NUM_Q_STATES, " states.");
  }

//+------------------------------------------------------------------+
//| Initialize Schrödinger Equation Components                       |
//+------------------------------------------------------------------+
void InitializeSchrodingerEquation()
  {
   ArrayResize(psi_wave, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(potential_array, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(market_momentum, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(hamiltonian, SchrodingerLookback * SchrodingerLookback, SchrodingerLookback * SchrodingerLookback+20);
   ArrayInitialize(psi_wave, 0);
   ArrayInitialize(potential_array, 0);
   ArrayInitialize(hamiltonian, 0);
   ArrayInitialize(market_momentum, 0);
   int center = SchrodingerLookback / 2;
   double sigma = SchrodingerLookback / 6.0;
   double norm_factor = 0.0;
   for(int i=0; i<SchrodingerLookback; i++)
     {
      psi_wave[i] = MathExp(-0.5 * MathPow((i - center) / sigma, 2));
      norm_factor += MathPow(psi_wave[i], 2);
     }
   norm_factor = MathSqrt(norm_factor);
   if(norm_factor > 0)
     {
      for(int i=0; i<SchrodingerLookback; i++)
         psi_wave[i] /= norm_factor;
     }
   if(EnableDebugMode)
      Print("Schrödinger components initialized with optimized memory allocation");
  }

//+------------------------------------------------------------------+
//| Evolve Quantum State based on market volatility                  |
//+------------------------------------------------------------------+
void EvolveQuantumState(double marketFactor)
  {
   double sigmoid = 1.0 / (1.0 + MathExp(-marketFactor * 2.0));
   double sum = 0.0;
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      QuantumState[i] *= (0.5 + sigmoid/(i+1)) * InpQuantumDecay;
      sum += QuantumState[i];
     }
   if(sum > 0)
     {
      double invSum = 1.0 / sum;
      for(int i=0; i<NUM_Q_STATES; i++)
         QuantumState[i] *= invSum;
     }
  }

//+------------------------------------------------------------------+
//| Collapse Quantum State to determine adjustment                   |
//+------------------------------------------------------------------+
int CollapseQuantumState()
  {
   double rnd = (double)MathRand()/32767.0;
   double accum = 0.0;
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      accum += QuantumState[i];
      if(rnd < accum)
         return i;
     }
   return(NUM_Q_STATES-1);
  }

//+------------------------------------------------------------------+
//| Quantum Signal Adjustment                                          |
//+------------------------------------------------------------------+
int QuantumSignalAdjustment()
  {
   int stateIndex = CollapseQuantumState();
   return (stateIndex - (NUM_Q_STATES / 2)) * 2;
  }

//+------------------------------------------------------------------+
//| Update Quantum Potential based on price movements                |
//+------------------------------------------------------------------+
void UpdateQuantumPotential()
  {
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, close_buffer) <= 0 ||
      CopyHigh(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, high_buffer) <= 0 ||
      CopyLow(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, low_buffer) <= 0)
     {
      Print("Error copying price data");
      return;
     }
   for(int i=0; i<SchrodingerLookback-1; i++)
     {
      double weight = 1.0 + 0.1*(SchrodingerLookback-i-1)/SchrodingerLookback;
      market_momentum[i] = (close_buffer[i] - close_buffer[i+1]) * weight;
     }
   market_momentum[SchrodingerLookback-1] = 0;
   for(int i=0; i<SchrodingerLookback; i++)
     {
      double range = high_buffer[i] - low_buffer[i];
      double wellDepth = (range / point) * QuantumPotentialScaling;
      double momentumEffect = MathPow(MathAbs(market_momentum[i]), 1.5) * QuantumPotentialScaling;
      potential_array[i] = wellDepth * PotentialWellDepth - momentumEffect;
     }
  }

//+------------------------------------------------------------------+
//| Construct Hamiltonian Operator - optimized version               |
//+------------------------------------------------------------------+
void ConstructHamiltonian()
  {
   double hbar_squared_over_2m = MathPow(PlankConstant, 2) / (2.0 * WaveFunctionDiffusion);
   for(int i=0; i<SchrodingerLookback; i++)
     {
      hamiltonian[i * SchrodingerLookback + i] = potential_array[i] + (2.0 * hbar_squared_over_2m);
     }
   for(int i=0; i<SchrodingerLookback-1; i++)
     {
      hamiltonian[i * SchrodingerLookback + (i+1)] = -hbar_squared_over_2m;
      hamiltonian[(i+1) * SchrodingerLookback + i] = -hbar_squared_over_2m;
     }
  }

//+------------------------------------------------------------------+
//| Optimized Schrödinger Equation Solver                            |
//+------------------------------------------------------------------+
void SolveSchrodingerEquation()
  {
   if(UseGPUAcceleration)
     {
      if(!calculation_valid || !UseCachedCalculations)
        {
         UpdateQuantumPotential();
         ConstructHamiltonian();
         int iterations = UseSimplifiedQuantum ? MathMin(3, MaxIterations) : MaxIterations;
         int res = GPU_SolveSchrodingerEquation(hamiltonian, psi_wave, SchrodingerLookback, iterations, 0.02);
         if(res != 0)
            Print("GPU_SolveSchrodingerEquation error: ", res);
         calculation_valid = true;
         last_calculation_time = TimeCurrent();
        }
     }
   else
     {
      if(!calculation_valid || !UseCachedCalculations)
        {
         UpdateQuantumPotential();
         ConstructHamiltonian();
         double temp_psi[];
         ArrayResize(temp_psi, SchrodingerLookback, SchrodingerLookback+5);
         int iterations = UseSimplifiedQuantum ? MathMin(3, MaxIterations) : MaxIterations;
         for(int t=0; t<iterations; t++)
           {
            ArrayCopy(temp_psi, psi_wave);
            for(int i=0; i<SchrodingerLookback; i++)
              {
               double H_psi = 0;
               if(UseSimplifiedQuantum)
                 {
                  H_psi = hamiltonian[i * SchrodingerLookback + i] * psi_wave[i];
                  if(i > 0)
                     H_psi += hamiltonian[i * SchrodingerLookback + (i-1)] * psi_wave[i-1];
                  if(i < SchrodingerLookback-1)
                     H_psi += hamiltonian[i * SchrodingerLookback + (i+1)] * psi_wave[i+1];
                 }
               else
                 {
                  for(int j=0; j<SchrodingerLookback; j++)
                     H_psi += hamiltonian[i * SchrodingerLookback + j] * psi_wave[j];
                 }
               temp_psi[i] = psi_wave[i] - H_psi * 0.02;
              }
            ArrayCopy(psi_wave, temp_psi);
            if(t == iterations-1 || t % 2 == 0)
              {
               double norm = 0;
               for(int i=0; i<SchrodingerLookback; i++)
                  norm += psi_wave[i] * psi_wave[i];
               norm = MathSqrt(norm);
               if(norm > 0)
                 {
                  double invNorm = 1.0 / norm;
                  for(int i=0; i<SchrodingerLookback; i++)
                     psi_wave[i] *= invNorm;
                 }
              }
           }
         calculation_valid = true;
         last_calculation_time = TimeCurrent();
        }
     }
  }

//+------------------------------------------------------------------+
//| Calculate market prediction based on Schrödinger wave function   |
//+------------------------------------------------------------------+
double CalculateSchrodingerPrediction()
  {
   if(UseCachedCalculations && calculation_valid && cached_prediction != 0)
      return cached_prediction;
   SolveSchrodingerEquation();
   double expectation_position = 0;
   double expectation_momentum = 0;
   double uncertainty = 0;
   for(int i=0; i<SchrodingerLookback; i++)
     {
      double prob = psi_wave[i] * psi_wave[i];
      expectation_position += i * prob;
     }
   for(int i=1; i<SchrodingerLookback-1; i++)
     {
      double momentum_i = (psi_wave[i+1] - psi_wave[i-1]) / 2.0;
      double recency_weight = 1.0 + 0.5 * ((double)(SchrodingerLookback-i) / SchrodingerLookback);
      expectation_momentum += momentum_i * psi_wave[i] * recency_weight;
     }
   if(!UseSimplifiedQuantum)
     {
      for(int i=0; i<SchrodingerLookback; i++)
         uncertainty += MathPow(i - expectation_position, 2) * psi_wave[i] * psi_wave[i];
      uncertainty = MathSqrt(uncertainty);
     }
   else
     {
      uncertainty = SchrodingerLookback / 4.0;
     }
   double certainty_factor = 1.0 / (1.0 + uncertainty / 10.0);
   double prediction = expectation_momentum * 150.0 * certainty_factor;
   cached_prediction = prediction;
   if(EnableDebugMode)
      Print("Schrödinger prediction: ", prediction, " (certainty: ", certainty_factor, ")");
   return prediction;
  }

//+------------------------------------------------------------------+
//| Optimized Quantum Indicator                                      |
//+------------------------------------------------------------------+
double QuantumIndicator()
  {
   if(UseCachedCalculations && calculation_valid && cached_threshold != 0)
      return cached_threshold;
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   if(CopyBuffer(ma_handle, 0, 0, 1, ma_buffer) <= 0)
     {
      Print("Error copying MA data");
      return currentPrice;
     }
   double ma = ma_buffer[0];
   if(CopyBuffer(rsi_handle, 0, 0, 1, rsi_buffer) <= 0)
     {
      Print("Error copying RSI data");
      return currentPrice;
     }
   double rsi = rsi_buffer[0];
   double quantumFactor = 0;
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      double sq = (i+1) * (i+1);
      quantumFactor += QuantumState[i] * sq;
     }
   quantumFactor /= (NUM_Q_STATES * NUM_Q_STATES);
   double schrodingerPrediction = CalculateSchrodingerPrediction();
   double rsiBias = (rsi < 40) ? (40 - rsi) * 0.0001 : 0.0;
   double threshold = ma + (quantumFactor * 0.0005) + (schrodingerPrediction * 0.0005) - rsiBias;
   cached_threshold = threshold;
   return threshold;
  }

//+------------------------------------------------------------------+
//| Calculate trade costs                                            |
//+------------------------------------------------------------------+
double CalculateTradeCosts(double lotSize, ENUM_ORDER_TYPE orderType, int daysHeld = 1)
  {
   double commission = CommissionPerLot * lotSize;
   double spread = SpreadAverage * pip_value * (SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) / lotSize);
   double swap = (orderType == ORDER_TYPE_BUY ? SwapLong : SwapShort) * daysHeld * lotSize;
   return(commission + spread + MathAbs(swap));
  }

//+------------------------------------------------------------------+
//| Efficiently calculate potential profit                            |
//+------------------------------------------------------------------+
double CalculatePotentialProfit(double entryPrice, double targetPrice, double lotSize, ENUM_ORDER_TYPE orderType)
  {
   double priceDiff = (orderType == ORDER_TYPE_BUY) ? (targetPrice - entryPrice) : (entryPrice - targetPrice);
   static double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   static double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double rawProfit = priceDiff * lotSize * tickValue / tickSize;
   double costs = CalculateTradeCosts(lotSize, orderType);
   return rawProfit - costs;
  }

//+------------------------------------------------------------------+
//| Check Advanced Trade Entry with optimized calculations            |
//+------------------------------------------------------------------+
bool CheckTradeEntryAdvanced()
  {
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double threshold = QuantumIndicator();
   double prediction = CalculateSchrodingerPrediction();
   double lotSize = InpInitialLotSize;
   double targetPrice = currentPrice + (prediction * pip_value);
   double potentialProfit = CalculatePotentialProfit(currentPrice, targetPrice, lotSize, ORDER_TYPE_BUY);
   if(currentPrice <= threshold * 0.998)
      return false;
   if(prediction <= -0.5 && potentialProfit <= -0.5)
      return false;
   if(EnableDebugMode)
      Print("Trade Check - Price: ", currentPrice, " Threshold: ", threshold, ", Prediction: ", prediction, ", Profit: ", potentialProfit);
   return true;
  }

//+------------------------------------------------------------------+
//| Calculate position size based on risk management                 |
//+------------------------------------------------------------------+
double CalculatePositionSize(double stopLossPips)
  {
   if(stopLossPips <= 0) return InpInitialLotSize;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (RiskPercent / 100.0);
   static double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   static double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pipValue = (pip_value / tickSize) * tickValue;
   double positionSize = riskAmount / (stopLossPips * pipValue);
   static double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   static double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   positionSize = MathMin(maxLot, MathMax(minLot, MathFloor(positionSize * 100) / 100));
   return positionSize;
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   calculationCounter++;
   if(TimeCurrent() - last_calculation_time > 60)
      calculation_valid = false;
   bool performCalculations = (calculationCounter % CalculationFrequency == 0) || !calculation_valid;
   if(performCalculations)
     {
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0)
        {
         Print("Error copying ATR data");
         return;
        }
      double marketVolatility = atr_buffer[0];
      EvolveQuantumState(marketVolatility);
      calculation_valid = false;
      cached_prediction = 0;
      cached_threshold = 0;
     }
   if(CheckTradeEntryAdvanced())
     {
      if(PositionsTotal() < MaxOpenPositions)
        {
         double prediction = CalculateSchrodingerPrediction();
         double lotSize = CalculatePositionSize(QuantumSL);
         double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double predictionFactor = MathMax(0.5, MathMin(2.0, 1.0 + prediction / 50.0));
         double adjustedTP = QuantumTP * predictionFactor;
         double targetPrice = bidPrice + (adjustedTP * pip_value);
         double potentialProfit = CalculatePotentialProfit(bidPrice, targetPrice, lotSize, ORDER_TYPE_BUY);
         if(potentialProfit > -1.0 || prediction > 2.0)
           {
            if(trade.Buy(lotSize, _Symbol, askPrice, askPrice - QuantumSL*pip_value, askPrice + adjustedTP*pip_value, "EinsteinEvolved4 Buy Order"))
              {
               Print("Buy order executed at price ", askPrice, " lot size: ", lotSize);
               Print("Adjusted TP: ", adjustedTP, " pips, Potential profit: ", DoubleToString(potentialProfit, 2));
               tradeCounter++;
              }
            else
              {
               Print("Buy order failed. Error: ", GetLastError());
              }
           }
         else if(EnableDebugMode)
           {
            Print("Trade skipped - insufficient potential profit: ", DoubleToString(potentialProfit, 2));
           }
        }
      else if(EnableDebugMode)
        {
         Print("Max positions reached (", MaxOpenPositions, ")");
        }
     }
  } 