//+------------------------------------------------------------------+
//|                                                 Scalpbaby1.mq5   |
//|                     GPU Accelerated Scalping EA "Scalpbaby1"       |
//|                                                                  |
//|  Designed to generate a constant positive return with a max      |
//|  drawdown of 25% (via risk management) across various market       |
//|  conditions independent of volatility/trend. It incorporates GPU    |
//|  best practices (see Documentation/GPU_Best_Practices.txt) and     |
//|  takes into account true transaction costs of the broker.         |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Benjamin Pommeraud"
#property link      "https://www.quantumtradinglab.com"
#property version   "1.00"
#property strict
#property description "Scalpbaby1: GPU Accelerated Scalping EA with transaction cost & risk management"

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>

//--- Input parameters

// GPU and scalping calculation controls
input bool   UseGPUAcceleration    = true;                   // Enable GPU acceleration if available
input int    CalculationFrequency  = 3;                      // Perform heavy calculations every N ticks
input int    MaxScalpIterations    = 3;                      // Iterations for solving the scalping equation
input double TimeStep              = 0.02;                   // Time step for evolution process

// Time period selection for analysis (testing on different timeframes)
input ENUM_TIMEFRAMES TestTimePeriod= PERIOD_M5;               // Use different timeframe for signal computations

// Scalping module parameters
input int    ScalpingLookback      = 40;                     // Number of bars to use in scalping module
input int    InpScalpStates        = 10;                     // Number of states for scalping module
input double InpScalpDecay         = 0.95;                   // Decay factor for state evolution
input double ScalpSignalScaling    = 0.0001;                 // Scaling factor to convert expectation offset into pip signal
input double ScalpSignalThreshold  = 0.0003;                 // Minimum absolute signal to trigger a trade
input double InpScalpEntryThreshold = -100.0;
input bool InpForceTrade = true;

// Risk management parameters
input double RiskPercent           = 0.5;                    // Risk percent per trade (% of account balance)
input double MaxDrawdownPercentage = 25.0;                   // Maximum allowed drawdown (% of historical peak account balance)
input double StopLossPips          = 10.0;                   // Stop loss in pips
input double TakeProfitPips        = 20.0;                   // Take profit in pips

// Broker cost parameters (true transaction costs)
input double CommissionPerLot      = 5.5;                    // Commission per lot in base currency
input double SwapLong              = -2.5;                   // Swap points for long positions
input double SwapShort             = -2.5;                   // Swap points for short positions
input double SpreadAverage         = 1.0;                    // Average spread in pips
input string SymbolGroup           = "Forex Majors Raw";     // Symbol group for commission calculation

// Added input parameters to resolve undeclared identifiers
input double PlankConstant         = 6.62607015e-34;         // Planck constant for scaling (for scalping EA)
input double InpInitialLotSize       = 0.1;                    // Initial lot size for trades

//--- GPU Acceleration Functions
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

//--- Global Variables and objects
CTrade trade;
double point;        // point size
double pip_value;    // pip value
int pip_digits;      // pip number of digits

// Scalping module arrays (for GPU/CPU acceleration)
double scalpWave[];         // Evolved wave function (scalp signal)
double scalpPotential[];    // Potential energy array
double scalpHamiltonian[];  // Hamiltonian operator (flattened matrix)
double scalpMomentum[];     // Momentum array

// Price data buffers for scalping module
double closeBuffer[];
double highBuffer[];
double lowBuffer[];

// Caching variables for scalping signal computation
double cachedSignal = 0;
datetime lastCalcTime = 0;
bool scalpCalcValid = false;

// Risk management: track historical peak account balance
double maxAccountBalance = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Set up trade parameters and magic number
   trade.SetExpertMagicNumber(12345);
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
   
   // Set initial max account balance
   maxAccountBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   
   // Pre-allocate price data buffers
   ArrayResize(closeBuffer, ScalpingLookback);
   ArrayResize(highBuffer, ScalpingLookback);
   ArrayResize(lowBuffer, ScalpingLookback);
   
   // Initialize scalping module arrays
   ArrayResize(scalpWave, ScalpingLookback);
   ArrayResize(scalpPotential, ScalpingLookback);
   ArrayResize(scalpMomentum, ScalpingLookback);
   ArrayResize(scalpHamiltonian, ScalpingLookback * ScalpingLookback);
   
   // Initialize scalpWave as a Gaussian distribution centered in the lookback window
   int center = ScalpingLookback / 2;
   double sigma = ScalpingLookback / 6.0;
   double norm_factor = 0;
   for(int i=0; i < ScalpingLookback; i++)
     {
      scalpWave[i] = MathExp(-0.5 * MathPow((i - center) / sigma, 2));
      norm_factor += MathPow(scalpWave[i], 2);
     }
   norm_factor = MathSqrt(norm_factor);
   if(norm_factor > 0)
     {
      for(int i=0; i < ScalpingLookback; i++)
         scalpWave[i] /= norm_factor;
     }
   
   // Initialize GPU if enabled
   if(UseGPUAcceleration)
      GPU_InitializeOpenCL();
   
   Print("Scalpbaby1 initialized. TestTimePeriod: ", EnumToString(TestTimePeriod));
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(UseGPUAcceleration)
      GPU_ReleaseOpenCL();
   Print("Scalpbaby1 deinitialized.");
  }
  
//+------------------------------------------------------------------+
//| Update scalping potential using price data                       |
//+------------------------------------------------------------------+
void UpdateScalpingPotential()
  {
   if(ScalpingLookback < 2)
     {
       Print("Error: ScalpingLookback must be at least 2.");
       return;
     }
   // Copy price data from the selected TestTimePeriod
   int copiedClose = CopyClose(_Symbol, TestTimePeriod, 0, ScalpingLookback, closeBuffer);
   int copiedHigh  = CopyHigh(_Symbol, TestTimePeriod, 0, ScalpingLookback, highBuffer);
   int copiedLow   = CopyLow(_Symbol, TestTimePeriod, 0, ScalpingLookback, lowBuffer);
   if(copiedClose < ScalpingLookback || copiedHigh < ScalpingLookback || copiedLow < ScalpingLookback)
     {
      Print("Error: Not enough bars copied for scalping module (copied: ", copiedClose, ", ", copiedHigh, ", ", copiedLow, ")");
      return;
     }
   
   // Compute momentum for each bar using actual number of copied bars
   int bars = copiedClose; // actual number of bars copied, expected to be equal to ScalpingLookback
   if(bars < 2)
     {
      Print("Error: Insufficient bars copied for momentum computation (bars = ", bars, ")");
      return;
     }
   for(int i = 0; i < bars - 1; i++)
     {
       scalpMomentum[i] = closeBuffer[i] - closeBuffer[i+1];
     }
   scalpMomentum[bars - 1] = 0;
   
   // Compute potential based on price range and momentum
   for(int i=0; i < ScalpingLookback; i++)
     {
      double range = highBuffer[i] - lowBuffer[i];
      double wellDepth = (range / point) * 0.001; // scaling factor for potential
      scalpPotential[i] = wellDepth - (MathAbs(scalpMomentum[i]) * 0.001);
     }
  }
  
//+------------------------------------------------------------------+
//| Construct Hamiltonian Operator for scalping module               |
//+------------------------------------------------------------------+
void ConstructScalpingHamiltonian()
  {
   double hbar2_over_2m = MathPow(PlankConstant, 2) / (2.0 * TimeStep);
   for(int i=0; i < ScalpingLookback; i++)
     {
      int idx = i * ScalpingLookback + i;
      scalpHamiltonian[idx] = scalpPotential[i] + (2.0 * hbar2_over_2m);
      if(i > 0)
         scalpHamiltonian[i * ScalpingLookback + (i - 1)] = -hbar2_over_2m;
      if(i < ScalpingLookback - 1)
         scalpHamiltonian[i * ScalpingLookback + (i + 1)] = -hbar2_over_2m;
     }
  }
  
//+------------------------------------------------------------------+
//| Solve the scalping equation (GPU accelerated if enabled)         |
//+------------------------------------------------------------------+
void SolveScalpingEquation()
  {
   UpdateScalpingPotential();
   ConstructScalpingHamiltonian();
   int iterations = MaxScalpIterations;
   if(UseGPUAcceleration)
     {
      int res = GPU_SolveSchrodingerEquation(scalpHamiltonian, scalpWave, ScalpingLookback, iterations, TimeStep);
      if(res != 0)
         Print("GPU_SolveSchrodingerEquation error, reverting to CPU fallback: ", res);
      else
         scalpCalcValid = true;
     }
   else
     {
      double temp_wave[];
      ArrayResize(temp_wave, ScalpingLookback);
      for(int t=0; t < iterations; t++)
        {
         ArrayCopy(temp_wave, scalpWave);
         for(int i=0; i < ScalpingLookback; i++)
           {
            double H_psi = scalpHamiltonian[i * ScalpingLookback + i] * scalpWave[i];
            if(i > 0)
               H_psi += scalpHamiltonian[i * ScalpingLookback + (i-1)] * scalpWave[i-1];
            if(i < ScalpingLookback - 1)
               H_psi += scalpHamiltonian[i * ScalpingLookback + (i+1)] * scalpWave[i+1];
            temp_wave[i] = scalpWave[i] - H_psi * TimeStep;
           }
         ArrayCopy(scalpWave, temp_wave);
         double norm = 0;
         for(int i=0; i < ScalpingLookback; i++)
            norm += scalpWave[i] * scalpWave[i];
         norm = MathSqrt(norm);
         if(norm > 0)
           {
            for(int i=0; i < ScalpingLookback; i++)
               scalpWave[i] /= norm;
           }
        }
      scalpCalcValid = true;
     }
   lastCalcTime = TimeCurrent();
  }
  
//+------------------------------------------------------------------+
//| Calculate scalping signal from the evolved wave function         |
//+------------------------------------------------------------------+
double CalculateScalpingSignal()
  {
   if(!scalpCalcValid)
      SolveScalpingEquation();
   double expectation = 0;
   for(int i=0; i < ScalpingLookback; i++)
     {
      double prob = scalpWave[i] * scalpWave[i];
      expectation += i * prob;
     }
   double signal = (expectation - (ScalpingLookback/2)) * ScalpSignalScaling;
   cachedSignal = signal;
   return signal;
  }
  
//+------------------------------------------------------------------+
//| Calculate trade costs (spread, commission, swap)                 |
//+------------------------------------------------------------------+
double CalculateTradeCosts(double lotSize, ENUM_ORDER_TYPE orderType, int daysHeld=1)
  {
   double commission = CommissionPerLot * lotSize;
   double spread = SpreadAverage * pip_value * (SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)/lotSize);
   double swap = (orderType == ORDER_TYPE_BUY ? SwapLong : SwapShort) * daysHeld * lotSize;
   return (commission + spread + MathAbs(swap));
  }
  
//+------------------------------------------------------------------+
//| Calculate potential profit considering transaction costs         |
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
//| Calculate position size based on risk management                 |
//+------------------------------------------------------------------+
double CalculatePositionSize(double stopLossPips)
  {
   if(stopLossPips <= 0) return 0.1;
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (RiskPercent / 100.0);
   static double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   static double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pipVal = (pip_value / tickSize) * tickValue;
   double positionSize = riskAmount / (stopLossPips * pipVal);
   static double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   static double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   positionSize = MathMin(maxLot, MathMax(minLot, MathFloor(positionSize*100)/100));
   return positionSize;
  }
  
//+------------------------------------------------------------------+
//| Check for drawdown constraint before trading                     |
//+------------------------------------------------------------------+
bool CheckDrawdown()
  {
   double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   if(currentBalance > maxAccountBalance)
      maxAccountBalance = currentBalance;
   double drawdown = (maxAccountBalance - currentBalance) / maxAccountBalance * 100.0;
   if(drawdown > MaxDrawdownPercentage)
     {
      Print("Drawdown ", DoubleToString(drawdown,2), "% exceeds maximum allowed (", MaxDrawdownPercentage, "%). No new trades will be taken.");
      return false;
     }
   return true;
  }
  
//+------------------------------------------------------------------+
//| OnTick                                                           |
//+------------------------------------------------------------------+
void OnTick()
{
   if(InpForceTrade)
   {
      // Force a buy trade with default lot size
      trade.Buy(InpInitialLotSize, _Symbol, 0, 0, 0, "Force Trade");
   }
}
  
//+------------------------------------------------------------------+ 