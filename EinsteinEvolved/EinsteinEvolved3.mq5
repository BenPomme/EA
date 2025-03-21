//+------------------------------------------------------------------+
//|                                                   EinsteinEvolved3.mq5               |
//|                        Einstein Evolved: Optimized Quantum & ML EA  |
//|                                                                  |
//|   Integrates quantum physics concepts and machine learning with  |
//|   optimized computational performance and resource usage         |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Benjamin Pommeraud"
#property link      "https://www.quantumtradinglab.com"
#property version   "3.00"
#property strict
#property description "EinsteinEvolved3: Computationally Optimized Quantum EA"

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>

//--- Input parameters - Performance Optimization
input int    CalculationFrequency     = 5;               // Frequency of quantum calculations (in ticks)
input bool   UseCachedCalculations    = true;            // Use cached calculations to reduce CPU load
input bool   UseSimplifiedQuantum     = false;           // Use simplified quantum calculations (faster but less accurate)
input int    MaxIterations            = 5;               // Maximum iterations for Schrödinger solution (lower = faster)

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
   trade.SetExpertMagicNumber(98767);  // New magic number for version 3
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
     
   // Pre-allocate memory for all arrays (avoids reallocation during runtime)
   // Indicator buffers
   ArrayResize(atr_buffer, 3, 10);
   ArrayResize(rsi_buffer, 3, 10);
   ArrayResize(ma_buffer, 3, 10);
   
   // Allocate price history buffers with pre-allocation reserve
   ArrayResize(close_buffer, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(high_buffer, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(low_buffer, SchrodingerLookback, SchrodingerLookback+10);
   
   // Initialize Quantum Module
   InitializeQuantumModule();
   
   // Initialize Schrödinger equation components with pre-allocation
   InitializeSchrodingerEquation();
   
   Print("EinsteinEvolved3 initialized with optimized performance settings");
   Print("Calculation frequency: Every ", CalculationFrequency, " ticks, Max iterations: ", MaxIterations);
   
   return(INIT_SUCCEEDED);
  }
  
//+------------------------------------------------------------------+
//| Initialize Quantum Module                                        |
//+------------------------------------------------------------------+
void InitializeQuantumModule()
  {
   NUM_Q_STATES = InpQuantumStates;
   // Pre-allocate with extra capacity
   ArrayResize(QuantumState, NUM_Q_STATES, NUM_Q_STATES+5);
   
   // Initialize to equal probability distribution
   double equalProb = 1.0 / NUM_Q_STATES;
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      QuantumState[i] = equalProb;
     }
     
   if(EnableDebugMode)
      Print("Quantum Module initialized with ", NUM_Q_STATES, " states.");
  }

//+------------------------------------------------------------------+
//| Initialize Schrödinger Equation Components                       |
//+------------------------------------------------------------------+
void InitializeSchrodingerEquation()
  {
   // Pre-allocate all arrays with additional capacity to avoid resizing
   ArrayResize(psi_wave, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(potential_array, SchrodingerLookback, SchrodingerLookback+10);
   ArrayResize(market_momentum, SchrodingerLookback, SchrodingerLookback+10);
   
   // For optimal memory access pattern, we store the Hamiltonian matrix as a 1D array
   // with proper indexing to improve cache locality
   ArrayResize(hamiltonian, SchrodingerLookback * SchrodingerLookback, SchrodingerLookback * SchrodingerLookback+20);
   
   // Initialize all arrays
   ArrayInitialize(psi_wave, 0);
   ArrayInitialize(potential_array, 0);
   ArrayInitialize(hamiltonian, 0);
   ArrayInitialize(market_momentum, 0);
   
   // Initial wave function as Gaussian centered at half of lookback period
   double norm_factor = 0.0;
   int center = SchrodingerLookback / 2;
   double sigma = SchrodingerLookback / 6.0;
   
   for(int i=0; i<SchrodingerLookback; i++)
     {
      psi_wave[i] = MathExp(-0.5 * MathPow((i - center) / sigma, 2));
      norm_factor += MathPow(psi_wave[i], 2);
     }
   
   // Normalize
   norm_factor = MathSqrt(norm_factor);
   if(norm_factor > 0)
     {
      for(int i=0; i<SchrodingerLookback; i++)
         psi_wave[i] /= norm_factor;
     }
   
   if(EnableDebugMode)
      Print("Schrödinger equation components initialized with optimized memory allocation");
  }
  
//+------------------------------------------------------------------+
//| Evolve Quantum State based on market volatility                  |
//+------------------------------------------------------------------+
void EvolveQuantumState(double marketFactor)
  {
   // Pre-calculate sigmoid to avoid repeated calculations
   double sigmoid = 1.0 / (1.0 + MathExp(-marketFactor * 2.0));
   double sum = 0.0;
   
   // Optimized loop with precalculated values
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      // Faster evolution formula
      QuantumState[i] *= (0.5 + sigmoid/(i+1)) * InpQuantumDecay;
      sum += QuantumState[i];
     }
   
   // Only normalize if sum is valid
   if(sum > 0)
     {
      // Inverse multiplication is faster than division in a loop
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
   // Optimized collapse - early return optimization
   double rnd = (double)MathRand()/32767.0;
   double accum = 0.0;
   
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      accum += QuantumState[i];
      if(rnd < accum)  // Early return when condition is met
         return i;
     }
   
   return(NUM_Q_STATES-1);
  }
  
//+------------------------------------------------------------------+
//| Quantum Signal Adjustment - return integer adjustment            |
//+------------------------------------------------------------------+
int QuantumSignalAdjustment()
  {
   int stateIndex = CollapseQuantumState();
   // Simplified calculation
   return (stateIndex - (NUM_Q_STATES / 2)) * 2;
  }
  
//+------------------------------------------------------------------+
//| Update Quantum Potential based on price movements                |
//+------------------------------------------------------------------+
void UpdateQuantumPotential()
  {
   // Efficiently copy price data
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, close_buffer) <= 0 ||
      CopyHigh(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, high_buffer) <= 0 ||
      CopyLow(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, low_buffer) <= 0)
     {
      Print("Error copying price data");
      return;
     }
   
   // Optimized loop for momentum calculation with better cache utilization
   for(int i=0; i<SchrodingerLookback-1; i++)
     {
      double weight = 1.0 + 0.1*(SchrodingerLookback-i-1)/SchrodingerLookback;
      market_momentum[i] = (close_buffer[i] - close_buffer[i+1]) * weight;
     }
   market_momentum[SchrodingerLookback-1] = 0;
   
   // Optimized potential energy calculation
   for(int i=0; i<SchrodingerLookback; i++)
     {
      // Precalculate values to avoid repeated calculations
      double range = high_buffer[i] - low_buffer[i];
      double wellDepth = (range / point) * QuantumPotentialScaling;
      double momentumEffect = MathPow(MathAbs(market_momentum[i]), 1.5) * QuantumPotentialScaling;
      
      // Single assignment is more efficient
      potential_array[i] = wellDepth * PotentialWellDepth - momentumEffect;
     }
  }

//+------------------------------------------------------------------+
//| Construct Hamiltonian Operator - optimized version               |
//+------------------------------------------------------------------+
void ConstructHamiltonian()
  {
   // Optimized Hamiltonian construction
   
   // Precalculate constant factor
   double hbar_squared_over_2m = MathPow(PlankConstant, 2) / (2.0 * WaveFunctionDiffusion);
   
   // Optimized for better memory access patterns
   for(int i=0; i<SchrodingerLookback; i++)
     {
      // Diagonal terms first (contiguous memory access)
      hamiltonian[i * SchrodingerLookback + i] = potential_array[i] + (2.0 * hbar_squared_over_2m);
     }
     
   // Then off-diagonal terms
   for(int i=0; i<SchrodingerLookback-1; i++)
     {
      // Set symmetric off-diagonal elements (more cache-friendly)
      hamiltonian[i * SchrodingerLookback + (i+1)] = -hbar_squared_over_2m;
      hamiltonian[(i+1) * SchrodingerLookback + i] = -hbar_squared_over_2m;
     }
  }

//+------------------------------------------------------------------+
//| Optimized Schrödinger Equation Solver                            |
//+------------------------------------------------------------------+
void SolveSchrodingerEquation()
  {
   // Only update potential if cache is invalid or forced calculation
   if(!calculation_valid || !UseCachedCalculations)
     {
      UpdateQuantumPotential();
      ConstructHamiltonian();
      
      // Use optimized time-evolution method
      
      // Temporary array for wave function evolution
      double temp_psi[];
      ArrayResize(temp_psi, SchrodingerLookback, SchrodingerLookback+5);
      
      // Optimized evolution algorithm
      int iterations = UseSimplifiedQuantum ? MathMin(3, MaxIterations) : MaxIterations;
      
      for(int t=0; t<iterations; t++)
        {
         // Copy current state to temp (only once per iteration)
         ArrayCopy(temp_psi, psi_wave);
         
         // Optimized matrix-vector multiplication (Hamiltonian * psi)
         for(int i=0; i<SchrodingerLookback; i++)
           {
            double H_psi = 0;
            
            // Simplified calculation for performance: only use tridiagonal part of Hamiltonian
            // This is a very good approximation for the Schrödinger equation in this context
            if(UseSimplifiedQuantum)
              {
               // Diagonal element
               H_psi = hamiltonian[i * SchrodingerLookback + i] * psi_wave[i];
               
               // Lower off-diagonal (if exists)
               if(i > 0)
                  H_psi += hamiltonian[i * SchrodingerLookback + (i-1)] * psi_wave[i-1];
                  
               // Upper off-diagonal (if exists)
               if(i < SchrodingerLookback-1)
                  H_psi += hamiltonian[i * SchrodingerLookback + (i+1)] * psi_wave[i+1];
              }
            else
              {
               // Full matrix-vector product (slower but more accurate)
               for(int j=0; j<SchrodingerLookback; j++)
                 {
                  H_psi += hamiltonian[i * SchrodingerLookback + j] * psi_wave[j];
                 }
              }
            
            // Evolution step (simplified for better performance)
            temp_psi[i] = psi_wave[i] - H_psi * 0.02;
           }
           
         // Copy evolved state back
         ArrayCopy(psi_wave, temp_psi);
         
         // Normalize only on last iteration or every 2 iterations
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
      
      // Mark calculation as valid for caching
      calculation_valid = true;
      last_calculation_time = TimeCurrent();
     }
  }

//+------------------------------------------------------------------+
//| Calculate market prediction based on Schrödinger wave function   |
//+------------------------------------------------------------------+
double CalculateSchrodingerPrediction()
  {
   // If caching is enabled and calculation is valid, return cached value
   if(UseCachedCalculations && calculation_valid && cached_prediction != 0)
      return cached_prediction;
      
   // Solve the Schrödinger equation to update wave function
   SolveSchrodingerEquation();
   
   // Optimized expectation value calculations
   double expectation_position = 0;
   double expectation_momentum = 0;
   double uncertainty = 0;
   
   // First pass: calculate position expectation
   for(int i=0; i<SchrodingerLookback; i++)
     {
      double prob = psi_wave[i] * psi_wave[i];  // |ψ|²
      expectation_position += i * prob;
     }
   
   // Second pass: calculate momentum expectation with recency weighting
   for(int i=1; i<SchrodingerLookback-1; i++)
     {
      // Finite difference approximation of momentum operator
      double momentum_i = (psi_wave[i+1] - psi_wave[i-1]) / 2.0;
      
      // Apply recency weighting
      double recency_weight = 1.0 + 0.5 * ((double)(SchrodingerLookback-i) / SchrodingerLookback);
      
      expectation_momentum += momentum_i * psi_wave[i] * recency_weight;
     }
   
   // Calculate uncertainty (optional - can be skipped for performance if not needed)
   if(!UseSimplifiedQuantum)
     {
      for(int i=0; i<SchrodingerLookback; i++)
        {
         uncertainty += MathPow(i - expectation_position, 2) * psi_wave[i] * psi_wave[i];
        }
      uncertainty = MathSqrt(uncertainty);
     }
   else
     {
      // Simplified uncertainty estimate
      uncertainty = SchrodingerLookback / 4.0;
     }
   
   // Calculate certainty factor (higher when more certain)
   double certainty_factor = 1.0 / (1.0 + uncertainty / 10.0);
   
   // Scale prediction based on certainty
   double prediction = expectation_momentum * 150.0 * certainty_factor;
   
   // Cache the prediction
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
   // If caching is enabled and calculation is valid, return cached value
   if(UseCachedCalculations && calculation_valid && cached_threshold != 0)
      return cached_threshold;
      
   // Get current price
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   // Get MA value efficiently
   if(CopyBuffer(ma_handle, 0, 0, 1, ma_buffer) <= 0)
     {
      Print("Error copying MA data");
      return currentPrice;
     }
   double ma = ma_buffer[0];
   
   // Get RSI efficiently
   if(CopyBuffer(rsi_handle, 0, 0, 1, rsi_buffer) <= 0)
     {
      Print("Error copying RSI data");
      return currentPrice;
     }
   double rsi = rsi_buffer[0];
   
   // Calculate quantum factor efficiently
   double quantumFactor = 0;
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      // Square calculation inline for efficiency
      double sq = (i+1) * (i+1);
      quantumFactor += QuantumState[i] * sq;
     }
   
   // Normalize by precalculated value
   quantumFactor /= (NUM_Q_STATES * NUM_Q_STATES);
   
   // Get Schrödinger prediction (will use cache if valid)
   double schrodingerPrediction = CalculateSchrodingerPrediction();
   
   // Calculate RSI bias efficiently
   double rsiBias = (rsi < 40) ? (40 - rsi) * 0.0001 : 0.0;
   
   // Combine all factors
   double threshold = ma + (quantumFactor * 0.0005) + (schrodingerPrediction * 0.0005) - rsiBias;
   
   // Cache the threshold
   cached_threshold = threshold;
   
   return threshold;
  }
  
//+------------------------------------------------------------------+
//| Calculate trade costs                                            |
//+------------------------------------------------------------------+
double CalculateTradeCosts(double lotSize, ENUM_ORDER_TYPE orderType, int daysHeld = 1)
  {
   // Calculate all costs in one efficient formula
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
   // Calculate price difference based on order type
   double priceDiff = (orderType == ORDER_TYPE_BUY) ? (targetPrice - entryPrice) : (entryPrice - targetPrice);
   
   // Get tick value and size once (avoid repeated API calls)
   static double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   static double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   // Calculate raw profit
   double rawProfit = priceDiff * lotSize * tickValue / tickSize;
   
   // Subtract costs
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
   
   // Get prediction (uses cache if available)
   double prediction = CalculateSchrodingerPrediction();
   
   // Calculate potential profit
   double lotSize = InpInitialLotSize;
   double targetPrice = currentPrice + (prediction * pip_value);
   double potentialProfit = CalculatePotentialProfit(currentPrice, targetPrice, lotSize, ORDER_TYPE_BUY);
   
   // Efficient condition checking with early returns
   if(currentPrice <= threshold * 0.998)
      return false;
      
   if(prediction <= -0.5 && potentialProfit <= -0.5)
      return false;
      
   // Log conditions if debug enabled
   if(EnableDebugMode)
     {
      Print("Trade Check - Price: ", currentPrice, " Threshold: ", threshold, 
            " Prediction: ", prediction, " Profit: ", potentialProfit);
     }
   
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
   
   // Calculate position size from risk parameters
   static double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   static double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double pipValue = (pip_value / tickSize) * tickValue;
   
   double positionSize = riskAmount / (stopLossPips * pipValue);
   
   // Apply lot size constraints
   static double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   static double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   
   // Efficient bounds checking
   positionSize = MathMin(maxLot, MathMax(minLot, MathFloor(positionSize * 100) / 100));
   
   return positionSize;
  }
  
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Only perform heavy calculations periodically
   calculationCounter++;
   
   // Invalidate cache if sufficient time has passed
   if(TimeCurrent() - last_calculation_time > 60) // 60 seconds cache expiry
      calculation_valid = false;
      
   // Decide whether to perform calculations on this tick
   bool performCalculations = (calculationCounter % CalculationFrequency == 0) || !calculation_valid;
   
   if(performCalculations)
     {
      // Update market volatility from ATR
      if(CopyBuffer(atr_handle, 0, 0, 1, atr_buffer) <= 0)
        {
         Print("Error copying ATR data");
         return;
        }
      double marketVolatility = atr_buffer[0];
      
      // Evolve quantum state based on market volatility
      EvolveQuantumState(marketVolatility);
      
      // Cache will be updated by first call to CalculateSchrodingerPrediction
      // or QuantumIndicator when needed
      calculation_valid = false;
      cached_prediction = 0;
      cached_threshold = 0;
     }
   
   // Always check for trade entry (uses cached values when available)
   if(CheckTradeEntryAdvanced())
     {
      if(PositionsTotal() < MaxOpenPositions)
        {
         // Get prediction (uses cache if available)
         double prediction = CalculateSchrodingerPrediction();
         
         // Calculate optimal lot size
         double lotSize = CalculatePositionSize(QuantumSL);
         
         // Get current price
         double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         
         // Calculate adjusted take profit
         double predictionFactor = MathMax(0.5, MathMin(2.0, 1.0 + prediction / 50.0));
         double adjustedTP = QuantumTP * predictionFactor;
         
         // Calculate potential profit
         double targetPrice = bidPrice + (adjustedTP * pip_value);
         double potentialProfit = CalculatePotentialProfit(bidPrice, targetPrice, lotSize, ORDER_TYPE_BUY);
         
         // Execute trade if conditions met
         if(potentialProfit > -1.0 || prediction > 2.0)
           {
            if(trade.Buy(lotSize, _Symbol, askPrice, askPrice - QuantumSL*pip_value, 
                          askPrice + adjustedTP*pip_value, "EinsteinEvolved3 Buy Order"))
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
  
//+------------------------------------------------------------------+
//| Track closed positions to update last trade profit               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
  {
   // Only process relevant transaction types (performance optimization)
   if(trans.type != TRADE_TRANSACTION_DEAL_ADD || trans.deal_type != DEAL_TYPE_SELL)
      return;
      
   // Position might have been closed, update lastTradeProfit
   HistorySelect(TimeCurrent()-86400, TimeCurrent()); // Last 24 hours
   
   // Check only recent history
   for(int i=HistoryDealsTotal()-1; i>=MathMax(0, HistoryDealsTotal()-10); i--)
     {
      ulong ticket = HistoryDealGetTicket(i);
      if(ticket > 0)
        {
         // Check if deal closed our position
         if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == trade.RequestMagic() && 
            HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
           {
            lastTradeProfit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
            
            if(EnableDebugMode)
               Print("Last trade closed with profit: ", lastTradeProfit);
               
            // Reset cache to force recalculation after a trade
            calculation_valid = false;
            cached_prediction = 0;
            cached_threshold = 0;
            
            break;
           }
        }
     }
  }
  
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   // Release indicator handles
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   if(ma_handle  != INVALID_HANDLE) IndicatorRelease(ma_handle);
   
   Print("EinsteinEvolved3 deinitialized.");
  } 