//+------------------------------------------------------------------+
//|                                                   EinsteinEvolved2.mq5               |
//|                        Einstein Evolved: Quantum & ML Trading EA |
//|                                                                  |
//|   Integrates quantum physics concepts and machine learning to    |
//|   dynamically optimize trading parameters based on market data   |
//|   and performance. Combines features from EINSTEIN1 and EinsteinJR. |
//+------------------------------------------------------------------+
#property copyright "Copyright 2023, Benjamin Pommeraud"
#property link      "https://www.quantumtradinglab.com"
#property version   "2.00"
#property strict
#property description "EinsteinEvolved2: Enhanced Quantum Physics and Machine Learning EA"

//--- Include necessary libraries
#include <Trade\Trade.mqh>
#include <Math\Stat\Math.mqh>
#include <Math\Stat\Normal.mqh>

//--- Input parameters - Quantum Constants (Modified for more active trading)
input double PlankConstant            = 6.62607015e-24;  // Planck constant for scaling (increased for stronger effects)
input double QuantumDecayRate         = 0.65;            // Quantum decay factor (increased from 0.5)
input int    UncertaintyPeriod        = 12;              // Uncertainty window period (reduced from 16)
input double EntanglementThreshold    = 0.2;             // Entanglement threshold (reduced from 0.3)
input double WaveFunctionCollapse     = 1.5;             // Wave function collapse multiplier (increased from 1.0)

//--- Input parameters - Trading Execution (Adjusted for $5000 capital with 1:500 leverage)
input double RiskPercent              = 0.8;             // Risk percent per trade (increased from 0.5)
input double QuantumSL                = 20.0;            // Stop Loss in pips (reduced from 25)
input double QuantumTP                = 40.0;            // Take Profit in pips (reduced from 48)
input int    MaxOpenPositions         = 2;               // Maximum open positions (increased from 1)

//--- Input parameters - Machine Learning and Quantum Module (Optimized for more signals)
input int    InpTrainingFrequency     = 5;               // Trades after which to retrain model (reduced from 10)
input double InpInitialLotSize        = 0.2;             // Initial lot size for trades (increased for $5000 capital)
input int    InpSignalPeriod          = 10;              // Base signal period for indicator (reduced from 14)
input int    InpQuantumStates         = 12;              // Number of quantum states (increased from 10)
input double InpQuantumDecay          = 0.90;            // Decay factor (reduced from 0.95)
input bool   EnableDebugMode          = true;            // Enable detailed debug logging (set to true)

//--- Input parameters - Indicator Settings (Adjusted for more sensitivity)
input int    ATRPeriod                = 12;              // Period for ATR indicator (reduced from 14)
input int    RSI_Period               = 10;              // RSI period (reduced from 14)
input int    MAMethodPeriod           = 40;              // Period for Moving Average (reduced from 50)

//--- Input parameters - Advanced Schrödinger Settings (Modified for stronger signals)
input int    SchrodingerLookback      = 60;              // Lookback period (reduced from 80)
input int    PotentialWellDepth       = 30;              // Potential well depth factor (increased from 20)
input double QuantumPotentialScaling  = 0.005;           // Scaling factor (increased from 0.001)
input double WaveFunctionDiffusion    = 0.2;             // Diffusion coefficient (increased from 0.1)

//--- Input parameters - Commission and Fees (Kept unchanged as requested)
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

double OptimizedSignalPeriod;  // Adaptive signal period optimized via ML
double lastTradeProfit = 0.0;  // Profit from the last trade
int tradeCounter = 0;        // Counter for executed trades

//--- Quantum Module Variables
int NUM_Q_STATES;
double QuantumState[];

//--- Advanced Schrödinger Equation Variables
double psi_wave[];           // Wave function array
double potential_array[];    // Potential energy array
double hamiltonian[];        // Hamiltonian operator matrix
double eigen_values[];       // Eigenvalues of energy states
double eigen_vectors[];      // Eigenvectors of quantum states
double market_momentum[];    // Market momentum buffer

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
double volume_buffer[];

// Account variables for capital management
double initialCapital;
double maxLeverage;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   // Set up trade parameters
   trade.SetExpertMagicNumber(98766); // New magic number for version 2
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
     
   // Store account capital and leverage info
   initialCapital = 5000.0;  // As specified, $5000 initial capital
   maxLeverage = 500.0;      // As specified, 1:500 leverage
      
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
     
   // Allocate indicator buffers
   ArrayResize(atr_buffer, 3);
   ArrayResize(rsi_buffer, 3);
   ArrayResize(ma_buffer, 3);
   
   // Allocate price history buffers
   ArrayResize(close_buffer, SchrodingerLookback);
   ArrayResize(high_buffer, SchrodingerLookback);
   ArrayResize(low_buffer, SchrodingerLookback);
   ArrayResize(volume_buffer, SchrodingerLookback);
   
   // Initialize Quantum Module
   InitializeQuantumModule();
   
   // Initialize Schrödinger equation components
   InitializeSchrodingerEquation();
   
   Print("EinsteinEvolved2 initialized successfully with capital: $", initialCapital, " and leverage: 1:", maxLeverage);
   return(INIT_SUCCEEDED);
  }
  
//+------------------------------------------------------------------+
//| Initialize Quantum Module                                        |
//+------------------------------------------------------------------+
void InitializeQuantumModule()
  {
   NUM_Q_STATES = InpQuantumStates;
   ArrayResize(QuantumState, NUM_Q_STATES);
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      QuantumState[i] = 1.0 / NUM_Q_STATES;
     }
   if(EnableDebugMode)
      Print("Quantum Module initialized with ", NUM_Q_STATES, " states.");
  }

//+------------------------------------------------------------------+
//| Initialize Schrödinger Equation Components                       |
//+------------------------------------------------------------------+
void InitializeSchrodingerEquation()
  {
   // Initialize wave function array
   ArrayResize(psi_wave, SchrodingerLookback);
   ArrayInitialize(psi_wave, 0);
   
   // Initialize potential energy array
   ArrayResize(potential_array, SchrodingerLookback);
   ArrayInitialize(potential_array, 0);
   
   // Initialize Hamiltonian matrix (square matrix of size SchrodingerLookback)
   ArrayResize(hamiltonian, SchrodingerLookback * SchrodingerLookback);
   ArrayInitialize(hamiltonian, 0);
   
   // Initialize eigenvalues and eigenvectors
   ArrayResize(eigen_values, SchrodingerLookback);
   ArrayInitialize(eigen_values, 0);
   
   ArrayResize(eigen_vectors, SchrodingerLookback * SchrodingerLookback);
   ArrayInitialize(eigen_vectors, 0);
   
   // Initialize market momentum buffer
   ArrayResize(market_momentum, SchrodingerLookback);
   ArrayInitialize(market_momentum, 0);
   
   if(EnableDebugMode)
      Print("Schrödinger equation components initialized with lookback period: ", SchrodingerLookback);
  }
  
//+------------------------------------------------------------------+
//| Evolve Quantum State based on market volatility                  |
//+------------------------------------------------------------------+
void EvolveQuantumState(double marketFactor)
  {
   // Enhanced evolution formula - more responsive to market changes
   double sum = 0.0;
   for(int i=0; i<NUM_Q_STATES; i++)
     {
      // More aggressive evolution using sigmoid function
      double sigmoid = 1.0 / (1.0 + MathExp(-marketFactor * 2.0));
      QuantumState[i] = QuantumState[i] * (0.5 + sigmoid/(i+1)) * InpQuantumDecay;
      sum += QuantumState[i];
     }
   if(sum > 0)
     {
      for(int i=0; i<NUM_Q_STATES; i++)
         QuantumState[i] /= sum;
     }
   if(EnableDebugMode)
      Print("Quantum state evolved based on market factor: ", marketFactor, " sigmoid: ", 1.0/(1.0+MathExp(-marketFactor*2.0)));
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
//| Quantum Signal Adjustment                                        |
//+------------------------------------------------------------------+
int QuantumSignalAdjustment()
  {
   int stateIndex = CollapseQuantumState();
   // More aggressive adjustment
   int adjustment = (stateIndex - (NUM_Q_STATES / 2)) * 2;  // Multiplied by 2 for stronger effect
   return adjustment;
  }
  
//+------------------------------------------------------------------+
//| Advanced ML Training: Adjust signal period based on performance  |
//+------------------------------------------------------------------+
void AdvancedTrainModel()
  {
   double oldPeriod = OptimizedSignalPeriod;
   int quantumAdjustment = QuantumSignalAdjustment();
   
   // More aggressive learning based on profit/loss
   if(lastTradeProfit > 0)
      OptimizedSignalPeriod = MathMax(1, OptimizedSignalPeriod - 2 + quantumAdjustment);  // Faster learning
   else if(lastTradeProfit < 0)
      OptimizedSignalPeriod = OptimizedSignalPeriod + 2 + quantumAdjustment;  // Faster adaptation
   else
      OptimizedSignalPeriod = OptimizedSignalPeriod + quantumAdjustment;
   
   // Keep period within reasonable bounds
   OptimizedSignalPeriod = MathMax(3, MathMin(30, OptimizedSignalPeriod));
   
   if(EnableDebugMode)
     Print("Training Model: Signal period adjusted from ", oldPeriod, " to ", OptimizedSignalPeriod, " (Quantum Adj: ", quantumAdjustment, ")");
  }
  
//+------------------------------------------------------------------+
//| Quantum Indicator: Combines MA and quantum state for threshold   |
//+------------------------------------------------------------------+
double QuantumIndicator()
  {
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   if(CopyBuffer(ma_handle, 0, 0, 3, ma_buffer) <= 0)
     {
      Print("Error copying MA data");
      return currentPrice;
     }
   double ma = ma_buffer[0];
   
   // Get RSI for additional signal
   if(CopyBuffer(rsi_handle, 0, 0, 3, rsi_buffer) <= 0)
     {
      Print("Error copying RSI data");
      return currentPrice;
     }
   double rsi = rsi_buffer[0];
   
   // Calculate quantum factor with enhanced weighting
   double quantumFactor = 0.0;
   for(int i=0; i<NUM_Q_STATES; i++)
      quantumFactor += QuantumState[i] * (i+1) * (i+1);  // Square for stronger effect
   quantumFactor /= (NUM_Q_STATES * NUM_Q_STATES);
   
   // Incorporate Schrödinger prediction
   double schrodingerPrediction = CalculateSchrodingerPrediction();
   
   // Calculate RSI bias (more buy signals when RSI is low)
   double rsiBias = 0;
   if(rsi < 40) rsiBias = (40 - rsi) * 0.0001;  // More positive bias when RSI is oversold
   
   // Combine moving average, quantum factor, Schrödinger prediction, and RSI bias
   double threshold = ma + (quantumFactor * 0.0005) + (schrodingerPrediction * 0.0005) - rsiBias;
   
   if(EnableDebugMode)
      Print("Quantum Indicator: MA=", ma, " QuantumFactor=", quantumFactor, " SchrodPred=", schrodingerPrediction, " RSI=", rsi, " Threshold=", threshold);
      
   return threshold;
  }

//+------------------------------------------------------------------+
//| Update Quantum Potential based on price movements                |
//+------------------------------------------------------------------+
void UpdateQuantumPotential()
  {
   // Copy price data
   if(CopyClose(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, close_buffer) <= 0 ||
      CopyHigh(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, high_buffer) <= 0 ||
      CopyLow(_Symbol, PERIOD_CURRENT, 0, SchrodingerLookback, low_buffer) <= 0)
     {
      Print("Error copying price data");
      return;
     }
   
   // Calculate market momentum (price velocity) with enhanced weighting
   for(int i=1; i<SchrodingerLookback; i++)
     {
      market_momentum[i-1] = (close_buffer[i] - close_buffer[i-1]) * (1.0 + 0.1*(SchrodingerLookback-i)/SchrodingerLookback);
     }
   market_momentum[SchrodingerLookback-1] = 0;
   
   // Create the potential energy landscape based on price action
   for(int i=0; i<SchrodingerLookback; i++)
     {
      // Higher potential for price range (creates barriers)
      double range = high_buffer[i] - low_buffer[i];
      // Add potential wells at significant price levels
      double wellDepth = (range / point) * QuantumPotentialScaling;
      
      // Enhanced potential energy calculation
      potential_array[i] = wellDepth * PotentialWellDepth - (MathPow(MathAbs(market_momentum[i]), 1.5) * QuantumPotentialScaling);
     }
     
   if(EnableDebugMode)
      Print("Quantum potential updated with enhanced weighting");
  }

//+------------------------------------------------------------------+
//| Construct Hamiltonian Operator for Schrödinger Equation          |
//+------------------------------------------------------------------+
void ConstructHamiltonian()
  {
   // The Hamiltonian is H = T + V where T is kinetic energy operator and V is potential energy
   // In discrete form, we use finite difference method for second derivatives (kinetic energy)
   
   // Reset Hamiltonian
   ArrayInitialize(hamiltonian, 0);
   
   // Constant for kinetic energy term (ħ²/2m)
   double hbar_squared_over_2m = MathPow(PlankConstant, 2) / (2.0 * WaveFunctionDiffusion);
   
   // Construct the matrix (store as flattened 1D array)
   for(int i=0; i<SchrodingerLookback; i++)
     {
      // Diagonal terms: potential energy + kinetic energy diagonal term
      int idx = i * SchrodingerLookback + i;
      hamiltonian[idx] = potential_array[i] + (2.0 * hbar_squared_over_2m);
      
      // Off-diagonal terms: kinetic energy connections to neighbors
      if(i > 0)
        {
         hamiltonian[i * SchrodingerLookback + (i-1)] = -hbar_squared_over_2m;
        }
      if(i < SchrodingerLookback - 1)
        {
         hamiltonian[i * SchrodingerLookback + (i+1)] = -hbar_squared_over_2m;
        }
     }
     
   if(EnableDebugMode)
      Print("Hamiltonian constructed with enhanced parameters");
  }

//+------------------------------------------------------------------+
//| Solve Time-Independent Schrödinger Equation (enhanced version)   |
//+------------------------------------------------------------------+
void SolveSchrodingerEquation()
  {
   // Update the potential energy landscape
   UpdateQuantumPotential();
   
   // Construct the Hamiltonian operator
   ConstructHamiltonian();
   
   // Enhanced approach: use iterative method with more steps
   
   // Initialize wave function - use previous market momentum for non-random initialization
   double norm_factor = 0.0;
   for(int i=0; i<SchrodingerLookback; i++)
     {
      // Initialize to a combination of previous state and market momentum
      if(MathAbs(psi_wave[i]) < 1e-10)  // First run or reset state
         psi_wave[i] = MathExp(-0.5 * MathPow(i - SchrodingerLookback/2, 2) / (SchrodingerLookback/5.0));
      else
         psi_wave[i] = 0.8 * psi_wave[i] + 0.2 * (0.5 + 0.5*MathTanh(market_momentum[i] * 100.0));
         
      norm_factor += MathPow(psi_wave[i], 2);
     }
   
   // Normalize wave function
   norm_factor = MathSqrt(norm_factor);
   if(norm_factor > 0)
     {
      for(int i=0; i<SchrodingerLookback; i++)
        {
         psi_wave[i] /= norm_factor;
        }
     }
   
   // Time evolution (enhanced approximation)
   double temp_psi[];
   ArrayResize(temp_psi, SchrodingerLookback);
   ArrayCopy(temp_psi, psi_wave);
   
   // Evolve using simplified time-evolution operator - more iterations
   for(int t=0; t<10; t++) // Increased from 5 to 10 iterations
     {
      for(int i=0; i<SchrodingerLookback; i++)
        {
         // Apply Hamiltonian to wave function
         double H_psi = 0.0;
         for(int j=0; j<SchrodingerLookback; j++)
           {
            H_psi += hamiltonian[i * SchrodingerLookback + j] * psi_wave[j];
           }
         // Update wave function (enhanced evolution)
         temp_psi[i] = psi_wave[i] - H_psi * 0.02; // Double the time step for faster evolution
        }
      // Copy back to psi
      ArrayCopy(psi_wave, temp_psi);
      
      // Re-normalize periodically
      if(t % 3 == 0)
        {
         norm_factor = 0.0;
         for(int i=0; i<SchrodingerLookback; i++)
           {
            norm_factor += MathPow(psi_wave[i], 2);
           }
         norm_factor = MathSqrt(norm_factor);
         if(norm_factor > 0)
           {
            for(int i=0; i<SchrodingerLookback; i++)
              {
               psi_wave[i] /= norm_factor;
              }
           }
        }
     }
     
   if(EnableDebugMode)
      Print("Enhanced Schrödinger equation solved with 10 iterations");
  }

//+------------------------------------------------------------------+
//| Calculate market prediction based on Schrödinger wave function   |
//+------------------------------------------------------------------+
double CalculateSchrodingerPrediction()
  {
   // Solve the Schrödinger equation to update wave function
   SolveSchrodingerEquation();
   
   // Calculate expected value of position (momentum direction)
   double expectation_position = 0.0;
   for(int i=0; i<SchrodingerLookback; i++)
     {
      expectation_position += i * MathPow(psi_wave[i], 2);
     }
   
   // Calculate expected value of momentum - enhanced calculation
   double expectation_momentum = 0.0;
   for(int i=1; i<SchrodingerLookback-1; i++)
     {
      // Improved finite difference for momentum operator
      double momentum_i = (psi_wave[i+1] - psi_wave[i-1]) / 2.0;
      // Apply weighting that emphasizes recent data
      double recency_weight = 1.0 + 0.5 * ((double)(SchrodingerLookback-i) / SchrodingerLookback);
      expectation_momentum += momentum_i * psi_wave[i] * recency_weight;
     }
   
   // Calculate wave function concentration (uncertainty measure)
   double uncertainty = 0.0;
   double mean_position = expectation_position;
   for(int i=0; i<SchrodingerLookback; i++)
     {
      uncertainty += MathPow(i - mean_position, 2) * MathPow(psi_wave[i], 2);
     }
   uncertainty = MathSqrt(uncertainty);
   
   // Combine to get prediction with uncertainty weighting
   double certainty_factor = 1.0 / (1.0 + uncertainty / 10.0);  // Higher when more certain
   
   // Scale prediction based on certainty (more aggressive when more certain)
   double prediction = expectation_momentum * 150.0 * certainty_factor;
   
   if(EnableDebugMode)
      Print("Enhanced Schrödinger prediction: ", prediction, " (Certainty: ", certainty_factor, ")");
      
   return prediction;
  }
  
//+------------------------------------------------------------------+
//| Calculate trade costs including spread, commission, and swap     |
//+------------------------------------------------------------------+
double CalculateTradeCosts(double lotSize, ENUM_ORDER_TYPE orderType, int daysHeld = 1)
  {
   double commission = CommissionPerLot * lotSize;
   double spread = SpreadAverage * pip_value * (SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN) / lotSize);
   
   double swap = 0.0;
   if(orderType == ORDER_TYPE_BUY)
      swap = SwapLong * daysHeld * lotSize;
   else
      swap = SwapShort * daysHeld * lotSize;
   
   return(commission + spread + MathAbs(swap));
  }

//+------------------------------------------------------------------+
//| Calculate potential profit considering all fees                  |
//+------------------------------------------------------------------+
double CalculatePotentialProfit(double entryPrice, double targetPrice, double lotSize, ENUM_ORDER_TYPE orderType)
  {
   // Calculate raw profit in price difference
   double priceDiff = 0;
   if(orderType == ORDER_TYPE_BUY)
      priceDiff = targetPrice - entryPrice;
   else
      priceDiff = entryPrice - targetPrice;
   
   // Convert to account currency
   double rawProfit = priceDiff * lotSize * SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE) / SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   // Subtract trading costs
   double costs = CalculateTradeCosts(lotSize, orderType);
   double netProfit = rawProfit - costs;
   
   return netProfit;
  }

//+------------------------------------------------------------------+
//| Check Advanced Trade Entry using Quantum-Enhanced Indicator      |
//+------------------------------------------------------------------+
bool CheckTradeEntryAdvanced()
  {
   double currentPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double threshold = QuantumIndicator();
   
   // Get prediction from Schrödinger equation
   double prediction = CalculateSchrodingerPrediction();
   
   // Calculate potential profit including all costs
   double lotSize = InpInitialLotSize;
   double targetPrice = currentPrice + (prediction * pip_value);
   
   // For long position
   double potentialProfit = CalculatePotentialProfit(currentPrice, targetPrice, lotSize, ORDER_TYPE_BUY);
   
   // Less restrictive condition: Now we'll consider trading even with small predicted profit
   // And we'll trade if price is near threshold rather than just above
   bool priceCondition = (currentPrice > threshold * 0.998); // Allow 0.2% flexibility
   bool predictionCondition = (prediction > -0.5); // Allow slightly negative predictions
   bool profitCondition = (potentialProfit > -0.5); // Allow very small losses if other indicators are strong
   
   // For debugging
   if(EnableDebugMode)
     {
      Print("Trade Check - Price: ", currentPrice, " Threshold: ", threshold, " Prediction: ", prediction, 
            " PotentialProfit: ", potentialProfit);
      Print("Conditions - Price: ", priceCondition, " Prediction: ", predictionCondition, " Profit: ", profitCondition);
     }
   
   // Relaxed conditions for more trading activity
   if(priceCondition && (predictionCondition || profitCondition))
      return true;
      
   return false;
  }

//+------------------------------------------------------------------+
//| Calculate position size based on risk management                 |
//+------------------------------------------------------------------+
double CalculatePositionSize(double stopLossPips)
  {
   if(stopLossPips <= 0) return InpInitialLotSize;
   
   double balance = AccountInfoDouble(ACCOUNT_BALANCE);
   double riskAmount = balance * (RiskPercent / 100.0);
   
   double tickValue = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   double tickSize = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   
   // Calculate pip value in account currency
   double pipValue = (pip_value / tickSize) * tickValue;
   
   // Calculate position size based on risk
   double positionSize = riskAmount / (stopLossPips * pipValue);
   
   // Round to standard lot size
   positionSize = MathFloor(positionSize * 100) / 100;
   
   // Apply minimum lot size
   double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   if(positionSize < minLot)
      positionSize = minLot;
      
   // Apply maximum lot size
   double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   if(positionSize > maxLot)
      positionSize = maxLot;
      
   return positionSize;
  }
  
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   // Update market volatility from ATR
   if(CopyBuffer(atr_handle, 0, 0, 3, atr_buffer) <= 0)
     {
      Print("Error copying ATR data");
      return;
     }
   double marketVolatility = atr_buffer[0];
   
   // Evolve quantum state based on market volatility
   EvolveQuantumState(marketVolatility);
   
   // Check for trade entry signal with enhanced Schrödinger prediction
   if(CheckTradeEntryAdvanced())
     {
      if(PositionsTotal() < MaxOpenPositions)
        {
         // Get prediction from Schrödinger equation
         double prediction = CalculateSchrodingerPrediction();
         
         // Calculate lot size based on risk management
         double lotSize = CalculatePositionSize(QuantumSL);
         
         // Get current price
         double bidPrice = SymbolInfoDouble(_Symbol, SYMBOL_BID);
         double askPrice = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         
         // Adjust take profit based on prediction (dynamic)
         double predictionFactor = MathMax(0.5, MathMin(2.0, 1.0 + prediction / 50.0));
         double adjustedTP = QuantumTP * predictionFactor;
         
         // Calculate potential profit including all costs
         double targetPrice = bidPrice + (adjustedTP * pip_value);
         double potentialProfit = CalculatePotentialProfit(bidPrice, targetPrice, lotSize, ORDER_TYPE_BUY);
         
         // Slightly relaxed profit condition
         if(potentialProfit > -1.0 || prediction > 2.0)  // Allow small negative expected profit if prediction is strong
           {
            // Place a buy order
            if(trade.Buy(lotSize, _Symbol, askPrice, askPrice - QuantumSL*pip_value, 
                          askPrice + adjustedTP*pip_value, "EinsteinEvolved2 Buy Order"))
              {
               Print("Buy order executed at price ", askPrice, " lot size: ", lotSize);
               Print("Adjusted TP: ", adjustedTP, " pips, Potential profit: ", DoubleToString(potentialProfit, 2));
               tradeCounter++;
               
               // Store last trade info for machine learning
               lastTradeProfit = 0; // Will be updated when trade closes
              }
            else
              {
               Print("Buy order failed. Error: ", GetLastError());
              }
           }
         else
           {
            if(EnableDebugMode)
               Print("Trade skipped - insufficient potential profit: ", DoubleToString(potentialProfit, 2));
           }
        }
      else
        {
         if(EnableDebugMode)
            Print("Max open positions (", MaxOpenPositions, ") reached, skipping trade");
        }
     }
     
   // Call training model after the specified number of trades
   if(tradeCounter > 0 && tradeCounter % InpTrainingFrequency == 0)
      AdvancedTrainModel();
  }
  
//+------------------------------------------------------------------+
//| Track closed positions to update last trade profit               |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                       const MqlTradeRequest& request,
                       const MqlTradeResult& result)
  {
   // Check for position closing
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD && trans.deal_type == DEAL_TYPE_SELL)
     {
      // Position might have been closed, update lastTradeProfit
      HistorySelect(TimeCurrent()-86400, TimeCurrent()); // Last 24 hours
      
      for(int i=HistoryDealsTotal()-1; i>=0; i--)
        {
         ulong ticket = HistoryDealGetTicket(i);
         if(ticket > 0)
           {
            // Check if this deal closed a position with our magic number
            if(HistoryDealGetInteger(ticket, DEAL_MAGIC) == trade.RequestMagic() && 
               HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT)
              {
               lastTradeProfit = HistoryDealGetDouble(ticket, DEAL_PROFIT);
               
               if(EnableDebugMode)
                  Print("Last trade closed with profit: ", lastTradeProfit);
                  
               break;
              }
           }
        }
     }
  }
  
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(atr_handle != INVALID_HANDLE) IndicatorRelease(atr_handle);
   if(rsi_handle != INVALID_HANDLE) IndicatorRelease(rsi_handle);
   if(ma_handle  != INVALID_HANDLE) IndicatorRelease(ma_handle);
   Print("EinsteinEvolved2 deinitialized.");
  } 