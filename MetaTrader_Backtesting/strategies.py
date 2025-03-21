import backtrader as bt
import numpy as np

class MovingAverageCrossover(bt.Strategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates buy signals when the fast moving average crosses above
    the slow moving average, and sell signals when the fast moving average crosses
    below the slow moving average.
    """
    
    params = (
        ('fast_period', 10),  # Fast moving average period
        ('slow_period', 30),  # Slow moving average period
        ('order_pct', 0.95),  # Order percentage of portfolio
        ('stop_loss_pct', 0.02),  # Stop loss percentage
        ('take_profit_pct', 0.03),  # Take profit percentage
    )
    
    def __init__(self):
        """
        Initialize the strategy
        """
        # Initialize moving averages
        self.fast_ma = bt.indicators.SMA(self.data.close, period=self.params.fast_period)
        self.slow_ma = bt.indicators.SMA(self.data.close, period=self.params.slow_period)
        
        # Initialize crossover indicator
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
        
        # Keep track of open orders
        self.orders = {}
        
        # Keep track of whether we're in the market
        self.in_market = False
    
    def log(self, txt, dt=None):
        """
        Log strategy information
        
        Parameters:
        -----------
        txt : str
            Text to log
        dt : datetime.datetime, optional
            Datetime of the log entry (default: None)
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')
    
    def notify_order(self, order):
        """
        Notification of order status change
        
        Parameters:
        -----------
        order : backtrader.Order
            Order instance
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return
        
        # Check if order is completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
                
                # Set stop loss and take profit orders
                stop_price = self.buyprice * (1.0 - self.params.stop_loss_pct)
                target_price = self.buyprice * (1.0 + self.params.take_profit_pct)
                
                # Submit stop loss order
                self.orders['stop'] = self.sell(exectype=bt.Order.Stop, price=stop_price)
                
                # Submit take profit order
                self.orders['target'] = self.sell(exectype=bt.Order.Limit, price=target_price)
                
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Cancel any existing orders
                for name, o in list(self.orders.items()):
                    if o is not order:
                        self.cancel(o)
                        del self.orders[name]
                
                # Reset orders dictionary
                self.orders = {}
                
                # Update market status
                self.in_market = False
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected')
            
            # If this order was in our dictionary, remove it
            for name, o in list(self.orders.items()):
                if o is order:
                    del self.orders[name]
    
    def notify_trade(self, trade):
        """
        Notification of trade status change
        
        Parameters:
        -----------
        trade : backtrader.Trade
            Trade instance
        """
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """
        Next bar strategy logic
        """
        # Skip if we have pending orders
        if self.orders:
            return
        
        # Check if we're in the market
        if not self.position:
            # We are not in the market
            self.in_market = False
            
            # Buy signal: crossover > 0
            if self.crossover > 0:
                self.log(f'BUY CREATE, {self.data.close[0]:.5f}')
                
                # Calculate position size based on order percentage
                cash = self.broker.getcash()
                value = self.broker.getvalue()
                size = self.params.order_pct * value / self.data.close[0]
                
                # Create buy order
                self.orders['main'] = self.buy(size=size)
                self.in_market = True
        
        # Note: We don't need to handle sell signals here because they are managed
        # by stop loss and take profit orders.


class RSIMeanReversion(bt.Strategy):
    """
    RSI Mean Reversion Strategy
    
    This strategy buys when the RSI is oversold and sells when the RSI is overbought.
    It also uses ATR for position sizing.
    """
    
    params = (
        ('rsi_period', 14),  # RSI period
        ('rsi_overbought', 70),  # RSI overbought level
        ('rsi_oversold', 30),  # RSI oversold level
        ('atr_period', 14),  # ATR period
        ('risk_pct', 0.02),  # Risk percentage per trade
        ('stop_loss_atr', 2.0),  # Stop loss in ATR units
        ('take_profit_atr', 3.0),  # Take profit in ATR units
    )
    
    def __init__(self):
        """
        Initialize the strategy
        """
        # Initialize indicators
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)
        self.atr = bt.indicators.ATR(self.data, period=self.params.atr_period)
        
        # Keep track of open orders
        self.orders = {}
        
        # Initialize variable to store price when entering trade
        self.trade_entry_price = None
    
    def log(self, txt, dt=None):
        """
        Log strategy information
        
        Parameters:
        -----------
        txt : str
            Text to log
        dt : datetime.datetime, optional
            Datetime of the log entry (default: None)
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')
    
    def notify_order(self, order):
        """
        Notification of order status change
        
        Parameters:
        -----------
        order : backtrader.Order
            Order instance
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return
        
        # Check if order is completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                self.trade_entry_price = order.executed.price
                
                # Set stop loss and take profit levels
                stop_price = self.trade_entry_price - self.params.stop_loss_atr * self.atr[0]
                target_price = self.trade_entry_price + self.params.take_profit_atr * self.atr[0]
                
                # Submit stop loss order
                self.orders['stop'] = self.sell(exectype=bt.Order.Stop, price=stop_price, size=order.executed.size)
                
                # Submit take profit order
                self.orders['target'] = self.sell(exectype=bt.Order.Limit, price=target_price, size=order.executed.size)
                
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Size: {order.executed.size:.2f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Cancel any existing orders if we sold everything
                if not self.position:
                    for name, o in list(self.orders.items()):
                        if o is not order:
                            self.cancel(o)
                            del self.orders[name]
                    
                    # Reset orders dictionary
                    self.orders = {}
                    
                    # Reset trade entry price
                    self.trade_entry_price = None
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected')
            
            # If this order was in our dictionary, remove it
            for name, o in list(self.orders.items()):
                if o is order:
                    del self.orders[name]
    
    def notify_trade(self, trade):
        """
        Notification of trade status change
        
        Parameters:
        -----------
        trade : backtrader.Trade
            Trade instance
        """
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """
        Next bar strategy logic
        """
        # Skip if we have pending entry orders
        if 'main' in self.orders:
            return
        
        # Check if we're in the market
        if not self.position:
            # We are not in the market
            
            # Buy signal: RSI is oversold
            if self.rsi[0] < self.params.rsi_oversold:
                # Calculate position size based on risk
                risk_amount = self.broker.getvalue() * self.params.risk_pct
                stop_loss_price = self.data.close[0] - self.params.stop_loss_atr * self.atr[0]
                risk_per_share = self.data.close[0] - stop_loss_price
                
                if risk_per_share > 0:
                    size = risk_amount / risk_per_share
                    
                    self.log(f'BUY CREATE, Price: {self.data.close[0]:.5f}, Size: {size:.2f}')
                    
                    # Create buy order
                    self.orders['main'] = self.buy(size=size)
        
        else:
            # We are in the market
            
            # Sell signal: RSI is overbought (only if not already managed by stop/target)
            if self.rsi[0] > self.params.rsi_overbought and not self.orders:
                self.log(f'SELL CREATE (OVERBOUGHT), Price: {self.data.close[0]:.5f}')
                
                # Create sell order for current position
                self.orders['main'] = self.sell(size=self.position.size)


class BollingerBreakout(bt.Strategy):
    """
    Bollinger Bands Breakout Strategy
    
    This strategy buys when price breaks above the upper Bollinger Band
    and sells when price breaks below the lower Bollinger Band.
    It uses a trailing stop for exit.
    """
    
    params = (
        ('bb_period', 20),  # Bollinger Bands period
        ('bb_dev', 2.0),  # Bollinger Bands standard deviation
        ('order_pct', 0.95),  # Order percentage of portfolio
        ('trail_percent', 0.05),  # Trailing stop percentage
    )
    
    def __init__(self):
        """
        Initialize the strategy
        """
        # Initialize Bollinger Bands
        self.bbands = bt.indicators.BollingerBands(
            self.data.close, 
            period=self.params.bb_period,
            devfactor=self.params.bb_dev
        )
        
        # Keep track of open orders
        self.orders = {}
        
        # Store price direction for trailing stop
        self.long_position = False
        self.short_position = False
    
    def log(self, txt, dt=None):
        """
        Log strategy information
        
        Parameters:
        -----------
        txt : str
            Text to log
        dt : datetime.datetime, optional
            Datetime of the log entry (default: None)
        """
        dt = dt or self.datas[0].datetime.date(0)
        print(f'[{dt.isoformat()}] {txt}')
    
    def notify_order(self, order):
        """
        Notification of order status change
        
        Parameters:
        -----------
        order : backtrader.Order
            Order instance
        """
        if order.status in [order.Submitted, order.Accepted]:
            # Order submitted/accepted - nothing to do
            return
        
        # Check if order is completed
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'BUY EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Set trailing stop
                self.orders['stop'] = self.sell(
                    exectype=bt.Order.StopTrail,
                    trailpercent=self.params.trail_percent
                )
                
                # Update position status
                self.long_position = True
                self.short_position = False
                
            else:  # Sell
                self.log(f'SELL EXECUTED, Price: {order.executed.price:.5f}, Cost: {order.executed.value:.2f}, Comm: {order.executed.comm:.2f}')
                
                # Cancel any existing orders
                for name, o in list(self.orders.items()):
                    if o is not order:
                        self.cancel(o)
                        del self.orders[name]
                
                # Reset orders dictionary
                self.orders = {}
                
                # Update position status
                self.long_position = False
                self.short_position = False
        
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected')
            
            # If this order was in our dictionary, remove it
            for name, o in list(self.orders.items()):
                if o is order:
                    del self.orders[name]
    
    def notify_trade(self, trade):
        """
        Notification of trade status change
        
        Parameters:
        -----------
        trade : backtrader.Trade
            Trade instance
        """
        if not trade.isclosed:
            return
        
        self.log(f'OPERATION PROFIT, GROSS: {trade.pnl:.2f}, NET: {trade.pnlcomm:.2f}')
    
    def next(self):
        """
        Next bar strategy logic
        """
        # Skip if we have pending orders
        if 'main' in self.orders:
            return
        
        # Check if we're in the market
        if not self.position:
            # We are not in the market
            
            # Buy signal: Close price crosses above upper Bollinger Band
            if self.data.close[0] > self.bbands.lines.top[0] and self.data.close[-1] <= self.bbands.lines.top[-1]:
                self.log(f'BUY CREATE (BREAKOUT), Price: {self.data.close[0]:.5f}')
                
                # Calculate position size based on order percentage
                cash = self.broker.getcash()
                value = self.broker.getvalue()
                size = self.params.order_pct * value / self.data.close[0]
                
                # Create buy order
                self.orders['main'] = self.buy(size=size)
        
        # Note: Selling is handled by the trailing stop 