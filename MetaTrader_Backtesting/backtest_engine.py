import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
import pytz
from data_processor import MetaTraderDataProcessor

class BacktestEngine:
    """
    Engine for backtesting strategies using backtrader
    """
    
    def __init__(self, data_dir='data', output_dir='results'):
        """
        Initialize the backtest engine
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory containing the data files (default: 'data')
        output_dir : str, optional
            Directory to save the results (default: 'results')
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.cerebro = bt.Cerebro()
        self.data_processor = MetaTraderDataProcessor(data_dir)
        
        # Create directories if they don't exist
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Default configuration
        self.cerebro.broker.setcash(10000.0)  # Set initial cash
        self.cerebro.broker.setcommission(commission=0.0001)  # 0.01% commission
        
        # Set default analyzers
        self.add_default_analyzers()
    
    def add_default_analyzers(self):
        """
        Add default analyzers to cerebro
        """
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
        self.cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
        self.cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
        self.cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')
        self.cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    
    def set_cash(self, cash):
        """
        Set initial cash amount
        
        Parameters:
        -----------
        cash : float
            Initial cash amount
        """
        self.cerebro.broker.setcash(cash)
    
    def set_commission(self, commission):
        """
        Set commission rate
        
        Parameters:
        -----------
        commission : float
            Commission rate
        """
        self.cerebro.broker.setcommission(commission=commission)
    
    def add_strategy(self, strategy, **kwargs):
        """
        Add strategy to cerebro
        
        Parameters:
        -----------
        strategy : backtrader.Strategy
            Strategy class
        **kwargs : dict
            Strategy parameters
        """
        self.cerebro.addstrategy(strategy, **kwargs)
    
    def add_data_from_csv(self, filename, name=None, timeframe='1D', start_date=None, end_date=None):
        """
        Add data from a CSV file
        
        Parameters:
        -----------
        filename : str
            CSV filename
        name : str, optional
            Data name
        timeframe : str, optional
            Data timeframe (default: '1D')
        start_date : datetime.datetime, optional
            Start date for data
        end_date : datetime.datetime, optional
            End date for data
        """
        # Process data using the data processor
        df = self.data_processor.load_mt4_csv(filename, timeframe)
        
        # Filter data by date if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Convert to backtrader format
        df = self.data_processor.prepare_backtest_data(df)
        
        # Save processed data
        processed_filename = f'processed_{filename}'
        self.data_processor.save_as_csv(df, processed_filename)
        
        # Create a data feed from the processed data
        data = bt.feeds.PandasData(
            dataname=df,
            name=name if name else filename.replace('.csv', ''),
            timeframe=self.get_bt_timeframe(timeframe),
            fromdate=start_date,
            todate=end_date
        )
        
        self.cerebro.adddata(data)
    
    def get_bt_timeframe(self, timeframe):
        """
        Convert string timeframe to backtrader timeframe
        
        Parameters:
        -----------
        timeframe : str
            Timeframe string (e.g., '1M', '1D', '4H', '1H')
            
        Returns:
        --------
        backtrader.TimeFrame
            Backtrader timeframe
        """
        timeframe_map = {
            '1M': bt.TimeFrame.Months,
            '1W': bt.TimeFrame.Weeks,
            '1D': bt.TimeFrame.Days,
            '4H': bt.TimeFrame.Minutes,
            '1H': bt.TimeFrame.Minutes,
            '30M': bt.TimeFrame.Minutes,
            '15M': bt.TimeFrame.Minutes,
            '5M': bt.TimeFrame.Minutes,
            '1M': bt.TimeFrame.Minutes
        }
        
        return timeframe_map.get(timeframe, bt.TimeFrame.Days)
    
    def run_backtest(self, plot=True):
        """
        Run the backtest
        
        Parameters:
        -----------
        plot : bool, optional
            Whether to plot the results (default: True)
            
        Returns:
        --------
        dict
            Backtest results
        """
        # Run the backtest
        results = self.cerebro.run()
        
        # Print final portfolio value
        final_value = self.cerebro.broker.getvalue()
        initial_value = self.cerebro.broker.startingcash
        profit_pct = (final_value - initial_value) / initial_value * 100
        
        print(f"Initial Portfolio Value: ${initial_value:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Profit/Loss: {profit_pct:.2f}%")
        
        # Process analyzer results
        backtest_results = {}
        strat = results[0]
        
        # Sharpe ratio
        sharpe = strat.analyzers.sharpe.get_analysis()
        backtest_results['sharpe_ratio'] = sharpe.get('sharperatio', 0.0)
        
        # Drawdown
        drawdown = strat.analyzers.drawdown.get_analysis()
        backtest_results['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0.0)
        
        # Trade analysis
        trades = strat.analyzers.trades.get_analysis()
        backtest_results['total_trades'] = trades.get('total', {}).get('total', 0)
        backtest_results['total_winning'] = trades.get('won', {}).get('total', 0)
        backtest_results['total_losing'] = trades.get('lost', {}).get('total', 0)
        
        # SQN
        sqn = strat.analyzers.sqn.get_analysis()
        backtest_results['sqn'] = sqn.get('sqn', 0.0)
        
        # Returns
        returns = strat.analyzers.returns.get_analysis()
        backtest_results['annual_return'] = returns.get('ravg', 0.0) * 252  # Assuming 252 trading days
        
        # Calculate win rate
        if backtest_results['total_trades'] > 0:
            backtest_results['win_rate'] = backtest_results['total_winning'] / backtest_results['total_trades'] * 100
        else:
            backtest_results['win_rate'] = 0.0
        
        # Print results
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2f}%")
        print(f"Total Trades: {backtest_results['total_trades']}")
        print(f"Win Rate: {backtest_results['win_rate']:.2f}%")
        print(f"SQN: {backtest_results['sqn']:.2f}")
        print(f"Annual Return: {backtest_results['annual_return']:.2f}%")
        
        # Plot if requested
        if plot:
            plt.figure(figsize=(12, 8))
            self.cerebro.plot(style='candle', barup='green', bardown='red')
            plt.savefig(os.path.join(self.output_dir, 'backtest_plot.png'))
        
        return backtest_results
    
    def forward_test(self, strategy, train_start, train_end, test_start, test_end, data_filename):
        """
        Perform a forward test
        
        Parameters:
        -----------
        strategy : backtrader.Strategy
            Strategy class
        train_start : datetime.datetime
            Start date for training period
        train_end : datetime.datetime
            End date for training period
        test_start : datetime.datetime
            Start date for testing period
        test_end : datetime.datetime
            End date for testing period
        data_filename : str
            CSV filename containing the data
            
        Returns:
        --------
        tuple
            (training_results, testing_results)
        """
        # Create a fresh cerebro instance for training
        train_cerebro = bt.Cerebro()
        train_cerebro.broker.setcash(10000.0)
        train_cerebro.broker.setcommission(commission=0.0001)
        self.add_default_analyzers()
        
        # Add data for training period
        df = self.data_processor.load_mt4_csv(data_filename)
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        train_df = self.data_processor.prepare_backtest_data(train_df)
        
        train_data = bt.feeds.PandasData(
            dataname=train_df,
            name='train_data',
            fromdate=train_start,
            todate=train_end
        )
        
        train_cerebro.adddata(train_data)
        train_cerebro.addstrategy(strategy)
        
        # Run training backtest
        print("Running training backtest...")
        train_results = train_cerebro.run()
        
        # Get optimized parameters if the strategy supports it
        optimized_params = {}
        try:
            optimized_params = train_results[0].get_optimized_params()
        except:
            # If strategy doesn't support optimized params, use defaults
            pass
        
        # Now run the forward test with optimized parameters
        test_cerebro = bt.Cerebro()
        test_cerebro.broker.setcash(10000.0)
        test_cerebro.broker.setcommission(commission=0.0001)
        self.add_default_analyzers()
        
        # Add data for testing period
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]
        test_df = self.data_processor.prepare_backtest_data(test_df)
        
        test_data = bt.feeds.PandasData(
            dataname=test_df,
            name='test_data',
            fromdate=test_start,
            todate=test_end
        )
        
        test_cerebro.adddata(test_data)
        test_cerebro.addstrategy(strategy, **optimized_params)
        
        # Run testing backtest
        print("Running forward test...")
        test_results = test_cerebro.run()
        
        # Process and return results
        return self._process_results(train_results), self._process_results(test_results)
    
    def _process_results(self, results):
        """
        Process backtest results
        
        Parameters:
        -----------
        results : list
            Results from cerebro.run()
            
        Returns:
        --------
        dict
            Processed results
        """
        processed_results = {}
        strat = results[0]
        
        # Sharpe ratio
        sharpe = strat.analyzers.sharpe.get_analysis()
        processed_results['sharpe_ratio'] = sharpe.get('sharperatio', 0.0)
        
        # Drawdown
        drawdown = strat.analyzers.drawdown.get_analysis()
        processed_results['max_drawdown'] = drawdown.get('max', {}).get('drawdown', 0.0)
        
        # Trade analysis
        trades = strat.analyzers.trades.get_analysis()
        processed_results['total_trades'] = trades.get('total', {}).get('total', 0)
        processed_results['total_winning'] = trades.get('won', {}).get('total', 0)
        processed_results['total_losing'] = trades.get('lost', {}).get('total', 0)
        
        # SQN
        sqn = strat.analyzers.sqn.get_analysis()
        processed_results['sqn'] = sqn.get('sqn', 0.0)
        
        # Returns
        returns = strat.analyzers.returns.get_analysis()
        processed_results['annual_return'] = returns.get('ravg', 0.0) * 252  # Assuming 252 trading days
        
        # Calculate win rate
        if processed_results['total_trades'] > 0:
            processed_results['win_rate'] = processed_results['total_winning'] / processed_results['total_trades'] * 100
        else:
            processed_results['win_rate'] = 0.0
        
        return processed_results 