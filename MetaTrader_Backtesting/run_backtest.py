import os
import sys
import datetime
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from backtest_engine import BacktestEngine
from data_processor import MetaTraderDataProcessor
from strategies import MovingAverageCrossover, RSIMeanReversion, BollingerBreakout

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Run backtest for trading strategies')
    
    parser.add_argument('--data-file', type=str, required=True,
                       help='CSV file containing OHLCV data')
    
    parser.add_argument('--strategy', type=str, default='ma_crossover',
                       choices=['ma_crossover', 'rsi_mean_reversion', 'bollinger_breakout'],
                       help='Trading strategy to backtest')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD)')
    
    parser.add_argument('--timeframe', type=str, default='1D',
                       help='Timeframe for backtest (e.g., 1D, 4H, 1H)')
    
    parser.add_argument('--cash', type=float, default=10000.0,
                       help='Initial cash amount')
    
    parser.add_argument('--commission', type=float, default=0.0001,
                       help='Commission rate')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory to save results')
    
    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    
    parser.add_argument('--forward-test', action='store_true',
                       help='Perform forward testing')
    
    parser.add_argument('--forward-ratio', type=float, default=0.25,
                       help='Ratio of data to use for forward testing (default: 0.25 = 25%%)')
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create backtest engine
    engine = BacktestEngine(data_dir=args.data_dir, output_dir=args.output_dir)
    
    # Set cash and commission
    engine.set_cash(args.cash)
    engine.set_commission(args.commission)
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Select strategy
    strategy_map = {
        'ma_crossover': MovingAverageCrossover,
        'rsi_mean_reversion': RSIMeanReversion,
        'bollinger_breakout': BollingerBreakout
    }
    
    strategy = strategy_map[args.strategy]
    
    # Perform forward test if requested
    if args.forward_test:
        # Load data to determine date range
        data_processor = MetaTraderDataProcessor(args.data_dir)
        
        try:
            df = data_processor.load_mt4_csv(args.data_file)
        except:
            try:
                df = data_processor.load_mt5_csv(args.data_file)
            except Exception as e:
                print(f"Error loading data file: {e}")
                sys.exit(1)
        
        # Apply date filters if specified
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Split data into training and testing periods
        split_idx = int(len(df) * (1 - args.forward_ratio))
        
        train_start = df.index[0]
        train_end = df.index[split_idx]
        test_start = df.index[split_idx + 1]
        test_end = df.index[-1]
        
        print(f"Training period: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"Testing period: {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")
        
        # Run forward test
        train_results, test_results = engine.forward_test(
            strategy=strategy,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            data_filename=args.data_file
        )
        
        # Print results
        print("\nTraining Results:")
        print(f"Sharpe Ratio: {train_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {train_results['max_drawdown']:.2f}%")
        print(f"Total Trades: {train_results['total_trades']}")
        print(f"Win Rate: {train_results['win_rate']:.2f}%")
        print(f"SQN: {train_results['sqn']:.2f}")
        print(f"Annual Return: {train_results['annual_return']:.2f}%")
        
        print("\nForward Test Results:")
        print(f"Sharpe Ratio: {test_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {test_results['max_drawdown']:.2f}%")
        print(f"Total Trades: {test_results['total_trades']}")
        print(f"Win Rate: {test_results['win_rate']:.2f}%")
        print(f"SQN: {test_results['sqn']:.2f}")
        print(f"Annual Return: {test_results['annual_return']:.2f}%")
        
    else:
        # Regular backtest
        engine.add_data_from_csv(
            filename=args.data_file,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        # Add strategy
        engine.add_strategy(strategy)
        
        # Run backtest
        results = engine.run_backtest(plot=not args.no_plot)
        
        # Save results to CSV
        result_df = pd.DataFrame([results])
        result_df.to_csv(os.path.join(args.output_dir, 'backtest_results.csv'), index=False)
        
        print(f"Results saved to {os.path.join(args.output_dir, 'backtest_results.csv')}")

if __name__ == '__main__':
    main() 