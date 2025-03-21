import os
import datetime
import argparse
from mt5_data_downloader import MT5DataDownloader

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Download and process tick data from MetaTrader 5')
    
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Symbol to download (default: EURUSD)')
    
    parser.add_argument('--timeframe', type=str, default='M1',
                       help='Timeframe to download (default: M1, for tick data use "TICK")')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date for data download (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for data download (YYYY-MM-DD)')
    
    parser.add_argument('--csv-path', type=str, default=None,
                       help='Path to already exported CSV file')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to save data files')
    
    # For direct downloading (if supported on your platform)
    parser.add_argument('--login', type=int, default=None,
                       help='MetaTrader 5 account login')
    
    parser.add_argument('--password', type=str, default=None,
                       help='MetaTrader 5 account password')
    
    parser.add_argument('--server', type=str, default=None,
                       help='MetaTrader 5 server name')
    
    parser.add_argument('--terminal-path', type=str, default=None,
                       help='Path to MetaTrader 5 terminal executable')
    
    return parser.parse_args()

def main():
    """
    Main function
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create data downloader with credentials if provided
    downloader = MT5DataDownloader(
        login=args.login,
        password=args.password,
        server=args.server,
        mt5_terminal_path=args.terminal_path,
        data_dir=args.data_dir
    )
    
    # Parse dates
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # If CSV path is provided, import it
    if args.csv_path:
        if os.path.exists(args.csv_path):
            result = downloader.import_csv_data(args.csv_path, args.symbol, args.timeframe)
            if result:
                print(f"Successfully imported {args.csv_path}")
            else:
                print(f"Failed to import {args.csv_path}")
        else:
            print(f"CSV file not found: {args.csv_path}")
    
    # If login, password, and server are provided, try direct download
    elif args.login and args.password and args.server:
        print("Attempting direct download from MetaTrader 5...")
        result = downloader.download_mt5_direct(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if result:
            print(f"Successfully downloaded data to {result}")
            print("You can now use this data for backtesting with:")
            print(f"python run_backtest.py --data-file {os.path.basename(result)} --strategy ma_crossover")
        else:
            print("Failed to download data directly.")
            print("Please follow the instructions to export data manually:")
            downloader.export_instructions(
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date
            )
    
    # Otherwise, show instructions for manual export
    else:
        print("\nTo download data directly, please provide MetaTrader 5 credentials:")
        print("python download_tick_data.py --login YOUR_LOGIN --password YOUR_PASSWORD --server YOUR_SERVER --symbol EURUSD --timeframe M1 --start-date 2020-01-01 --end-date 2024-01-01\n")
        
        print("Alternatively, you can manually export data and import it:")
        print("python download_tick_data.py --csv-path /path/to/exported/data.csv --symbol EURUSD --timeframe M1\n")
        
        print("Please follow these instructions to export data from MetaTrader 5:")
        downloader.export_instructions(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

if __name__ == "__main__":
    main() 