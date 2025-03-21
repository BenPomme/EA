import os
import sys
import subprocess
import datetime
import pandas as pd
import time
from pathlib import Path

# Try to import MetaTrader5 package, handle gracefully if not available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("MetaTrader5 package not available. Will use CSV export/import method instead.")

from data_processor import MetaTraderDataProcessor

class MT5DataDownloader:
    """
    Helper class to download data from MetaTrader 5 and prepare it for backtesting
    
    Since we can't directly use the MT5 API on this platform, this script provides instructions
    on how to export data from MT5 terminal and then import it into our system.
    """
    
    def __init__(self, login=None, password=None, server=None, mt5_terminal_path=None, data_dir='data'):
        """
        Initialize the data downloader
        
        Parameters:
        -----------
        login : int, optional
            MT5 account login number
        password : str, optional
            MT5 account password
        server : str, optional
            MT5 server name
        mt5_terminal_path : str, optional
            Path to MT5 terminal executable (needed for some platforms)
        data_dir : str, optional
            Directory to save processed data
        """
        self.login = login
        self.password = password
        self.server = server
        self.mt5_terminal_path = mt5_terminal_path
        self.data_dir = data_dir
        self.processor = MetaTraderDataProcessor(data_dir)
        self.connected = False
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Try to connect to MT5 if credentials are provided
        if MT5_AVAILABLE and login and password and server:
            self.connect_to_mt5()
    
    def connect_to_mt5(self):
        """
        Connect to MetaTrader 5 terminal
        
        Returns:
        --------
        bool
            True if connection successful, False otherwise
        """
        if not MT5_AVAILABLE:
            print("MetaTrader5 package not available. Cannot connect directly.")
            return False
        
        # Initialize MT5
        init_result = False
        
        if self.mt5_terminal_path:
            init_result = mt5.initialize(path=self.mt5_terminal_path)
        else:
            init_result = mt5.initialize()
        
        if not init_result:
            print(f"MetaTrader5 initialization failed, error code = {mt5.last_error()}")
            return False
        
        # Login to MT5 account
        authorized = mt5.login(self.login, server=self.server, password=self.password)
        if not authorized:
            print(f"Login failed, error code = {mt5.last_error()}")
            mt5.shutdown()
            return False
        
        print(f"Connected to MetaTrader5 server: {mt5.account_info().server}")
        self.connected = True
        return True
    
    def disconnect_from_mt5(self):
        """
        Disconnect from MetaTrader 5 terminal
        """
        if MT5_AVAILABLE and self.connected:
            mt5.shutdown()
            self.connected = False
            print("Disconnected from MetaTrader5 server")
    
    def export_instructions(self, symbol='EURUSD', timeframe='H1', start_date=None, end_date=None):
        """
        Print instructions for exporting data from MT5 terminal
        
        Parameters:
        -----------
        symbol : str, optional
            Symbol to export (default: 'EURUSD')
        timeframe : str, optional
            Timeframe to export (default: 'H1')
        start_date : str, optional
            Start date in format 'YYYY.MM.DD'
        end_date : str, optional
            End date in format 'YYYY.MM.DD'
        """
        # Convert timeframe to MT5 format
        timeframe_map = {
            '1M': 'MN1',
            '1W': 'W1',
            '1D': 'D1',
            '4H': 'H4',
            '1H': 'H1',
            '30M': 'M30',
            '15M': 'M15',
            '5M': 'M5',
            '1M': 'M1'
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, timeframe)
        
        # Calculate date range if not provided
        if not start_date:
            start_date_dt = datetime.datetime.now() - datetime.timedelta(days=365*4)  # 4 years ago
            start_date = start_date_dt.strftime('%Y.%m.%d')
        
        if not end_date:
            end_date = datetime.datetime.now().strftime('%Y.%m.%d')
        
        # Print instructions
        print(f"\n=== Instructions for Exporting {symbol} {mt5_timeframe} Data from MetaTrader 5 ===\n")
        print("Method 1: Using the Chart")
        print("1. Open a chart of the desired symbol and timeframe in MetaTrader 5")
        print("2. Right-click on the chart and select 'Chart properties' or press F8")
        print("3. Go to the 'Common' tab")
        print("4. Click on 'Save As...' or 'Export' to save the chart data as CSV")
        print(f"5. Set the date range from {start_date} to {end_date}")
        print("6. Save the file to a location you can easily access\n")
        
        print("Method 2: Using Tools > History")
        print("1. Open MetaTrader 5 terminal")
        print("2. Click on 'Tools' menu and then 'History Center' (or press F2)")
        print(f"3. Find '{symbol}' in the list and expand it")
        print(f"4. Select the '{mt5_timeframe}' timeframe")
        print("5. Right-click and select 'Export' or 'Save as...'")
        print(f"6. Set the date range from {start_date} to {end_date}")
        print("7. Save the file to a location you can easily access\n")
        
        print("After exporting, run the import_csv_data() method with the path to your exported file:")
        print(f"downloader.import_csv_data('/path/to/your/{symbol}_{mt5_timeframe}.csv')\n")
    
    def import_csv_data(self, csv_path, symbol=None, timeframe=None):
        """
        Import CSV data exported from MT5
        
        Parameters:
        -----------
        csv_path : str
            Path to the CSV file exported from MT5
        symbol : str, optional
            Symbol of the data (default: derived from filename)
        timeframe : str, optional
            Timeframe of the data (default: derived from filename)
            
        Returns:
        --------
        str
            Path to the processed data file
        """
        # Try to derive symbol and timeframe from filename if not provided
        if not symbol or not timeframe:
            filename = os.path.basename(csv_path)
            parts = filename.split('_')
            if len(parts) >= 2:
                if not symbol:
                    symbol = parts[0]
                if not timeframe:
                    tf_part = parts[1].split('.')[0]
                    # Convert MT5 timeframe format to our format
                    timeframe_map = {
                        'MN1': '1M',
                        'W1': '1W',
                        'D1': '1D',
                        'H4': '4H',
                        'H1': '1H',
                        'M30': '30M',
                        'M15': '15M',
                        'M5': '5M',
                        'M1': '1M'
                    }
                    timeframe = timeframe_map.get(tf_part, tf_part)
        
        print(f"Importing {symbol} {timeframe} data from {csv_path}...")
        
        # Copy the CSV file to our data directory
        dest_filename = f"{symbol}_{timeframe}.csv"
        dest_path = os.path.join(self.data_dir, dest_filename)
        
        # Read the CSV file and save it in our format
        try:
            # Try MT5 format first (tab-separated with Date and Time columns)
            try:
                df = pd.read_csv(csv_path, sep='\t', parse_dates=[[0, 1]])
                if 'Date_Time' in df.columns:
                    df.rename(columns={'Date_Time': 'datetime'}, inplace=True)
                df.set_index('datetime', inplace=True)
            except:
                # Try alternative MT5 format (comma-separated)
                df = pd.read_csv(csv_path)
                
                # Handle various date/time column formats
                if 'time' in df.columns:
                    # Unix timestamp format
                    if df['time'].dtype == int or df['time'].dtype == float:
                        df['datetime'] = pd.to_datetime(df['time'], unit='s')
                    else:
                        # String datetime format
                        df['datetime'] = pd.to_datetime(df['time'])
                    df.set_index('datetime', inplace=True)
                elif '<DATE>' in df.columns and '<TIME>' in df.columns:
                    # MetaTrader export format
                    df['datetime'] = pd.to_datetime(
                        df['<DATE>'].astype(str) + ' ' + df['<TIME>'].astype(str),
                        format='%Y.%m.%d %H:%M'
                    )
                    df.rename(columns={
                        '<OPEN>': 'open',
                        '<HIGH>': 'high',
                        '<LOW>': 'low',
                        '<CLOSE>': 'close',
                        '<TICKVOL>': 'volume'
                    }, inplace=True)
                    df.set_index('datetime', inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                elif 'Date' in df.columns and 'Time' in df.columns:
                    # Another common format
                    df['datetime'] = pd.to_datetime(
                        df['Date'].astype(str) + ' ' + df['Time'].astype(str)
                    )
                    df.set_index('datetime', inplace=True)
            
            # Check if we have the required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            # Convert column names to lowercase for checking
            df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
            
            # Map any standard column names to our format
            col_map = {
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vol': 'volume',
                'tick_volume': 'volume'
            }
            
            for old_col, new_col in col_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Check for missing columns
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"Warning: Missing columns: {missing_cols}")
                # If volume is missing, add a placeholder
                if 'volume' in missing_cols:
                    df['volume'] = 1
            
            # Save processed data
            df.to_csv(dest_path)
            print(f"Data successfully imported and saved to {dest_path}")
            
            # Prepare for backtesting
            processed_df = self.processor.prepare_backtest_data(df)
            processed_filename = f"processed_{dest_filename}"
            processed_path = os.path.join(self.data_dir, processed_filename)
            processed_df.to_csv(processed_path)
            print(f"Data prepared for backtesting and saved to {processed_path}")
            
            return processed_path
            
        except Exception as e:
            print(f"Error importing data: {e}")
            print("Please make sure the CSV file is in the correct format.")
            return None

    def download_mt5_direct(self, symbol='EURUSD', timeframe='1H', 
                          start_date=None, end_date=None):
        """
        Download data directly from MetaTrader 5 using the API
        
        Parameters:
        -----------
        symbol : str, optional
            Symbol to download (default: 'EURUSD')
        timeframe : str, optional
            Timeframe to download (default: '1H')
        start_date : datetime.datetime, optional
            Start date for data download
        end_date : datetime.datetime, optional
            End date for data download
            
        Returns:
        --------
        str
            Path to the saved data file
        """
        if not MT5_AVAILABLE:
            print("MetaTrader5 package not available.")
            print("Please use the export_instructions() method to export data manually.")
            self.export_instructions(symbol, timeframe, 
                                    start_date.strftime('%Y.%m.%d') if start_date else None,
                                    end_date.strftime('%Y.%m.%d') if end_date else None)
            return None
        
        # Connect to MT5 if not already connected
        if not self.connected:
            if not self.connect_to_mt5():
                print("Could not connect to MetaTrader5. Please check your credentials.")
                self.export_instructions(symbol, timeframe, 
                                         start_date.strftime('%Y.%m.%d') if start_date else None,
                                         end_date.strftime('%Y.%m.%d') if end_date else None)
                return None
        
        # Convert timeframe string to MT5 timeframe
        timeframe_map = {
            '1M': mt5.TIMEFRAME_MN1,
            '1W': mt5.TIMEFRAME_W1,
            '1D': mt5.TIMEFRAME_D1,
            '4H': mt5.TIMEFRAME_H4,
            '1H': mt5.TIMEFRAME_H1,
            '30M': mt5.TIMEFRAME_M30,
            '15M': mt5.TIMEFRAME_M15,
            '5M': mt5.TIMEFRAME_M5,
            '1M': mt5.TIMEFRAME_M1
        }
        
        mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
        
        # Calculate date range if not provided
        if not start_date:
            start_date = datetime.datetime.now() - datetime.timedelta(days=365*4)  # 4 years ago
        
        if not end_date:
            end_date = datetime.datetime.now()
        
        # Download the data
        print(f"Downloading {symbol} {timeframe} data from {start_date} to {end_date}...")
        
        # For ticks, we need to batch our requests due to limitations
        if timeframe.lower() == 'tick':
            all_ticks = []
            current_date = start_date
            
            while current_date < end_date:
                next_date = min(current_date + datetime.timedelta(days=30), end_date)
                
                print(f"Downloading batch from {current_date} to {next_date}...")
                ticks = mt5.copy_ticks_range(symbol, current_date, next_date, mt5.COPY_TICKS_ALL)
                
                if ticks is None or len(ticks) == 0:
                    print(f"No ticks for {current_date} to {next_date}, error code = {mt5.last_error()}")
                else:
                    print(f"Downloaded {len(ticks)} ticks")
                    all_ticks.extend(ticks)
                
                current_date = next_date
                time.sleep(1)  # Avoid overloading the API
            
            if len(all_ticks) == 0:
                print("No ticks downloaded")
                self.disconnect_from_mt5()
                return None
            
            # Convert ticks to DataFrame
            df = pd.DataFrame(all_ticks)
            
        else:
            # For regular timeframes, use copy_rates_range
            rates = mt5.copy_rates_range(symbol, mt5_timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f"No data downloaded, error code = {mt5.last_error()}")
                self.disconnect_from_mt5()
                return None
            
            print(f"Downloaded {len(rates)} bars")
            df = pd.DataFrame(rates)
        
        # Convert time column to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Create output filename and save
        output_filename = f"{symbol}_{timeframe}.csv"
        output_path = os.path.join(self.data_dir, output_filename)
        
        # Save data
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
        
        # Prepare for backtesting
        df.set_index('time', inplace=True)
        df.index.name = 'datetime'
        
        # Make sure we have the right column names
        bt_df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
        
        # If volume is missing, use tick_volume
        if 'volume' not in bt_df.columns and 'tick_volume' in df.columns:
            bt_df['volume'] = df['tick_volume']
        
        # Prepare for backtesting
        processed_filename = f"processed_{output_filename}"
        processed_path = os.path.join(self.data_dir, processed_filename)
        bt_df.to_csv(processed_path)
        print(f"Data prepared for backtesting and saved to {processed_path}")
        
        return processed_path
    
    def __del__(self):
        """
        Destructor to ensure MT5 connection is closed
        """
        self.disconnect_from_mt5()

def main():
    """
    Main function for command-line usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Download data from MetaTrader 5')
    
    parser.add_argument('--login', type=int, default=None,
                       help='MetaTrader 5 account login')
    
    parser.add_argument('--password', type=str, default=None,
                       help='MetaTrader 5 account password')
    
    parser.add_argument('--server', type=str, default=None,
                       help='MetaTrader 5 server name')
    
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Symbol to download')
    
    parser.add_argument('--timeframe', type=str, default='1H',
                       help='Timeframe to download')
    
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date (YYYY-MM-DD)')
    
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to save data')
    
    parser.add_argument('--csv-path', type=str, default=None,
                       help='Path to already exported CSV file')
    
    parser.add_argument('--terminal-path', type=str, default=None,
                       help='Path to MetaTrader 5 terminal executable')
    
    args = parser.parse_args()
    
    # Create data downloader
    downloader = MT5DataDownloader(
        login=args.login,
        password=args.password,
        server=args.server,
        mt5_terminal_path=args.terminal_path,
        data_dir=args.data_dir
    )
    
    # Parse dates if provided
    start_date = None
    end_date = None
    
    if args.start_date:
        start_date = datetime.datetime.strptime(args.start_date, '%Y-%m-%d')
    
    if args.end_date:
        end_date = datetime.datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # If CSV path is provided, import it
    if args.csv_path:
        result = downloader.import_csv_data(args.csv_path, args.symbol, args.timeframe)
        if result:
            print(f"Successfully imported data to {result}")
        else:
            print("Failed to import data")
    
    # If credentials are provided, try direct download
    elif args.login and args.password and args.server:
        result = downloader.download_mt5_direct(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=start_date,
            end_date=end_date
        )
        
        if result:
            print(f"Successfully downloaded data to {result}")
        else:
            print("Failed to download data directly")
    
    # Otherwise, show instructions
    else:
        print("Please provide MetaTrader 5 credentials for direct download or CSV path for import.")
        print("Examples:")
        print("   Direct download: python mt5_data_downloader.py --login 12345 --password 'your_pass' --server 'your_server' --symbol EURUSD --timeframe 1H --start-date 2020-01-01 --end-date 2024-01-01")
        print("   CSV import: python mt5_data_downloader.py --csv-path /path/to/data.csv --symbol EURUSD --timeframe 1H")
        
        # Show export instructions anyway
        downloader.export_instructions(
            symbol=args.symbol,
            timeframe=args.timeframe,
            start_date=args.start_date,
            end_date=args.end_date
        )

if __name__ == "__main__":
    main() 