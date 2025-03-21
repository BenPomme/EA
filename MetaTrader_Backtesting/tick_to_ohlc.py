import os
import argparse
import pandas as pd
import numpy as np
import datetime
from data_processor import MetaTraderDataProcessor

def parse_args():
    """
    Parse command-line arguments
    
    Returns:
    --------
    argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Convert tick data to OHLC data')
    
    parser.add_argument('--tick-file', type=str, required=True,
                       help='CSV file containing tick data')
    
    parser.add_argument('--output-timeframe', type=str, default='1H',
                       help='Timeframe for output OHLC data (e.g., 1M, 1D, 4H, 1H, 15M, 5M, 1M)')
    
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Symbol name')
    
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory containing data files')
    
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Directory to save output files')
    
    return parser.parse_args()

def convert_ticks_to_ohlc(tick_df, timeframe):
    """
    Convert tick data to OHLC data
    
    Parameters:
    -----------
    tick_df : pandas.DataFrame
        DataFrame containing tick data with bid, ask
    timeframe : str
        Target timeframe (e.g., '1M', '1D', '4H', '1H', '15M', '5M', '1M')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with OHLCV data
    """
    # Define resampling frequency
    freq_map = {
        '1M': 'M',
        '1W': 'W',
        '1D': 'D',
        '4H': '4H',
        '1H': 'H',
        '30M': '30T',
        '15M': '15T',
        '5M': '5T',
        '1M': '1T'
    }
    
    freq = freq_map.get(timeframe, 'H')
    
    # Calculate mid price
    if 'bid' in tick_df.columns and 'ask' in tick_df.columns:
        tick_df['price'] = (tick_df['bid'] + tick_df['ask']) / 2
    
    # Make sure we have time as index
    if 'time' in tick_df.columns and not isinstance(tick_df.index, pd.DatetimeIndex):
        tick_df.set_index('time', inplace=True)
    
    # Resample to target timeframe
    ohlc_data = tick_df['price'].resample(freq).ohlc()
    
    # Handle volume
    if 'volume' in tick_df.columns:
        volume = tick_df['volume'].resample(freq).sum()
        ohlc_data['volume'] = volume
    else:
        # If no volume, use count as volume
        volume = tick_df['price'].resample(freq).count()
        ohlc_data['volume'] = volume
    
    # Drop rows with missing values
    ohlc_data.dropna(inplace=True)
    
    return ohlc_data

def main():
    """
    Main function
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Create data processor
    processor = MetaTraderDataProcessor(args.data_dir)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tick data
    tick_file_path = os.path.join(args.data_dir, args.tick_file)
    if not os.path.exists(tick_file_path):
        tick_file_path = args.tick_file  # Try absolute path
    
    print(f"Loading tick data from {tick_file_path}...")
    
    try:
        # Try different formats
        try:
            # MT5 tick format
            tick_df = pd.read_csv(tick_file_path)
            if 'time' in tick_df.columns:
                tick_df['time'] = pd.to_datetime(tick_df['time'], unit='s')
        except:
            # Try standard CSV format
            tick_df = pd.read_csv(tick_file_path, parse_dates=True)
            if 'time' not in tick_df.columns:
                # Look for datetime column
                datetime_cols = [col for col in tick_df.columns if 'time' in col.lower() or 'date' in col.lower()]
                if datetime_cols:
                    tick_df.rename(columns={datetime_cols[0]: 'time'}, inplace=True)
                    tick_df['time'] = pd.to_datetime(tick_df['time'])
        
        # Convert ticks to OHLC
        print(f"Converting ticks to {args.output_timeframe} OHLC data...")
        ohlc_df = convert_ticks_to_ohlc(tick_df, args.output_timeframe)
        
        # Save to CSV
        output_filename = f"{args.symbol}_{args.output_timeframe}.csv"
        output_path = os.path.join(args.output_dir, output_filename)
        ohlc_df.to_csv(output_path)
        print(f"OHLC data saved to {output_path}")
        
        # Convert to format for backtesting
        print("Preparing data for backtesting...")
        bt_df = processor.prepare_backtest_data(ohlc_df)
        processed_filename = f"processed_{output_filename}"
        processor.save_as_csv(bt_df, processed_filename)
        print(f"Processed data saved to {os.path.join(args.output_dir, processed_filename)}")
        
    except Exception as e:
        print(f"Error processing tick data: {e}")
        print("Please make sure the tick data is in the correct format.")

if __name__ == "__main__":
    main() 