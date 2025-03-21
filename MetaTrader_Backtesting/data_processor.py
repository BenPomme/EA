import pandas as pd
import numpy as np
import datetime
import os
from pathlib import Path

class MetaTraderDataProcessor:
    """
    Class to process MetaTrader data files for backtesting
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the data processor
        
        Parameters:
        -----------
        data_path : str, optional
            Path to the directory containing MetaTrader CSV files
        """
        self.data_path = data_path
        
    def set_data_path(self, data_path):
        """
        Set the data path
        
        Parameters:
        -----------
        data_path : str
            Path to the directory containing MetaTrader CSV files
        """
        self.data_path = data_path
        
    def load_mt4_csv(self, filename, timeframe="1D"):
        """
        Load MetaTrader 4 CSV file
        
        Parameters:
        -----------
        filename : str
            Name of the CSV file
        timeframe : str, optional
            Timeframe of the data (default: "1D")
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with OHLCV data
        """
        filepath = os.path.join(self.data_path, filename)
        
        # Read the CSV file
        df = pd.read_csv(filepath, header=None)
        
        # Rename columns based on MT4 CSV format
        # Assuming the format: date,time,open,high,low,close,volume
        df.columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        
        # Convert date and time to datetime
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'], 
                                        format='%Y.%m.%d %H:%M')
        
        # Drop original date and time columns
        df.drop(['date', 'time'], axis=1, inplace=True)
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        # Convert values to appropriate types
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        
        return df
    
    def load_mt5_csv(self, filename, timeframe="1D"):
        """
        Load MetaTrader 5 CSV file
        
        Parameters:
        -----------
        filename : str
            Name of the CSV file
        timeframe : str, optional
            Timeframe of the data (default: "1D")
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with OHLCV data
        """
        filepath = os.path.join(self.data_path, filename)
        
        # Read the CSV file (MT5 format might be different)
        df = pd.read_csv(filepath)
        
        # Adjust column names if necessary
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        
        # Check if the required columns are in the DataFrame
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the CSV file: {missing_columns}")
        
        # Convert time to datetime
        df['datetime'] = pd.to_datetime(df['time'])
        
        # Drop original time column
        df.drop(['time'], axis=1, inplace=True)
        
        # Set datetime as index
        df.set_index('datetime', inplace=True)
        
        return df
    
    def resample_data(self, df, timeframe):
        """
        Resample data to a different timeframe
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
        timeframe : str
            Target timeframe (e.g., '1H', '4H', '1D')
            
        Returns:
        --------
        pandas.DataFrame
            Resampled DataFrame
        """
        # Define resampling rules
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample the data
        resampled_df = df.resample(timeframe).agg(ohlc_dict)
        
        # Drop rows with missing values
        resampled_df.dropna(inplace=True)
        
        return resampled_df
    
    def prepare_backtest_data(self, df):
        """
        Prepare data for backtrader backtesting
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame prepared for backtrader
        """
        # Make sure the DataFrame has the correct format for backtrader
        bt_df = df.copy()
        
        # Rename columns to match backtrader's expected format if necessary
        if 'open' in bt_df.columns:
            bt_df.rename(columns={
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }, inplace=True)
        
        # Make sure the index is a datetime index
        if not isinstance(bt_df.index, pd.DatetimeIndex):
            bt_df.index = pd.to_datetime(bt_df.index)
        
        return bt_df
    
    def save_as_csv(self, df, filename):
        """
        Save DataFrame as CSV file
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to save
        filename : str
            Output filename
        """
        output_path = os.path.join(self.data_path, filename)
        df.to_csv(output_path)
        print(f"Data saved to {output_path}") 