import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from ..utils.logger import TradingLogger
from ..utils.helpers import ConfigManager

class DataLoader:
    """Handle data loading and basic validation."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = TradingLogger("DataLoader")
        self.required_columns = self.config.get('data.features', [])
    
    def load_csv(self, filepath: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
        """Load data from CSV file with validation."""
        try:
            self.logger.info(f"Loading data from {filepath}")
            
            df = pd.read_csv(filepath)
            
            if parse_dates:
                for date_col in parse_dates:
                    if date_col in df.columns:
                        df[date_col] = pd.to_datetime(df[date_col])
            
            # Validate required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            self.logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data from {filepath}: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data quality and completeness."""
        try:
            # Check for empty dataframe
            if df.empty:
                raise ValueError("Dataframe is empty")
            
            # Check for missing values in critical columns
            critical_cols = ['close', 'high', 'low', 'volume']
            for col in critical_cols:
                if col in df.columns:
                    missing_pct = df[col].isnull().sum() / len(df)
                    if missing_pct > 0.05:  # More than 5% missing
                        self.logger.warning(f"Column {col} has {missing_pct:.2%} missing values")
            
            # Check for data consistency
            if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
                invalid_rows = (df['high'] < df['low']) | (df['close'] > df['high']) | (df['close'] < df['low'])
                if invalid_rows.any():
                    self.logger.warning(f"Found {invalid_rows.sum()} rows with invalid OHLC data")
            
            self.logger.info("Data validation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            return False
    
    def load_training_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both training and testing datasets."""
        train_df = self.load_csv(train_path)
        test_df = self.load_csv(test_path)
        
        # Validate both datasets
        if not (self.validate_data(train_df) and self.validate_data(test_df)):
            raise ValueError("Data validation failed")
        
        return train_df, test_df