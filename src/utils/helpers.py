import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any
import yaml
import joblib
import os

class ConfigManager:
    """Manage configuration files and parameters."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

class ModelPersistence:
    """Handle model saving and loading operations."""
    
    @staticmethod
    def save_model(model: Any, filepath: str) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
    
    @staticmethod
    def load_model(filepath: str) -> Any:
        """Load model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        return joblib.load(filepath)

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional technical indicators."""
    df = df.copy()
    
    # Price momentum
    df['price_momentum'] = df['close'].pct_change()
    
    # Bollinger Bands
    df['bb_upper'] = df['sma'] + (2 * df['close'].rolling(20).std())
    df['bb_lower'] = df['sma'] - (2 * df['close'].rolling(20).std())
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # Volume indicators
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']
    
    # Price position relative to high/low
    df['hl_ratio'] = (df['close'] - df['low']) / (df['high'] - df['low'])
    
    return df

def create_sequences(data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences for time series prediction."""
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i-sequence_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def calculate_profit_percentage(entry_price: float, exit_price: float, trade_type: str) -> float:
    """Calculate profit percentage for a trade."""
    if trade_type.upper() == 'BUY':
        return ((exit_price - entry_price) / entry_price) * 100
    elif trade_type.upper() == 'SELL':
        return ((entry_price - exit_price) / entry_price) * 100
    else:
        return 0.0