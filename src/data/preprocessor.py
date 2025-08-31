import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Dict, Any
from ..utils.logger import TradingLogger
from ..utils.helpers import ConfigManager, calculate_technical_indicators

class DataPreprocessor:
    """Handle data preprocessing and feature engineering."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = TradingLogger("DataPreprocessor")
        self.scalers = {}
        self.feature_columns = []
        
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main preprocessing pipeline."""
        self.logger.info("Starting data preprocessing")
        
        # Create a copy to avoid modifying original data
        processed_df = df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df)
        
        # Add technical indicators
        processed_df = calculate_technical_indicators(processed_df)
        
        # Create target variables
        processed_df = self._create_targets(processed_df)
        
        # Feature engineering
        processed_df = self._engineer_features(processed_df)
        
        # Remove any remaining NaN values
        processed_df = processed_df.dropna()
        
        self.logger.info(f"Preprocessing completed. Shape: {processed_df.shape}")
        
        return processed_df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        self.logger.info("Handling missing values")
        
        # Forward fill for price data
        price_columns = ['close', 'high', 'low', 'open'] if 'open' in df.columns else ['close', 'high', 'low']
        for col in price_columns:
            if col in df.columns:
                df[col] = df[col].fillna(method='ffill')
        
        # For technical indicators, use interpolation
        indicator_columns = [col for col in df.columns if col not in price_columns + ['volume']]
        for col in indicator_columns:
            if col in df.columns and df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].interpolate(method='linear')
        
        return df
    
    def _create_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction."""
        self.logger.info("Creating target variables")
        
        # Future price movement (classification target)
        future_periods = 5  # 5 periods ahead (25 minutes for 5-min data)
        df['future_close'] = df['close'].shift(-future_periods)
        df['price_change'] = (df['future_close'] - df['close']) / df['close']
        
        # Classification targets
        threshold = 0.005  # 0.5% threshold
        df['trade_signal'] = np.where(
            df['price_change'] > threshold, 1,  # BUY
            np.where(df['price_change'] < -threshold, -1, 0)  # SELL or SKIP
        )
        
        # Regression target (profit percentage)
        df['profit_target'] = np.abs(df['price_change']) * 100
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional engineered features."""
        self.logger.info("Engineering features")
        
        # Rolling statistics
        windows = [5, 10, 20]
        for window in windows:
            df[f'close_roll_mean_{window}'] = df['close'].rolling(window).mean()
            df[f'close_roll_std_{window}'] = df['close'].rolling(window).std()
            df[f'volume_roll_mean_{window}'] = df['volume'].rolling(window).mean()
        
        # Price ratios
        df['close_to_sma_ratio'] = df['close'] / df['sma']
        df['close_to_ema_ratio'] = df['close'] / df['ema']
        
        # Momentum indicators
        df['rsi_momentum'] = df['rsi'].diff()
        df['macd_momentum'] = df['MACD_3_9_21'].diff()
        
        # Volatility
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        return df
    
    def prepare_features(self, df: pd.DataFrame, fit_scalers: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features for model training."""
        self.logger.info("Preparing features for model training")
        
        # Select feature columns
        feature_cols = self.config.get('data.features', [])
        additional_features = [col for col in df.columns if 'roll_' in col or 'ratio' in col or 'momentum' in col or col == 'volatility']
        
        self.feature_columns = feature_cols + additional_features
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Extract features
        X = df[self.feature_columns].values
        
        # Extract targets
        y_classification = df['trade_signal'].values if 'trade_signal' in df.columns else None
        y_regression = df['profit_target'].values if 'profit_target' in df.columns else None
        
        # Scale features
        if fit_scalers:
            self.scalers['features'] = StandardScaler()
            X_scaled = self.scalers['features'].fit_transform(X)
        else:
            X_scaled = self.scalers['features'].transform(X)
        
        return X_scaled, y_classification, y_regression
    
    def create_sequences(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create sequences for LSTM/GRU models."""
        X_seq, y_class_seq, y_reg_seq = [], [], []
        
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_class_seq.append(y_class[i])
            y_reg_seq.append(y_reg[i])
        
        return np.array(X_seq), np.array(y_class_seq), np.array(y_reg_seq)
    
    def split_data(self, X: np.ndarray, y_class: np.ndarray, y_reg: np.ndarray) -> Dict[str, np.ndarray]:
        """Split data into train/validation/test sets."""
        test_size = self.config.get('data.test_size', 0.2)
        val_size = self.config.get('data.validation_size', 0.1)
        
        # First split: separate test set
        X_temp, X_test, y_class_temp, y_class_test, y_reg_temp, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=test_size, random_state=42, stratify=y_class
        )
        
        # Second split: separate train and validation
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_class_train, y_class_val, y_reg_train, y_reg_val = train_test_split(
            X_temp, y_class_temp, y_reg_temp, test_size=val_ratio, random_state=42, stratify=y_class_temp
        )
        
        return {
            'X_train': X_train, 'y_class_train': y_class_train, 'y_reg_train': y_reg_train,
            'X_val': X_val, 'y_class_val': y_class_val, 'y_reg_val': y_reg_val,
            'X_test': X_test, 'y_class_test': y_class_test, 'y_reg_test': y_reg_test
        }