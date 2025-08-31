"""
Basic usage example of the ML Trading System
"""

import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.helpers import ConfigManager
from data.data_loader import DataLoader
from data.preprocessor import DataPreprocessor
from models.ensemble_model import EnsembleModel
from models.lstm_model import LSTMModel
from trading.signal_generator import SignalGenerator
from trading.risk_manager import RiskManager
from evaluation.backtester import Backtester

class TradingSystem:
    """Main trading system orchestrator."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = ConfigManager(config_path)
        self.data_loader = DataLoader(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Models
        self.ensemble_model = EnsembleModel(self.config.get('models.ensemble', {}))
        self.lstm_model = LSTMModel(self.config.get('models.lstm', {}))
        
        self.models_trained = False
    
    def load_and_preprocess_data(self, train_path: str, test_path: str):
        """Load and preprocess training and testing data."""
        print("Loading and preprocessing data...")
        
        # Load data
        self.train_df, self.test_df = self.data_loader.load_training_data(train_path, test_path)
        
        # Preprocess training data
        self.processed_train = self.preprocessor.preprocess_data(self.train_df)
        
        # Prepare features for training
        X_train, y_class_train, y_reg_train = self.preprocessor.prepare_features(
            self.processed_train, fit_scalers=True
        )
        
        # Split data
        self.data_splits = self.preprocessor.split_data(X_train, y_class_train, y_reg_train)
        
        # Create sequences for LSTM
        seq_length = self.config.get('models.lstm.sequence_length', 60)
        X_seq, y_class_seq, y_reg_seq = self.preprocessor.create_sequences(
            X_train, y_class_train, y_reg_train, seq_length
        )
        
        self.lstm_data = {
            'X_seq': X_seq,
            'y_class_seq': y_class_seq,
            'y_reg_seq': y_reg_seq
        }
        
        print(f"Data preprocessing completed:")
        print(f"Training samples: {len(self.data_splits['X_train'])}")
        print(f"Validation samples: {len(self.data_splits['X_val'])}")
        print(f"Test samples: {len(self.data_splits['X_test'])}")
        print(f"LSTM sequences: {len(X_seq)}")
    
    def train_models(self):
        """Train all models."""
        print("Training models...")
        
        # Train ensemble classification model
        self.ensemble_model.build_model("both")
        
        print("Training ensemble classifier...")
        self.ensemble_model.train(
            self.data_splits['X_train'],
            self.data_splits['y_class_train'],
            self.data_splits['X_val'],
            self.data_splits['y_class_val'],
            task_type="classification"
        )
        
        # Train LSTM model for regression
        print("Training LSTM model...")
        input_shape = (self.lstm_data['X_seq'].shape[1], self.lstm_data['X_seq'].shape[2])
        self.lstm_model.build_model(input_shape, output_dim=1)
        
        # Split LSTM data
        split_idx = int(len(self.lstm_data['X_seq']) * 0.8)
        X_lstm_train = self.lstm_data['X_seq'][:split_idx]
        X_lstm_val = self.lstm_data['X_seq'][split_idx:]
        y_lstm_train = self.lstm_data['y_reg_seq'][:split_idx]
        y_lstm_val = self.lstm_data['y_reg_seq'][split_idx:]
        
        self.lstm_model.train(X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val)
        
        self.models_trained = True
        print("Model training completed!")
    
    def generate_predictions_and_signals(self, data: pd.DataFrame):
        """Generate predictions and trading signals for given data."""
        if not self.models_trained:
            raise ValueError("Models must be trained first!")
        
        print("Generating predictions and signals...")
        
        # Preprocess data
        processed_data = self.preprocessor.preprocess_data(data)
        X, y_class, y_reg = self.preprocessor.prepare_features(processed_data, fit_scalers=False)
        
        # Generate ensemble predictions
        classification_pred = self.ensemble_model.predict(X, task_type="classification")
        classification_proba = self.ensemble_model.predict_proba(X)
        
        # Generate LSTM predictions for sequences
        seq_length = self.config.get('models.lstm.sequence_length', 60)
        X_seq, _, _ = self.preprocessor.create_sequences(X, y_class, y_reg, seq_length)
        
        if len(X_seq) > 0:
            lstm_pred = self.lstm_model.predict(X_seq).flatten()
            
            # Align predictions (LSTM has fewer predictions due to sequence requirement)
            aligned_class_pred = classification_pred[-len(lstm_pred):]
            aligned_class_proba = classification_proba[-len(lstm_pred):]
            current_prices = processed_data['close'].iloc[-len(lstm_pred):].values
            
            # Generate signals
            signals_df = self.signal_generator.generate_signals(
                aligned_class_pred,
                aligned_class_proba,
                lstm_pred,
                current_prices
            )
            
            return signals_df, processed_data.iloc[-len(lstm_pred):]
        
        return pd.DataFrame(), processed_data
    
    def run_backtest(self, test_data: pd.DataFrame):
        """Run backtest on test data."""
        print("Running backtest...")
        
        signals_df, processed_test = self.generate_predictions_and_signals(test_data)
        
        if signals_df.empty:
            print("No signals generated for backtesting")
            return {}
        
        # Initialize backtester
        backtester = Backtester(self.config, initial_balance=10000)
        
        # Run backtest
        results = backtester.run_backtest(signals_df, processed_test)
        
        return results

def main():
    """Main execution function."""
    print("=== ML Trading System - Basic Usage Example ===")
    
    # Initialize system
    system = TradingSystem()
    
    # Example data paths (update these with your actual file paths)
    train_data_path = "data/raw/BNBUSDT_2024_5m.csv"
    test_data_path = "data/raw/BNBUSDT_2025_01_5m.csv"
    
    try:
        # Check if data files exist
        if not os.path.exists(train_data_path):
            print(f"Warning: Training data file not found: {train_data_path}")
            print("Please ensure you have the BNBUSDT data files in the correct location")
            return
        
        if not os.path.exists(test_data_path):
            print(f"Warning: Test data file not found: {test_data_path}")
            print("Please ensure you have the 2025-01 validation data file")
            return
        
        # Load and preprocess data
        system.load_and_preprocess_data(train_data_path, test_data_path)
        
        # Train models
        system.train_models()
        
        # Run backtest
        backtest_results = system.run_backtest(system.test_df)
        
        # Display results
        print("\n=== Backtest Results ===")
        for key, value in backtest_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        print("\n=== Trading System Completed Successfully! ===")
        
    except Exception as e:
        print(f"Error running trading system: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()