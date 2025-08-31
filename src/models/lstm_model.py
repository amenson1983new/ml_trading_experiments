import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from typing import Dict, Any, Optional, Tuple
from .base_model import BaseModel

class LSTMModel(BaseModel):
    """LSTM model for time series prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("LSTM", config)
        
    def build_model(self, input_shape: Tuple[int, int], output_dim: int = 1) -> None:
        """Build LSTM model architecture."""
        self.logger.info(f"Building LSTM model with input shape: {input_shape}")
        
        model = Sequential([
            LSTM(
                self.config.get('hidden_size', 128),
                return_sequences=True,
                input_shape=input_shape,
                dropout=self.config.get('dropout', 0.2),
                recurrent_dropout=self.config.get('dropout', 0.2)
            ),
            BatchNormalization(),
            
            LSTM(
                self.config.get('hidden_size', 128) // 2,
                return_sequences=False,
                dropout=self.config.get('dropout', 0.2),
                recurrent_dropout=self.config.get('dropout', 0.2)
            ),
            BatchNormalization(),
            
            Dense(64, activation='relu'),
            Dropout(self.config.get('dropout', 0.2)),
            
            Dense(32, activation='relu'),
            Dropout(self.config.get('dropout', 0.2)),
            
            Dense(output_dim, activation='linear' if output_dim == 1 else 'softmax')
        ])
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001))
        
        if output_dim == 1:  # Regression
            model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        else:  # Classification
            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        
        self.model = model
        self.logger.info("LSTM model built successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the LSTM model."""
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        self.logger.info("Starting LSTM model training")
        
        # Prepare callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
        ]
        
        # Training parameters
        epochs = self.config.get('epochs', 100)
        batch_size = self.config.get('batch_size', 32)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_trained = True
        self.logger.info("LSTM model training completed")
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metric_names = self.model.metrics_names
        
        return dict(zip(metric_names, results))