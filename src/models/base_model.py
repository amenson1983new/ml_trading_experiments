from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Tuple
from ..utils.logger import TradingLogger

class BaseModel(ABC):
    """Abstract base class for all trading models."""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = TradingLogger(f"Model_{name}")
        self.model = None
        self.is_trained = False
        
    @abstractmethod
    def build_model(self) -> None:
        """Build the model architecture."""
        pass
    
    @abstractmethod
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pass
    
    def save(self, filepath: str) -> None:
        """Save the model."""
        from ..utils.helpers import ModelPersistence
        ModelPersistence.save_model(self.model, filepath)
        self.logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load the model."""
        from ..utils.helpers import ModelPersistence
        self.model = ModelPersistence.load_model(filepath)
        self.is_trained = True
        self.logger.info(f"Model loaded from {filepath}")