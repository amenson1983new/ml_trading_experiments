import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from ..utils.logger import TradingLogger
from ..utils.helpers import ConfigManager

class SignalGenerator:
    """Generate trading signals based on model predictions."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = TradingLogger("SignalGenerator")
        self.confidence_threshold = self.config.get('trading.min_confidence_threshold', 0.7)
    
    def generate_signals(self, 
                        classification_pred: np.ndarray,
                        classification_proba: np.ndarray,
                        regression_pred: np.ndarray,
                        current_prices: np.ndarray) -> pd.DataFrame:
        """Generate comprehensive trading signals."""
        
        self.logger.info("Generating trading signals")
        
        signals = []
        
        for i in range(len(classification_pred)):
            signal = self._create_single_signal(
                classification_pred[i],
                classification_proba[i],
                regression_pred[i],
                current_prices[i],
                i
            )
            signals.append(signal)
        
        signals_df = pd.DataFrame(signals)
        self.logger.info(f"Generated {len(signals_df)} trading signals")
        
        return signals_df
    
    def _create_single_signal(self, 
                             class_pred: int,
                             class_proba: np.ndarray,
                             reg_pred: float,
                             current_price: float,
                             index: int) -> Dict:
        """Create a single trading signal."""
        
        # Get prediction confidence
        confidence = np.max(class_proba)
        
        # Determine signal type
        if confidence < self.confidence_threshold:
            signal_type = "SKIP"
            direction = 0
        elif class_pred == 1:  # BUY signal
            signal_type = "BUY"
            direction = 1
        elif class_pred == -1:  # SELL signal
            signal_type = "SELL"
            direction = -1
        else:
            signal_type = "SKIP"
            direction = 0
        
        # Calculate expected profit
        expected_profit = reg_pred if signal_type != "SKIP" else 0.0
        
        # Calculate position size based on confidence and expected profit
        position_size = self._calculate_position_size(confidence, expected_profit)
        
        return {
            'index': index,
            'signal_type': signal_type,
            'direction': direction,
            'confidence': confidence,
            'expected_profit': expected_profit,
            'current_price': current_price,
            'position_size': position_size,
            'class_probabilities': class_proba.tolist()
        }
    
    def _calculate_position_size(self, confidence: float, expected_profit: float) -> float:
        """Calculate position size based on confidence and expected profit."""
        base_size = 1.0
        
        # Adjust based on confidence
        confidence_multiplier = confidence
        
        # Adjust based on expected profit
        profit_multiplier = min(expected_profit / 2.0, 2.0)  # Cap at 2x
        
        position_size = base_size * confidence_multiplier * profit_multiplier
        
        # Ensure position size is within reasonable bounds
        return max(0.1, min(position_size, 5.0))
    
    def filter_signals(self, signals_df: pd.DataFrame, filters: Dict[str, any] = None) -> pd.DataFrame:
        """Apply additional filters to trading signals."""
        if filters is None:
            filters = {}
        
        filtered_df = signals_df.copy()
        
        # Filter by minimum confidence
        min_confidence = filters.get('min_confidence', self.confidence_threshold)
        filtered_df = filtered_df[
            (filtered_df['confidence'] >= min_confidence) | 
            (filtered_df['signal_type'] == 'SKIP')
        ]
        
        # Filter by minimum expected profit
        min_profit = filters.get('min_expected_profit', 0.5)
        filtered_df = filtered_df[
            (filtered_df['expected_profit'] >= min_profit) | 
            (filtered_df['signal_type'] == 'SKIP')
        ]
        
        # Filter by maximum position size
        max_position = filters.get('max_position_size', 3.0)
        filtered_df = filtered_df[filtered_df['position_size'] <= max_position]
        
        self.logger.info(f"Applied filters, remaining signals: {len(filtered_df)}")
        
        return filtered_df
    
    def get_signal_summary(self, signals_df: pd.DataFrame) -> Dict[str, any]:
        """Get summary statistics of generated signals."""
        summary = {
            'total_signals': len(signals_df),
            'buy_signals': len(signals_df[signals_df['signal_type'] == 'BUY']),
            'sell_signals': len(signals_df[signals_df['signal_type'] == 'SELL']),
            'skip_signals': len(signals_df[signals_df['signal_type'] == 'SKIP']),
            'avg_confidence': signals_df['confidence'].mean(),
            'avg_expected_profit': signals_df[signals_df['signal_type'] != 'SKIP']['expected_profit'].mean(),
            'total_position_size': signals_df[signals_df['signal_type'] != 'SKIP']['position_size'].sum()
        }
        
        return summary