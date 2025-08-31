import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from ..utils.logger import TradingLogger
from ..utils.helpers import ConfigManager

class RiskManager:
    """Manage trading risks and calculate stop-loss/take-profit levels."""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.logger = TradingLogger("RiskManager")
        self.risk_per_trade = self.config.get('trading.risk_per_trade', 0.02)
        self.stop_loss_multiplier = self.config.get('trading.stop_loss_multiplier', 2.0)
        self.take_profit_multiplier = self.config.get('trading.take_profit_multiplier', 3.0)
    
    def calculate_stop_loss_take_profit(self, 
                                      entry_price: float,
                                      signal_type: str,
                                      volatility: float,
                                      confidence: float) -> Tuple[float, float]:
        """Calculate stop-loss and take-profit levels."""
        
        # Base risk amount (percentage of entry price)
        base_risk = self.risk_per_trade
        
        # Adjust risk based on volatility
        volatility_adjusted_risk = base_risk * (1 + volatility)
        
        # Adjust based on confidence (higher confidence = tighter stops)
        confidence_adjustment = 2 - confidence  # Lower confidence = wider stops
        adjusted_risk = volatility_adjusted_risk * confidence_adjustment
        
        if signal_type == "BUY":
            stop_loss = entry_price * (1 - adjusted_risk * self.stop_loss_multiplier)
            take_profit = entry_price * (1 + adjusted_risk * self.take_profit_multiplier)
        elif signal_type == "SELL":
            stop_loss = entry_price * (1 + adjusted_risk * self.stop_loss_multiplier)
            take_profit = entry_price * (1 - adjusted_risk * self.take_profit_multiplier)
        else:
            stop_loss = entry_price
            take_profit = entry_price
        
        return stop_loss, take_profit
    
    def calculate_position_size(self, 
                              account_balance: float,
                              entry_price: float,
                              stop_loss: float,
                              risk_percentage: Optional[float] = None) -> float:
        """Calculate position size based on risk management rules."""
        
        if risk_percentage is None:
            risk_percentage = self.risk_per_trade
        
        # Maximum amount to risk
        risk_amount = account_balance * risk_percentage
        
        # Price difference to stop loss
        if stop_loss != entry_price:
            price_diff = abs(entry_price - stop_loss)
            position_size = risk_amount / price_diff
        else:
            # If no stop loss, use minimum position size
            position_size = account_balance * 0.01
        
        # Ensure position size doesn't exceed account balance
        max_position = account_balance * 0.95  # Leave 5% margin
        position_size = min(position_size, max_position)
        
        return position_size
    
    def assess_portfolio_risk(self, portfolio_data: Dict) -> Dict[str, float]:
        """Assess overall portfolio risk metrics."""
        
        total_exposure = portfolio_data.get('total_exposure', 0)
        account_balance = portfolio_data.get('account_balance', 1)
        open_positions = portfolio_data.get('open_positions', [])
        
        # Calculate various risk metrics
        risk_metrics = {
            'total_exposure_ratio': total_exposure / account_balance,
            'number_of_positions': len(open_positions),
            'max_single_position_risk': 0,
            'portfolio_var': 0,  # Value at Risk
            'sharpe_ratio': 0
        }
        
        if open_positions:
            # Calculate maximum single position risk
            position_risks = []
            for position in open_positions:
                entry_price = position.get('entry_price', 0)
                stop_loss = position.get('stop_loss', entry_price)
                position_size = position.get('position_size', 0)
                
                position_risk = abs(entry_price - stop_loss) * position_size / account_balance
                position_risks.append(position_risk)
            
            risk_metrics['max_single_position_risk'] = max(position_risks)
            risk_metrics['avg_position_risk'] = np.mean(position_risks)
            risk_metrics['total_portfolio_risk'] = sum(position_risks)
        
        return risk_metrics
    
    def should_take_trade(self, 
                         signal: Dict,
                         portfolio_metrics: Dict,
                         market_conditions: Dict) -> Tuple[bool, str]:
        """Determine if a trade should be taken based on risk assessment."""
        
        reasons = []
        
        # Check portfolio exposure
        total_exposure = portfolio_metrics.get('total_exposure_ratio', 0)
        if total_exposure > 0.8:  # More than 80% exposed
            return False, "Portfolio over-exposed"
        
        # Check number of open positions
        num_positions = portfolio_metrics.get('number_of_positions', 0)
        if num_positions > 10:  # Too many positions
            return False, "Too many open positions"
        
        # Check signal confidence
        confidence = signal.get('confidence', 0)
        if confidence < self.config.get('trading.min_confidence_threshold', 0.7):
            return False, "Signal confidence too low"
        
        # Check expected profit
        expected_profit = signal.get('expected_profit', 0)
        if expected_profit < 1.0:  # Less than 1% expected profit
            return False, "Expected profit too low"
        
        # Check market volatility
        volatility = market_conditions.get('volatility', 0)
        if volatility > 0.05:  # More than 5% volatility
            reasons.append("High volatility - reduced position size")
        
        return True, "; ".join(reasons) if reasons else "Trade approved"
    
    def update_stop_loss(self, 
                        position: Dict,
                        current_price: float,
                        trailing_percentage: float = 0.02) -> float:
        """Update stop-loss for trailing stop strategy."""
        
        entry_price = position.get('entry_price', current_price)
        current_stop = position.get('stop_loss', entry_price)
        signal_type = position.get('signal_type', 'BUY')
        
        if signal_type == "BUY":
            # For long positions, only move stop loss up
            trailing_stop = current_price * (1 - trailing_percentage)
            new_stop = max(current_stop, trailing_stop)
        else:
            # For short positions, only move stop loss down
            trailing_stop = current_price * (1 + trailing_percentage)
            new_stop = min(current_stop, trailing_stop)
        
        return new_stop