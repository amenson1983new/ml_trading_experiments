import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from typing import Dict, Any, List, Optional, Tuple
from .base_model import BaseModel

class EnsembleModel(BaseModel):
    """Ensemble model combining multiple ML algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Ensemble", config)
        self.classification_models = {}
        self.regression_models = {}
        
    def build_model(self, task_type: str = "both") -> None:
        """Build ensemble of models for classification and/or regression."""
        self.logger.info(f"Building ensemble model for {task_type}")
        
        if task_type in ["classification", "both"]:
            self.classification_models = {
                'random_forest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                ),
                'svm': SVC(
                    kernel='rbf',
                    probability=True,
                    random_state=42
                )
            }
        
        if task_type in ["regression", "both"]:
            self.regression_models = {
                'random_forest': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                ),
                'svr': SVR(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1
                )
            }
        
        self.logger.info("Ensemble models built successfully")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              task_type: str = "classification") -> Dict[str, Any]:
        """Train ensemble models."""
        self.logger.info(f"Training ensemble models for {task_type}")
        
        results = {}
        
        if task_type == "classification" and self.classification_models:
            for name, model in self.classification_models.items():
                self.logger.info(f"Training {name} classifier")
                model.fit(X_train, y_train)
                
                # Evaluate on validation set if provided
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_accuracy = accuracy_score(y_val, val_pred)
                    results[f"{name}_val_accuracy"] = val_accuracy
                    self.logger.info(f"{name} validation accuracy: {val_accuracy:.4f}")
        
        elif task_type == "regression" and self.regression_models:
            for name, model in self.regression_models.items():
                self.logger.info(f"Training {name} regressor")
                model.fit(X_train, y_train)
                
                # Evaluate on validation set if provided
                if X_val is not None and y_val is not None:
                    val_pred = model.predict(X_val)
                    val_mse = mean_squared_error(y_val, val_pred)
                    val_r2 = r2_score(y_val, val_pred)
                    results[f"{name}_val_mse"] = val_mse
                    results[f"{name}_val_r2"] = val_r2
                    self.logger.info(f"{name} validation MSE: {val_mse:.4f}, R2: {val_r2:.4f}")
        
        self.is_trained = True
        self.logger.info("Ensemble training completed")
        
        return results
    
    def predict(self, X: np.ndarray, task_type: str = "classification") -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        predictions = []
        
        if task_type == "classification":
            for name, model in self.classification_models.items():
                pred = model.predict_proba(X)
                predictions.append(pred)
        else:
            for name, model in self.regression_models.items():
                pred = model.predict(X)
                predictions.append(pred)
        
        # Ensemble by averaging
        ensemble_pred = np.mean(predictions, axis=0)
        
        if task_type == "classification":
            return np.argmax(ensemble_pred, axis=1)
        else:
            return ensemble_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities for classification."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        probabilities = []
        for name, model in self.classification_models.items():
            prob = model.predict_proba(X)
            probabilities.append(prob)
        
        return np.mean(probabilities, axis=0)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, task_type: str = "classification") -> Dict[str, float]:
        """Evaluate ensemble performance."""
        if not self.is_trained:
            raise ValueError("Models not trained. Call train() first.")
        
        results = {}
        
        if task_type == "classification":
            predictions = self.predict(X_test, task_type)
            accuracy = accuracy_score(y_test, predictions)
            results['ensemble_accuracy'] = accuracy
            
            # Individual model performances
            for name, model in self.classification_models.items():
                pred = model.predict(X_test)
                acc = accuracy_score(y_test, pred)
                results[f"{name}_accuracy"] = acc
        
        else:
            predictions = self.predict(X_test, task_type)
            mse = mean_squared_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            results['ensemble_mse'] = mse
            results['ensemble_r2'] = r2
            
            # Individual model performances
            for name, model in self.regression_models.items():
                pred = model.predict(X_test)
                mse_ind = mean_squared_error(y_test, pred)
                r2_ind = r2_score(y_test, pred)
                results[f"{name}_mse"] = mse_ind
                results[f"{name}_r2"] = r2_ind
        
        return results