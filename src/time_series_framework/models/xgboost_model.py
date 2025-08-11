"""
XGBoost implementation for time series forecasting.

This module implements XGBoost regression for time series prediction with
feature engineering, hyperparameter tuning, and comprehensive evaluation.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import time
from dataclasses import dataclass


@dataclass
class XGBoostResults:
    """Results container for XGBoost predictions."""
    predictions: np.ndarray
    feature_importance: Dict[str, float]
    training_metrics: Dict[str, float]
    training_time: float
    model_info: Dict[str, Any]


class XGBoostModel:
    """
    XGBoost model for time series forecasting.
    
    This class implements gradient boosting with tree-based ensemble learning
    for time series prediction with comprehensive feature engineering.
    """
    
    def __init__(self,
                 n_estimators: int = 25,
                 learning_rate: float = 0.5,
                 max_depth: int = 6,
                 sequence_length: int = 10,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 objective: str = 'reg:squarederror',
                 booster: str = 'gbtree',
                 train_split: float = 0.7):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage (eta)
            max_depth: Maximum tree depth
            sequence_length: Length of sequences for feature engineering
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            reg_alpha: L1 regularization term
            reg_lambda: L2 regularization term
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel threads
            objective: Learning objective
            booster: Booster type ('gbtree', 'gblinear', 'dart')
            train_split: Fraction of data for training
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.sequence_length = sequence_length
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.objective = objective
        self.booster = booster
        self.train_split = train_split
        
        # Model components
        self.model = None
        self.scaler = None
        self.feature_names = None
        
        # Initialize XGBoost and sklearn components
        self._init_components()
    
    def _init_components(self):
        """Initialize required components."""
        try:
            import xgboost as xgb
            from sklearn.preprocessing import RobustScaler
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            
            self.xgb = xgb
            self.RobustScaler = RobustScaler
            self.mean_squared_error = mean_squared_error
            self.r2_score = r2_score
            self.mean_absolute_error = mean_absolute_error
            self.available = True
            
        except ImportError:
            print("Warning: XGBoost not available. Install with: pip install xgboost")
            self.available = False
    
    def _create_time_features(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> np.ndarray:
        """Create time-based features from data."""
        features = []
        n = len(data)
        
        # Basic time index
        time_index = np.arange(n).reshape(-1, 1)
        features.append(time_index)
        
        # Lagged features
        for lag in range(1, min(self.sequence_length + 1, n)):
            lagged = np.roll(data, lag).reshape(-1, 1)
            lagged[:lag] = data[0]  # Forward fill initial values
            features.append(lagged)
        
        # Rolling statistics
        window_sizes = [3, 5, 7, 10]
        for window in window_sizes:
            if window < n:
                # Rolling mean
                rolling_mean = pd.Series(data).rolling(window, min_periods=1).mean().values.reshape(-1, 1)
                features.append(rolling_mean)
                
                # Rolling std
                rolling_std = pd.Series(data).rolling(window, min_periods=1).std().fillna(0).values.reshape(-1, 1)
                features.append(rolling_std)
        
        # Trend features
        linear_trend = np.linspace(0, 1, n).reshape(-1, 1)
        features.append(linear_trend)
        
        # Seasonal features (assuming monthly or daily patterns)
        if n > 12:
            seasonal_12 = np.sin(2 * np.pi * np.arange(n) / 12).reshape(-1, 1)
            seasonal_12_cos = np.cos(2 * np.pi * np.arange(n) / 12).reshape(-1, 1)
            features.extend([seasonal_12, seasonal_12_cos])
        
        if n > 7:
            seasonal_7 = np.sin(2 * np.pi * np.arange(n) / 7).reshape(-1, 1)
            seasonal_7_cos = np.cos(2 * np.pi * np.arange(n) / 7).reshape(-1, 1)
            features.extend([seasonal_7, seasonal_7_cos])
        
        # Combine all features
        X = np.hstack(features)
        
        # Create feature names
        self.feature_names = ['time_index']
        for lag in range(1, min(self.sequence_length + 1, n)):
            self.feature_names.append(f'lag_{lag}')
        
        for window in window_sizes:
            if window < n:
                self.feature_names.extend([f'rolling_mean_{window}', f'rolling_std_{window}'])
        
        self.feature_names.append('linear_trend')
        
        if n > 12:
            self.feature_names.extend(['seasonal_12_sin', 'seasonal_12_cos'])
        if n > 7:
            self.feature_names.extend(['seasonal_7_sin', 'seasonal_7_cos'])
        
        return X
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        n = len(X)
        split_idx = int(n * self.train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, data: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train XGBoost model on time series data.
        
        Args:
            data: Time series data
            timestamps: Optional timestamps for time-based features
            
        Returns:
            Dictionary containing training metrics
        """
        if not self.available:
            raise ImportError("XGBoost required for XGBoost model")
        
        start_time = time.time()
        
        # Scale target data
        self.scaler = self.RobustScaler()
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create features
        X = self._create_time_features(scaled_data, timestamps)
        y = scaled_data
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Create XGBoost regressor
        self.model = self.xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            objective=self.objective,
            booster=self.booster
        )
        
        # Fit model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_test, y_test)],
            verbose=False
        )
        
        training_time = time.time() - start_time
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(self.mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(self.mean_squared_error(y_test, y_test_pred))
        train_r2 = self.r2_score(y_train, y_train_pred)
        test_r2 = self.r2_score(y_test, y_test_pred)
        
        return {
            'training_time': training_time,
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_r2': float(train_r2),
            'test_r2': float(test_r2),
            'n_features': X.shape[1],
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
    
    def predict(self, data: np.ndarray, steps_ahead: int = 1) -> XGBoostResults:
        """
        Make predictions using trained XGBoost model.
        
        Args:
            data: Input time series data
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            XGBoostResults containing predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        predictions = []
        current_data = scaled_data.copy()
        
        for step in range(steps_ahead):
            # Create features for current data
            X_current = self._create_time_features(current_data)
            
            # Make prediction
            pred_scaled = self.model.predict(X_current[-1:])
            predictions.append(pred_scaled[0])
            
            # Update data with prediction for next step
            current_data = np.append(current_data, pred_scaled[0])
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        # Get feature importance
        feature_importance = {}
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                feature_importance[name] = float(importance)
        
        return XGBoostResults(
            predictions=predictions,
            feature_importance=feature_importance,
            training_metrics={},
            training_time=0.0,  # Set during fit
            model_info={
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'objective': self.objective,
                'booster': self.booster
            }
        )
    
    def evaluate(self, data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        results = self.predict(data, steps_ahead=len(target))
        predictions = results.predictions[:len(target)]
        
        return {
            'rmse': np.sqrt(self.mean_squared_error(target, predictions)),
            'mae': self.mean_absolute_error(target, predictions),
            'r2': self.r2_score(target, predictions),
            'mape': np.mean(np.abs((target - predictions) / target)) * 100
        }
    
    def get_feature_importance(self, plot: bool = False) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            plot: Whether to plot feature importance
            
        Returns:
            Dictionary of feature importance scores
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        importance_dict = {}
        if hasattr(self.model, 'feature_importances_') and self.feature_names:
            for name, importance in zip(self.feature_names, self.model.feature_importances_):
                importance_dict[name] = float(importance)
        
        if plot:
            try:
                import matplotlib.pyplot as plt
                
                # Sort by importance
                sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                features, importances = zip(*sorted_items)
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(features)), importances)
                plt.yticks(range(len(features)), features)
                plt.xlabel('Feature Importance')
                plt.title('XGBoost Feature Importance')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("Matplotlib required for plotting")
        
        return importance_dict
    
    def get_tree_splits(self) -> pd.DataFrame:
        """
        Get information about tree splits.
        
        Returns:
            DataFrame containing split information
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            booster = self.model.get_booster()
            df_tree = booster.trees_to_dataframe()
            
            # Filter for root nodes (Node == 0) to get most important splits
            root_splits = df_tree[df_tree.Node == 0].copy()
            root_splits = root_splits.sort_values('Gain', ascending=False)
            
            return root_splits
            
        except Exception as e:
            print(f"Error getting tree splits: {e}")
            return pd.DataFrame()
    
    def plot_tree_splits(self, n_splits: int = 5):
        """
        Plot the most important tree splits.
        
        Args:
            n_splits: Number of top splits to visualize
        """
        try:
            import matplotlib.pyplot as plt
            
            splits_df = self.get_tree_splits()
            if splits_df.empty:
                print("No split information available")
                return
            
            top_splits = splits_df.head(n_splits)
            
            plt.figure(figsize=(12, 8))
            
            # Plot gain by split value
            plt.subplot(2, 1, 1)
            plt.bar(range(len(top_splits)), top_splits['Gain'])
            plt.title('Top Tree Splits by Gain')
            plt.xlabel('Split Rank')
            plt.ylabel('Gain')
            plt.xticks(range(len(top_splits)), 
                      [f"Tree {t}, Split {s:.2f}" for t, s in zip(top_splits['Tree'], top_splits['Split'])])
            plt.xticks(rotation=45)
            
            # Plot split values over trees
            plt.subplot(2, 1, 2)
            plt.plot(top_splits['Tree'], top_splits['Split'], 'o-')
            plt.title('Split Values by Tree')
            plt.xlabel('Tree Number')
            plt.ylabel('Split Value')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")
    
    def visualize_trees(self, tree_index: int = 0):
        """
        Visualize a specific tree in the ensemble.
        
        Args:
            tree_index: Index of tree to visualize
        """
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        try:
            import matplotlib.pyplot as plt
            
            # Get booster and create graphviz representation
            booster = self.model.get_booster()
            graph = self.xgb.to_graphviz(
                booster, 
                num_trees=tree_index,
                rankdir='TB',
                feature_names=self.feature_names if self.feature_names else None
            )
            
            return graph
            
        except ImportError:
            print("Graphviz required for tree visualization")
            return None
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        self.model.save_model(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        if not self.available:
            raise ImportError("XGBoost required to load model")
        
        self.model = self.xgb.XGBRegressor()
        self.model.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def forecast_multi_step(self, data: np.ndarray, n_steps: int) -> np.ndarray:
        """
        Perform multi-step ahead forecasting.
        
        Args:
            data: Historical time series data
            n_steps: Number of steps to forecast
            
        Returns:
            Array of forecasted values
        """
        if self.model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        results = self.predict(data, steps_ahead=n_steps)
        return results.predictions
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        if self.model is None:
            raise ValueError("Model must be fitted first")
        
        summary = {
            'model_type': 'XGBoost Regressor',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'objective': self.objective,
            'booster': self.booster,
            'n_features': len(self.feature_names) if self.feature_names else 0,
            'feature_names': self.feature_names if self.feature_names else [],
            'regularization': {
                'reg_alpha': self.reg_alpha,
                'reg_lambda': self.reg_lambda
            },
            'sampling': {
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree
            }
        }
        
        return summary