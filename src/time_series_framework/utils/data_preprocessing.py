"""
Data preprocessing utilities for time series analysis.

This module provides comprehensive data loading, cleaning, and preprocessing
functions for time series data used across all models in the framework.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PreprocessingResults:
    """Results container for preprocessing operations."""
    processed_data: np.ndarray
    original_data: np.ndarray
    scaler: Any
    metadata: Dict[str, Any]


class TimeSeriesProcessor:
    """
    Comprehensive time series data processor.
    
    This class handles data loading, cleaning, scaling, and transformation
    operations commonly needed for time series forecasting models.
    """
    
    def __init__(self, 
                 scaling_method: str = 'minmax',
                 handle_missing: str = 'interpolate',
                 detect_outliers: bool = True,
                 outlier_method: str = 'iqr'):
        """
        Initialize time series processor.
        
        Args:
            scaling_method: Scaling method ('minmax', 'standard', 'robust', 'none')
            handle_missing: Missing value handling ('interpolate', 'forward_fill', 'backward_fill', 'drop')
            detect_outliers: Whether to detect and handle outliers
            outlier_method: Outlier detection method ('iqr', 'zscore', 'isolation')
        """
        self.scaling_method = scaling_method
        self.handle_missing = handle_missing
        self.detect_outliers = detect_outliers
        self.outlier_method = outlier_method
        
        # Initialize scaling components
        self._init_scalers()
    
    def _init_scalers(self):
        """Initialize scaling components."""
        try:
            from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
            from sklearn.ensemble import IsolationForest
            
            self.MinMaxScaler = MinMaxScaler
            self.StandardScaler = StandardScaler
            self.RobustScaler = RobustScaler
            self.IsolationForest = IsolationForest
            self.sklearn_available = True
            
        except ImportError:
            print("Warning: scikit-learn not available for advanced preprocessing")
            self.sklearn_available = False
    
    def load_csv_data(self, 
                     filepath: Union[str, Path], 
                     target_column: str,
                     date_column: Optional[str] = None,
                     **kwargs) -> pd.DataFrame:
        """
        Load time series data from CSV file.
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            date_column: Name of date column (optional)
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        # Default CSV reading parameters
        default_params = {
            'engine': 'c',
            'low_memory': False
        }
        default_params.update(kwargs)
        
        # Load data
        df = pd.read_csv(filepath, **default_params)
        
        # Handle missing values in target
        if self.handle_missing == 'interpolate':
            df[target_column] = df[target_column].interpolate()
        elif self.handle_missing == 'forward_fill':
            df[target_column] = df[target_column].fillna(method='ffill')
        elif self.handle_missing == 'backward_fill':
            df[target_column] = df[target_column].fillna(method='bfill')
        elif self.handle_missing == 'drop':
            df = df.dropna(subset=[target_column])
        
        # Handle date column if provided
        if date_column and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df = df.set_index(date_column)
            df = df.sort_index()
        
        return df
    
    def detect_and_handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """
        Detect and handle outliers in time series data.
        
        Args:
            data: Input time series data
            
        Returns:
            Data with outliers handled
        """
        if not self.detect_outliers:
            return data
        
        data_clean = data.copy()
        
        if self.outlier_method == 'iqr':
            # Interquartile range method
            Q1 = np.percentile(data, 25)
            Q3 = np.percentile(data, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            data_clean = np.clip(data_clean, lower_bound, upper_bound)
            
        elif self.outlier_method == 'zscore':
            # Z-score method
            z_scores = np.abs((data - np.mean(data)) / np.std(data))
            threshold = 3.0
            
            # Replace outliers with mean
            outliers = z_scores > threshold
            data_clean[outliers] = np.mean(data[~outliers])
            
        elif self.outlier_method == 'isolation' and self.sklearn_available:
            # Isolation Forest method
            isolation_forest = self.IsolationForest(contamination=0.1, random_state=42)
            outliers = isolation_forest.fit_predict(data.reshape(-1, 1)) == -1
            
            # Replace outliers with median
            data_clean[outliers] = np.median(data[~outliers])
        
        return data_clean
    
    def create_scaler(self) -> Any:
        """Create appropriate scaler based on scaling method."""
        if not self.sklearn_available and self.scaling_method != 'none':
            print("Warning: scikit-learn not available, using no scaling")
            return None
        
        if self.scaling_method == 'minmax':
            return self.MinMaxScaler(feature_range=(0, 1))
        elif self.scaling_method == 'standard':
            return self.StandardScaler()
        elif self.scaling_method == 'robust':
            return self.RobustScaler()
        else:
            return None
    
    def scale_data(self, data: np.ndarray, scaler: Any = None, fit: bool = True) -> Tuple[np.ndarray, Any]:
        """
        Scale time series data.
        
        Args:
            data: Input data to scale
            scaler: Pre-fitted scaler (optional)
            fit: Whether to fit the scaler
            
        Returns:
            Tuple of (scaled_data, scaler)
        """
        if self.scaling_method == 'none':
            return data, None
        
        if scaler is None:
            scaler = self.create_scaler()
        
        data_reshaped = data.reshape(-1, 1)
        
        if fit:
            scaled_data = scaler.fit_transform(data_reshaped)
        else:
            scaled_data = scaler.transform(data_reshaped)
        
        return scaled_data.flatten(), scaler
    
    def inverse_scale_data(self, data: np.ndarray, scaler: Any) -> np.ndarray:
        """
        Inverse scale data back to original range.
        
        Args:
            data: Scaled data
            scaler: Fitted scaler
            
        Returns:
            Data in original scale
        """
        if scaler is None:
            return data
        
        data_reshaped = data.reshape(-1, 1)
        original_data = scaler.inverse_transform(data_reshaped)
        
        return original_data.flatten()
    
    def resample_data(self, 
                     df: pd.DataFrame, 
                     target_column: str,
                     frequency: str = 'M',
                     aggregation: str = 'mean') -> pd.DataFrame:
        """
        Resample time series data to different frequency.
        
        Args:
            df: DataFrame with datetime index
            target_column: Target column name
            frequency: Resampling frequency ('D', 'W', 'M', 'Q', 'Y')
            aggregation: Aggregation method ('mean', 'sum', 'min', 'max', 'median')
            
        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a datetime index for resampling")
        
        if aggregation == 'mean':
            return df[[target_column]].resample(frequency).mean()
        elif aggregation == 'sum':
            return df[[target_column]].resample(frequency).sum()
        elif aggregation == 'min':
            return df[[target_column]].resample(frequency).min()
        elif aggregation == 'max':
            return df[[target_column]].resample(frequency).max()
        elif aggregation == 'median':
            return df[[target_column]].resample(frequency).median()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def create_sequences(self, 
                        data: np.ndarray, 
                        sequence_length: int,
                        forecast_horizon: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for supervised learning.
        
        Args:
            data: Time series data
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to predict ahead
            
        Returns:
            Tuple of (X, y) arrays
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length - forecast_horizon + 1):
            X.append(data[i:(i + sequence_length)])
            
            if forecast_horizon == 1:
                y.append(data[i + sequence_length])
            else:
                y.append(data[(i + sequence_length):(i + sequence_length + forecast_horizon)])
        
        return np.array(X), np.array(y)
    
    def split_train_test(self, 
                        data: np.ndarray, 
                        train_ratio: float = 0.8,
                        method: str = 'temporal') -> Tuple[np.ndarray, np.ndarray]:
        """
        Split time series data into train and test sets.
        
        Args:
            data: Time series data
            train_ratio: Fraction of data for training
            method: Split method ('temporal', 'random')
            
        Returns:
            Tuple of (train_data, test_data)
        """
        split_point = int(len(data) * train_ratio)
        
        if method == 'temporal':
            # Preserve temporal order
            train_data = data[:split_point]
            test_data = data[split_point:]
        elif method == 'random':
            # Random split (not recommended for time series)
            indices = np.random.permutation(len(data))
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            train_data = data[train_indices]
            test_data = data[test_indices]
        else:
            raise ValueError(f"Unknown split method: {method}")
        
        return train_data, test_data
    
    def process_full_pipeline(self, 
                             data: Union[np.ndarray, pd.DataFrame],
                             target_column: Optional[str] = None) -> PreprocessingResults:
        """
        Execute full preprocessing pipeline.
        
        Args:
            data: Input data (array or DataFrame)
            target_column: Target column name (for DataFrame input)
            
        Returns:
            PreprocessingResults with all preprocessing outputs
        """
        # Extract array from DataFrame if needed
        if isinstance(data, pd.DataFrame):
            if target_column is None:
                raise ValueError("target_column required for DataFrame input")
            original_data = data[target_column].values
        else:
            original_data = data.copy()
        
        # Step 1: Handle outliers
        cleaned_data = self.detect_and_handle_outliers(original_data)
        
        # Step 2: Scale data
        scaled_data, scaler = self.scale_data(cleaned_data)
        
        # Collect metadata
        metadata = {
            'original_shape': original_data.shape,
            'processed_shape': scaled_data.shape,
            'scaling_method': self.scaling_method,
            'outlier_method': self.outlier_method if self.detect_outliers else None,
            'missing_handling': self.handle_missing,
            'outliers_detected': not np.array_equal(original_data, cleaned_data),
            'data_stats': {
                'original_mean': float(np.mean(original_data)),
                'original_std': float(np.std(original_data)),
                'original_min': float(np.min(original_data)),
                'original_max': float(np.max(original_data)),
                'processed_mean': float(np.mean(scaled_data)),
                'processed_std': float(np.std(scaled_data)),
                'processed_min': float(np.min(scaled_data)),
                'processed_max': float(np.max(scaled_data))
            }
        }
        
        return PreprocessingResults(
            processed_data=scaled_data,
            original_data=original_data,
            scaler=scaler,
            metadata=metadata
        )
    
    def get_data_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """
        Get comprehensive statistics for time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary of statistics
        """
        return {
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'var': float(np.var(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'skewness': float(self._calculate_skewness(data)),
            'kurtosis': float(self._calculate_kurtosis(data)),
            'length': int(len(data))
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def plot_data_overview(self, 
                          data: np.ndarray, 
                          title: str = "Time Series Data Overview"):
        """
        Plot comprehensive data overview.
        
        Args:
            data: Time series data
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Time series plot
            axes[0, 0].plot(data)
            axes[0, 0].set_title('Time Series')
            axes[0, 0].set_xlabel('Time')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Histogram
            axes[0, 1].hist(data, bins=50, alpha=0.7, edgecolor='black')
            axes[0, 1].set_title('Distribution')
            axes[0, 1].set_xlabel('Value')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Box plot
            axes[1, 0].boxplot(data)
            axes[1, 0].set_title('Box Plot')
            axes[1, 0].set_ylabel('Value')
            axes[1, 0].grid(True, alpha=0.3)
            
            # ACF plot (simple version)
            lags = range(min(50, len(data) // 4))
            acf_values = [self._calculate_autocorr(data, lag) for lag in lags]
            axes[1, 1].plot(lags, acf_values)
            axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
            axes[1, 1].axhline(y=0.05, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].axhline(y=-0.05, color='r', linestyle='--', alpha=0.5)
            axes[1, 1].set_title('Autocorrelation')
            axes[1, 1].set_xlabel('Lag')
            axes[1, 1].set_ylabel('ACF')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")
    
    def _calculate_autocorr(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if lag >= len(data):
            return 0.0
        
        data_centered = data - np.mean(data)
        
        if lag == 0:
            return 1.0
        
        c0 = np.sum(data_centered ** 2) / len(data)
        ck = np.sum(data_centered[:-lag] * data_centered[lag:]) / len(data)
        
        return ck / c0 if c0 != 0 else 0.0