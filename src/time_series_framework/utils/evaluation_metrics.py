"""
Comprehensive evaluation metrics for time series forecasting models.

This module provides specialized metrics and evaluation functions for
assessing the performance of time series forecasting models.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings


@dataclass
class EvaluationResults:
    """Container for comprehensive evaluation results."""
    point_metrics: Dict[str, float]
    probabilistic_metrics: Dict[str, float]
    residual_analysis: Dict[str, Any]
    forecast_accuracy: Dict[str, float]
    statistical_tests: Dict[str, Any]


class TimeSeriesMetrics:
    """
    Comprehensive time series evaluation metrics.
    
    This class provides both traditional and advanced metrics specifically
    designed for time series forecasting evaluation.
    """
    
    def __init__(self):
        """Initialize metrics calculator."""
        self._init_components()
    
    def _init_components(self):
        """Initialize required statistical components."""
        try:
            from scipy import stats
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            
            self.stats = stats
            self.mean_squared_error = mean_squared_error
            self.mean_absolute_error = mean_absolute_error
            self.r2_score = r2_score
            self.scipy_available = True
            
        except ImportError:
            print("Warning: SciPy/scikit-learn not available for advanced metrics")
            self.scipy_available = False
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Root Mean Squared Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            RMSE value
        """
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAE value
        """
        return np.mean(np.abs(y_true - y_pred))
    
    def mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            MAPE value (as percentage)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mask = y_true != 0
            if not np.any(mask):
                return np.inf
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    def smape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Symmetric Mean Absolute Percentage Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            sMAPE value (as percentage)
        """
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not np.any(mask):
            return 0.0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def mase(self, y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray) -> float:
        """
        Mean Absolute Scaled Error.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data for scaling
            
        Returns:
            MASE value
        """
        # Calculate naive forecast MAE (seasonal naive with period=1)
        naive_mae = np.mean(np.abs(y_train[1:] - y_train[:-1]))
        
        if naive_mae == 0:
            return np.inf if np.any(y_true != y_pred) else 0.0
        
        return self.mae(y_true, y_pred) / naive_mae
    
    def r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        R-squared (coefficient of determination).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            R² value
        """
        if self.scipy_available:
            return float(self.r2_score(y_true, y_pred))
        else:
            # Manual calculation
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    def directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Directional Accuracy - percentage of correctly predicted directions.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Directional accuracy (as percentage)
        """
        if len(y_true) < 2:
            return np.nan
        
        true_direction = np.diff(y_true) >= 0
        pred_direction = np.diff(y_pred) >= 0
        
        return np.mean(true_direction == pred_direction) * 100
    
    def theil_u(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Theil's U statistic (U2).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Theil's U statistic
        """
        if len(y_true) < 2:
            return np.nan
        
        # Calculate relative changes
        actual_changes = (y_true[1:] - y_true[:-1]) / y_true[:-1]
        predicted_changes = (y_pred[1:] - y_pred[:-1]) / y_true[:-1]
        
        # Handle division by zero
        mask = y_true[:-1] != 0
        if not np.any(mask):
            return np.inf
        
        actual_changes = actual_changes[mask]
        predicted_changes = predicted_changes[mask]
        
        numerator = np.sqrt(np.mean((actual_changes - predicted_changes) ** 2))
        denominator = np.sqrt(np.mean(actual_changes ** 2))
        
        return numerator / denominator if denominator != 0 else np.inf
    
    def forecast_bias(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Forecast Bias (Mean Error).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Forecast bias
        """
        return np.mean(y_pred - y_true)
    
    def normalized_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Normalized RMSE (by the range of true values).
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Normalized RMSE
        """
        rmse_val = self.rmse(y_true, y_pred)
        range_val = np.max(y_true) - np.min(y_true)
        
        return rmse_val / range_val if range_val != 0 else np.inf
    
    def log_likelihood(self, 
                      y_true: np.ndarray, 
                      y_pred_mean: np.ndarray, 
                      y_pred_std: np.ndarray) -> float:
        """
        Negative log-likelihood for probabilistic forecasts.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_std: Predicted standard deviations
            
        Returns:
            Negative log-likelihood
        """
        if not self.scipy_available:
            print("Warning: SciPy required for log-likelihood calculation")
            return np.nan
        
        # Avoid numerical issues
        y_pred_std = np.maximum(y_pred_std, 1e-8)
        
        # Calculate log-likelihood
        log_probs = self.stats.norm.logpdf(y_true, y_pred_mean, y_pred_std)
        
        return -np.mean(log_probs)
    
    def coverage_probability(self, 
                           y_true: np.ndarray, 
                           y_pred_lower: np.ndarray, 
                           y_pred_upper: np.ndarray) -> float:
        """
        Coverage probability for prediction intervals.
        
        Args:
            y_true: True values
            y_pred_lower: Lower bounds of prediction intervals
            y_pred_upper: Upper bounds of prediction intervals
            
        Returns:
            Coverage probability (as percentage)
        """
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        return coverage * 100
    
    def interval_width(self, 
                      y_pred_lower: np.ndarray, 
                      y_pred_upper: np.ndarray) -> Dict[str, float]:
        """
        Calculate prediction interval width statistics.
        
        Args:
            y_pred_lower: Lower bounds of prediction intervals
            y_pred_upper: Upper bounds of prediction intervals
            
        Returns:
            Dictionary with width statistics
        """
        widths = y_pred_upper - y_pred_lower
        
        return {
            'mean_width': float(np.mean(widths)),
            'median_width': float(np.median(widths)),
            'std_width': float(np.std(widths)),
            'min_width': float(np.min(widths)),
            'max_width': float(np.max(widths))
        }
    
    def residual_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Comprehensive residual analysis.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary with residual statistics and tests
        """
        residuals = y_true - y_pred
        
        results = {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'skewness_residual': float(self._calculate_skewness(residuals)),
            'kurtosis_residual': float(self._calculate_kurtosis(residuals)),
            'min_residual': float(np.min(residuals)),
            'max_residual': float(np.max(residuals)),
            'autocorrelation': self._calculate_residual_autocorrelation(residuals)
        }
        
        # Statistical tests if scipy is available
        if self.scipy_available:
            # Ljung-Box test for autocorrelation
            results['ljung_box_test'] = self._ljung_box_test(residuals)
            
            # Jarque-Bera test for normality
            results['jarque_bera_test'] = self._jarque_bera_test(residuals)
            
            # Durbin-Watson test for autocorrelation
            results['durbin_watson'] = self._durbin_watson_test(residuals)
        
        return results
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _calculate_residual_autocorrelation(self, residuals: np.ndarray, max_lags: int = 10) -> Dict[str, float]:
        """Calculate autocorrelation of residuals."""
        n_lags = min(max_lags, len(residuals) // 4)
        autocorr = {}
        
        for lag in range(1, n_lags + 1):
            if lag < len(residuals):
                corr = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                autocorr[f'lag_{lag}'] = float(corr) if not np.isnan(corr) else 0.0
        
        return autocorr
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation."""
        if not self.scipy_available:
            return {'statistic': np.nan, 'p_value': np.nan}
        
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            result = acorr_ljungbox(residuals, lags=lags, return_df=False)
            return {'statistic': float(result[0][-1]), 'p_value': float(result[1][-1])}
        except ImportError:
            # Manual implementation
            n = len(residuals)
            acf_values = []
            
            for lag in range(1, lags + 1):
                if lag < n:
                    acf = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    acf_values.append(acf if not np.isnan(acf) else 0.0)
            
            # Ljung-Box statistic
            statistic = n * (n + 2) * np.sum([acf**2 / (n - lag) for lag, acf in enumerate(acf_values, 1)])
            p_value = 1 - self.stats.chi2.cdf(statistic, lags)
            
            return {'statistic': float(statistic), 'p_value': float(p_value)}
    
    def _jarque_bera_test(self, residuals: np.ndarray) -> Dict[str, float]:
        """Jarque-Bera test for normality."""
        if not self.scipy_available:
            return {'statistic': np.nan, 'p_value': np.nan}
        
        try:
            statistic, p_value = self.stats.jarque_bera(residuals)
            return {'statistic': float(statistic), 'p_value': float(p_value)}
        except:
            return {'statistic': np.nan, 'p_value': np.nan}
    
    def _durbin_watson_test(self, residuals: np.ndarray) -> float:
        """Durbin-Watson test for autocorrelation."""
        if len(residuals) < 2:
            return np.nan
        
        diff_residuals = np.diff(residuals)
        statistic = np.sum(diff_residuals ** 2) / np.sum(residuals ** 2)
        
        return float(statistic)
    
    def calculate_all_metrics(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray,
                            y_train: Optional[np.ndarray] = None,
                            y_pred_std: Optional[np.ndarray] = None,
                            confidence_lower: Optional[np.ndarray] = None,
                            confidence_upper: Optional[np.ndarray] = None) -> EvaluationResults:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            y_train: Training data (for MASE calculation)
            y_pred_std: Prediction standard deviations (for probabilistic metrics)
            confidence_lower: Lower confidence bounds
            confidence_upper: Upper confidence bounds
            
        Returns:
            EvaluationResults with all metrics
        """
        # Point forecast metrics
        point_metrics = {
            'rmse': self.rmse(y_true, y_pred),
            'mae': self.mae(y_true, y_pred),
            'mape': self.mape(y_true, y_pred),
            'smape': self.smape(y_true, y_pred),
            'r2': self.r2(y_true, y_pred),
            'forecast_bias': self.forecast_bias(y_true, y_pred),
            'normalized_rmse': self.normalized_rmse(y_true, y_pred),
            'directional_accuracy': self.directional_accuracy(y_true, y_pred),
            'theil_u': self.theil_u(y_true, y_pred)
        }
        
        # Add MASE if training data provided
        if y_train is not None:
            point_metrics['mase'] = self.mase(y_true, y_pred, y_train)
        
        # Probabilistic metrics
        probabilistic_metrics = {}
        if y_pred_std is not None:
            probabilistic_metrics['log_likelihood'] = self.log_likelihood(y_true, y_pred, y_pred_std)
        
        if confidence_lower is not None and confidence_upper is not None:
            probabilistic_metrics['coverage_probability'] = self.coverage_probability(
                y_true, confidence_lower, confidence_upper
            )
            probabilistic_metrics.update(self.interval_width(confidence_lower, confidence_upper))
        
        # Residual analysis
        residual_analysis = self.residual_analysis(y_true, y_pred)
        
        # Forecast accuracy classification
        forecast_accuracy = self._classify_forecast_accuracy(point_metrics)
        
        # Statistical tests
        statistical_tests = {
            'residual_tests': residual_analysis.get('ljung_box_test', {}),
            'normality_test': residual_analysis.get('jarque_bera_test', {}),
            'durbin_watson': residual_analysis.get('durbin_watson', np.nan)
        }
        
        return EvaluationResults(
            point_metrics=point_metrics,
            probabilistic_metrics=probabilistic_metrics,
            residual_analysis=residual_analysis,
            forecast_accuracy=forecast_accuracy,
            statistical_tests=statistical_tests
        )
    
    def _classify_forecast_accuracy(self, metrics: Dict[str, float]) -> Dict[str, str]:
        """Classify forecast accuracy based on common thresholds."""
        mape = metrics.get('mape', np.inf)
        
        # MAPE-based classification
        if mape <= 10:
            mape_class = 'Excellent'
        elif mape <= 20:
            mape_class = 'Good'
        elif mape <= 50:
            mape_class = 'Acceptable'
        else:
            mape_class = 'Poor'
        
        # Theil's U classification
        theil_u = metrics.get('theil_u', np.inf)
        if theil_u < 1:
            theil_class = 'Better than naive'
        elif theil_u == 1:
            theil_class = 'Same as naive'
        else:
            theil_class = 'Worse than naive'
        
        return {
            'mape_classification': mape_class,
            'theil_u_classification': theil_class,
            'overall_quality': self._determine_overall_quality(metrics)
        }
    
    def _determine_overall_quality(self, metrics: Dict[str, float]) -> str:
        """Determine overall forecast quality."""
        r2 = metrics.get('r2', 0)
        mape = metrics.get('mape', np.inf)
        
        if r2 > 0.8 and mape < 15:
            return 'Excellent'
        elif r2 > 0.6 and mape < 25:
            return 'Good'
        elif r2 > 0.4 and mape < 40:
            return 'Acceptable'
        else:
            return 'Poor'
    
    def plot_evaluation_results(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               title: str = "Model Evaluation"):
        """
        Create comprehensive evaluation plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
        """
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Actual vs Predicted
            axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
            min_val = min(np.min(y_true), np.min(y_pred))
            max_val = max(np.max(y_true), np.max(y_pred))
            axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Time series plot
            axes[0, 1].plot(y_true, label='Actual', linewidth=2)
            axes[0, 1].plot(y_pred, label='Predicted', linewidth=2)
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].set_title('Time Series Comparison')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Residuals
            residuals = y_true - y_pred
            axes[1, 0].plot(residuals, alpha=0.7)
            axes[1, 0].axhline(y=0, color='r', linestyle='--')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Residual')
            axes[1, 0].set_title('Residuals')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Residual histogram
            axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].set_xlabel('Residual')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Residual Distribution')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle(title)
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")