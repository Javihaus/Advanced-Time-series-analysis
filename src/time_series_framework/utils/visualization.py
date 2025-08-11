"""
Advanced visualization utilities for time series analysis and model comparison.

This module provides comprehensive plotting functions for time series data,
model predictions, performance comparisons, and statistical analysis.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PlotConfig:
    """Configuration for plot styling and layout."""
    figsize: Tuple[int, int] = (12, 8)
    colors: List[str] = None
    style: str = 'default'
    grid: bool = True
    grid_alpha: float = 0.3
    legend: bool = True
    title_fontsize: int = 14
    label_fontsize: int = 12
    
    def __post_init__(self):
        if self.colors is None:
            self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


class TimeSeriesVisualizer:
    """
    Advanced visualization toolkit for time series analysis.
    
    This class provides comprehensive plotting capabilities for time series data,
    model predictions, comparisons, and statistical analysis.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            config: Plot configuration options
        """
        self.config = config or PlotConfig()
        self._init_plotting_backend()
    
    def _init_plotting_backend(self):
        """Initialize plotting libraries."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.patches import Rectangle
            import seaborn as sns
            
            self.plt = plt
            self.mdates = mdates
            self.Rectangle = Rectangle
            self.sns = sns
            self.matplotlib_available = True
            
            # Set style
            if self.config.style == 'seaborn':
                self.sns.set_style("whitegrid")
            
        except ImportError:
            print("Warning: matplotlib/seaborn not available for plotting")
            self.matplotlib_available = False
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            
            self.go = go
            self.px = px
            self.make_subplots = make_subplots
            self.plotly_available = True
            
        except ImportError:
            print("Warning: plotly not available for interactive plotting")
            self.plotly_available = False
    
    def plot_time_series(self, 
                        data: Union[np.ndarray, pd.Series, pd.DataFrame],
                        timestamps: Optional[np.ndarray] = None,
                        title: str = "Time Series Data",
                        xlabel: str = "Time",
                        ylabel: str = "Value",
                        interactive: bool = False) -> None:
        """
        Plot time series data.
        
        Args:
            data: Time series data
            timestamps: Optional timestamps
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            interactive: Use interactive plotting
        """
        if interactive and self.plotly_available:
            self._plot_time_series_plotly(data, timestamps, title, xlabel, ylabel)
        elif self.matplotlib_available:
            self._plot_time_series_matplotlib(data, timestamps, title, xlabel, ylabel)
        else:
            print("No plotting backend available")
    
    def _plot_time_series_matplotlib(self, data, timestamps, title, xlabel, ylabel):
        """Plot time series using matplotlib."""
        fig, ax = self.plt.subplots(figsize=self.config.figsize)
        
        if timestamps is not None:
            ax.plot(timestamps, data, color=self.config.colors[0], linewidth=2)
        else:
            ax.plot(data, color=self.config.colors[0], linewidth=2)
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xlabel(xlabel, fontsize=self.config.label_fontsize)
        ax.set_ylabel(ylabel, fontsize=self.config.label_fontsize)
        
        if self.config.grid:
            ax.grid(True, alpha=self.config.grid_alpha)
        
        self.plt.tight_layout()
        self.plt.show()
    
    def _plot_time_series_plotly(self, data, timestamps, title, xlabel, ylabel):
        """Plot time series using plotly."""
        fig = self.go.Figure()
        
        x_data = timestamps if timestamps is not None else np.arange(len(data))
        
        fig.add_trace(self.go.Scatter(
            x=x_data,
            y=data,
            mode='lines',
            name='Time Series',
            line=dict(color=self.config.colors[0], width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=self.config.legend,
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_predictions_comparison(self,
                                   y_true: np.ndarray,
                                   predictions: Dict[str, np.ndarray],
                                   timestamps: Optional[np.ndarray] = None,
                                   title: str = "Model Predictions Comparison",
                                   split_point: Optional[int] = None,
                                   interactive: bool = False) -> None:
        """
        Plot comparison of multiple model predictions.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions {model_name: predictions}
            timestamps: Optional timestamps
            title: Plot title
            split_point: Index where training ends and testing begins
            interactive: Use interactive plotting
        """
        if interactive and self.plotly_available:
            self._plot_predictions_plotly(y_true, predictions, timestamps, title, split_point)
        elif self.matplotlib_available:
            self._plot_predictions_matplotlib(y_true, predictions, timestamps, title, split_point)
        else:
            print("No plotting backend available")
    
    def _plot_predictions_matplotlib(self, y_true, predictions, timestamps, title, split_point):
        """Plot predictions comparison using matplotlib."""
        fig, ax = self.plt.subplots(figsize=self.config.figsize)
        
        x_data = timestamps if timestamps is not None else np.arange(len(y_true))
        
        # Plot true values
        ax.plot(x_data, y_true, label='Actual', color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color = self.config.colors[i % len(self.config.colors)]
            ax.plot(x_data[:len(y_pred)], y_pred, 
                   label=f'{model_name}', color=color, linewidth=2, alpha=0.7)
        
        # Add vertical line at split point
        if split_point is not None:
            ax.axvline(x=x_data[split_point], color='red', linestyle='--', alpha=0.5,
                      label='Train/Test Split')
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xlabel('Time', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Value', fontsize=self.config.label_fontsize)
        
        if self.config.legend:
            ax.legend()
        if self.config.grid:
            ax.grid(True, alpha=self.config.grid_alpha)
        
        self.plt.tight_layout()
        self.plt.show()
    
    def _plot_predictions_plotly(self, y_true, predictions, timestamps, title, split_point):
        """Plot predictions comparison using plotly."""
        fig = self.go.Figure()
        
        x_data = timestamps if timestamps is not None else np.arange(len(y_true))
        
        # Plot true values
        fig.add_trace(self.go.Scatter(
            x=x_data,
            y=y_true,
            mode='lines',
            name='Actual',
            line=dict(color='black', width=3),
            opacity=0.8
        ))
        
        # Plot predictions
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color = self.config.colors[i % len(self.config.colors)]
            fig.add_trace(self.go.Scatter(
                x=x_data[:len(y_pred)],
                y=y_pred,
                mode='lines',
                name=model_name,
                line=dict(color=color, width=2),
                opacity=0.7
            ))
        
        # Add vertical line at split point
        if split_point is not None:
            fig.add_vline(
                x=x_data[split_point],
                line_dash="dash",
                line_color="red",
                annotation_text="Train/Test Split"
            )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time',
            yaxis_title='Value',
            showlegend=self.config.legend,
            template='plotly_white'
        )
        
        fig.show()
    
    def plot_uncertainty_bands(self,
                              y_true: np.ndarray,
                              y_pred_mean: np.ndarray,
                              y_pred_std: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              confidence_levels: List[float] = [0.68, 0.95],
                              title: str = "Predictions with Uncertainty",
                              interactive: bool = False) -> None:
        """
        Plot predictions with uncertainty bands.
        
        Args:
            y_true: True values
            y_pred_mean: Predicted means
            y_pred_std: Predicted standard deviations
            timestamps: Optional timestamps
            confidence_levels: Confidence levels for bands
            title: Plot title
            interactive: Use interactive plotting
        """
        if not self.matplotlib_available:
            print("Matplotlib required for uncertainty plots")
            return
        
        fig, ax = self.plt.subplots(figsize=self.config.figsize)
        
        x_data = timestamps if timestamps is not None else np.arange(len(y_pred_mean))
        
        # Plot true values
        ax.plot(x_data[:len(y_true)], y_true, 'o-', label='Actual', 
                color='black', markersize=4, alpha=0.8)
        
        # Plot predicted mean
        ax.plot(x_data, y_pred_mean, '-', label='Predicted Mean', 
                color=self.config.colors[0], linewidth=2)
        
        # Plot confidence bands
        z_scores = [0.674, 1.96]  # 68% and 95% confidence levels
        alphas = [0.3, 0.2]
        
        for i, (confidence, z_score, alpha) in enumerate(zip(confidence_levels, z_scores, alphas)):
            lower = y_pred_mean - z_score * y_pred_std
            upper = y_pred_mean + z_score * y_pred_std
            
            ax.fill_between(x_data, lower, upper, 
                           alpha=alpha, 
                           color=self.config.colors[0],
                           label=f'{int(confidence*100)}% Confidence')
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xlabel('Time', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Value', fontsize=self.config.label_fontsize)
        
        if self.config.legend:
            ax.legend()
        if self.config.grid:
            ax.grid(True, alpha=self.config.grid_alpha)
        
        self.plt.tight_layout()
        self.plt.show()
    
    def plot_model_performance_comparison(self,
                                        metrics: Dict[str, Dict[str, float]],
                                        metric_names: List[str] = ['rmse', 'mae', 'r2'],
                                        title: str = "Model Performance Comparison") -> None:
        """
        Plot comparison of model performance metrics.
        
        Args:
            metrics: Dictionary of {model_name: {metric_name: value}}
            metric_names: List of metrics to plot
            title: Plot title
        """
        if not self.matplotlib_available:
            print("Matplotlib required for performance comparison plots")
            return
        
        n_metrics = len(metric_names)
        fig, axes = self.plt.subplots(1, n_metrics, figsize=(5*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        model_names = list(metrics.keys())
        
        for i, metric in enumerate(metric_names):
            values = [metrics[model].get(metric, 0) for model in model_names]
            colors = self.config.colors[:len(model_names)]
            
            bars = axes[i].bar(model_names, values, color=colors, alpha=0.7)
            axes[i].set_title(f'{metric.upper()}', fontsize=self.config.title_fontsize)
            axes[i].set_ylabel('Value', fontsize=self.config.label_fontsize)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            if self.config.grid:
                axes[i].grid(True, alpha=self.config.grid_alpha, axis='y')
            
            # Rotate x-axis labels if needed
            if len(model_names) > 3:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=self.config.title_fontsize + 2)
        plt.tight_layout()
        plt.show()
    
    def plot_residual_analysis(self,
                              residuals: np.ndarray,
                              timestamps: Optional[np.ndarray] = None,
                              title: str = "Residual Analysis") -> None:
        """
        Create comprehensive residual analysis plots.
        
        Args:
            residuals: Model residuals
            timestamps: Optional timestamps
            title: Plot title
        """
        if not self.matplotlib_available:
            print("Matplotlib required for residual analysis")
            return
        
        fig, axes = self.plt.subplots(2, 2, figsize=(15, 10))
        
        x_data = timestamps if timestamps is not None else np.arange(len(residuals))
        
        # Residuals over time
        axes[0, 0].plot(x_data, residuals, color=self.config.colors[0], alpha=0.7)
        axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Residual')
        if self.config.grid:
            axes[0, 0].grid(True, alpha=self.config.grid_alpha)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, alpha=0.7, color=self.config.colors[1], 
                       edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 1].set_title('Residual Distribution')
        axes[0, 1].set_xlabel('Residual')
        axes[0, 1].set_ylabel('Frequency')
        if self.config.grid:
            axes[0, 1].grid(True, alpha=self.config.grid_alpha)
        
        # Q-Q plot
        try:
            from scipy import stats
            stats.probplot(residuals, dist="norm", plot=axes[1, 0])
            axes[1, 0].set_title('Q-Q Plot')
            axes[1, 0].grid(True, alpha=self.config.grid_alpha)
        except ImportError:
            axes[1, 0].text(0.5, 0.5, 'SciPy required for Q-Q plot', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # Autocorrelation plot
        max_lags = min(50, len(residuals) // 4)
        lags = range(max_lags)
        autocorr = [self._calculate_autocorr(residuals, lag) for lag in lags]
        
        axes[1, 1].plot(lags, autocorr, 'o-', color=self.config.colors[2])
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_title('Residual Autocorrelation')
        axes[1, 1].set_xlabel('Lag')
        axes[1, 1].set_ylabel('Autocorrelation')
        if self.config.grid:
            axes[1, 1].grid(True, alpha=self.config.grid_alpha)
        
        plt.suptitle(title, fontsize=self.config.title_fontsize + 2)
        plt.tight_layout()
        plt.show()
    
    def _calculate_autocorr(self, data: np.ndarray, lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if lag >= len(data) or lag == 0:
            return 1.0 if lag == 0 else 0.0
        
        data_centered = data - np.mean(data)
        c0 = np.sum(data_centered ** 2) / len(data)
        ck = np.sum(data_centered[:-lag] * data_centered[lag:]) / len(data)
        
        return ck / c0 if c0 != 0 else 0.0
    
    def plot_feature_importance(self,
                               feature_importance: Dict[str, float],
                               title: str = "Feature Importance",
                               top_k: int = 20) -> None:
        """
        Plot feature importance.
        
        Args:
            feature_importance: Dictionary of feature importances
            title: Plot title
            top_k: Number of top features to show
        """
        if not self.matplotlib_available:
            print("Matplotlib required for feature importance plots")
            return
        
        # Sort by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        sorted_features = sorted_features[:top_k]
        
        features, importances = zip(*sorted_features)
        
        fig, ax = self.plt.subplots(figsize=(10, max(6, len(features) * 0.3)))
        
        bars = ax.barh(range(len(features)), importances, color=self.config.colors[0], alpha=0.7)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=self.config.title_fontsize)
        
        # Add value labels
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            ax.text(importance + max(importances) * 0.01, i,
                   f'{importance:.3f}', va='center')
        
        if self.config.grid:
            ax.grid(True, alpha=self.config.grid_alpha, axis='x')
        
        ax.invert_yaxis()
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curves(self,
                           training_history: Dict[str, List[float]],
                           title: str = "Learning Curves") -> None:
        """
        Plot training and validation learning curves.
        
        Args:
            training_history: Dictionary containing loss curves
            title: Plot title
        """
        if not self.matplotlib_available:
            print("Matplotlib required for learning curves")
            return
        
        fig, ax = self.plt.subplots(figsize=self.config.figsize)
        
        epochs = None
        for i, (curve_name, values) in enumerate(training_history.items()):
            if epochs is None:
                epochs = np.arange(1, len(values) + 1)
            
            color = self.config.colors[i % len(self.config.colors)]
            ax.plot(epochs, values, label=curve_name, color=color, linewidth=2)
        
        ax.set_title(title, fontsize=self.config.title_fontsize)
        ax.set_xlabel('Epoch', fontsize=self.config.label_fontsize)
        ax.set_ylabel('Loss', fontsize=self.config.label_fontsize)
        
        if self.config.legend:
            ax.legend()
        if self.config.grid:
            ax.grid(True, alpha=self.config.grid_alpha)
        
        plt.tight_layout()
        plt.show()
    
    def create_model_comparison_dashboard(self,
                                        y_true: np.ndarray,
                                        predictions: Dict[str, np.ndarray],
                                        metrics: Dict[str, Dict[str, float]],
                                        timestamps: Optional[np.ndarray] = None,
                                        title: str = "Model Comparison Dashboard") -> None:
        """
        Create comprehensive model comparison dashboard.
        
        Args:
            y_true: True values
            predictions: Dictionary of model predictions
            metrics: Dictionary of model metrics
            timestamps: Optional timestamps
            title: Dashboard title
        """
        if not self.matplotlib_available:
            print("Matplotlib required for dashboard")
            return
        
        fig = self.plt.figure(figsize=(20, 12))
        
        # Time series comparison (top half)
        ax1 = self.plt.subplot(2, 3, (1, 2))
        x_data = timestamps if timestamps is not None else np.arange(len(y_true))
        
        ax1.plot(x_data, y_true, label='Actual', color='black', linewidth=3, alpha=0.8)
        
        for i, (model_name, y_pred) in enumerate(predictions.items()):
            color = self.config.colors[i % len(self.config.colors)]
            ax1.plot(x_data[:len(y_pred)], y_pred, 
                    label=model_name, color=color, linewidth=2, alpha=0.7)
        
        ax1.set_title('Model Predictions Comparison', fontsize=self.config.title_fontsize)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True, alpha=self.config.grid_alpha)
        
        # Performance metrics (top right)
        ax2 = self.plt.subplot(2, 3, 3)
        model_names = list(metrics.keys())
        rmse_values = [metrics[model].get('rmse', 0) for model in model_names]
        
        bars = ax2.bar(model_names, rmse_values, color=self.config.colors[:len(model_names)], alpha=0.7)
        ax2.set_title('RMSE Comparison')
        ax2.set_ylabel('RMSE')
        
        for bar, value in zip(bars, rmse_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}', ha='center', va='bottom')
        
        ax2.grid(True, alpha=self.config.grid_alpha, axis='y')
        
        # Error analysis for best model
        best_model = min(metrics.keys(), key=lambda k: metrics[k].get('rmse', float('inf')))
        residuals = y_true[:len(predictions[best_model])] - predictions[best_model]
        
        # Residuals over time
        ax3 = self.plt.subplot(2, 3, 4)
        ax3.plot(x_data[:len(residuals)], residuals, color=self.config.colors[0], alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_title(f'Residuals - {best_model}')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Residual')
        ax3.grid(True, alpha=self.config.grid_alpha)
        
        # Residual histogram
        ax4 = self.plt.subplot(2, 3, 5)
        ax4.hist(residuals, bins=20, alpha=0.7, color=self.config.colors[1], edgecolor='black')
        ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('Residual Distribution')
        ax4.set_xlabel('Residual')
        ax4.set_ylabel('Frequency')
        ax4.grid(True, alpha=self.config.grid_alpha)
        
        # Model performance table
        ax5 = self.plt.subplot(2, 3, 6)
        ax5.axis('tight')
        ax5.axis('off')
        
        # Create performance table
        table_data = []
        metric_names = ['rmse', 'mae', 'r2']
        
        for model in model_names:
            row = [model] + [f"{metrics[model].get(metric, 0):.3f}" for metric in metric_names]
            table_data.append(row)
        
        table = ax5.table(cellText=table_data,
                         colLabels=['Model'] + [m.upper() for m in metric_names],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax5.set_title('Performance Summary', fontsize=self.config.title_fontsize)
        
        plt.suptitle(title, fontsize=self.config.title_fontsize + 4)
        plt.tight_layout()
        plt.show()
    
    def export_plot(self, filename: str, dpi: int = 300, format: str = 'png'):
        """
        Export the current plot to file.
        
        Args:
            filename: Output filename
            dpi: Resolution in DPI
            format: File format ('png', 'pdf', 'svg', 'jpg')
        """
        if not self.matplotlib_available:
            print("Matplotlib required for plot export")
            return
        
        self.plt.savefig(filename, dpi=dpi, format=format, bbox_inches='tight')
        print(f"Plot saved to {filename}")
    
    def set_style(self, style: str = 'default'):
        """
        Set plotting style.
        
        Args:
            style: Style name ('default', 'seaborn', 'ggplot', etc.)
        """
        if self.matplotlib_available:
            if style == 'seaborn' and hasattr(self, 'sns'):
                self.sns.set_style("whitegrid")
            else:
                self.plt.style.use(style)
            
            self.config.style = style
            print(f"Style set to {style}")