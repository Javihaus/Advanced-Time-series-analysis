"""
Advanced Time Series Analysis Framework

A comprehensive framework for comparing probabilistic programming, deep learning,
and gradient boosting methods for time series forecasting. Features mathematical
foundations and practical implementations of Gaussian Processes, RNNs, LSTMs, 
GRUs, and XGBoost.
"""

__version__ = "1.0.0"
__author__ = "Javier Marin" 
__email__ = "javier@jmarin.info"

from .models.gaussian_process import GaussianProcessModel
from .models.lstm_model import LSTMModel
from .models.rnn_pytorch import RNNModel, GRUModel, LSTMPyTorchModel, AutoregressiveModel
from .models.xgboost_model import XGBoostModel
from .comparison.comparative_analysis import ModelComparator
from .utils.data_preprocessing import TimeSeriesProcessor
from .utils.evaluation_metrics import TimeSeriesMetrics
from .utils.visualization import TimeSeriesVisualizer

__all__ = [
    'GaussianProcessModel',
    'LSTMModel', 
    'RNNModel',
    'GRUModel',
    'LSTMPyTorchModel',
    'AutoregressiveModel',
    'XGBoostModel',
    'ModelComparator',
    'TimeSeriesProcessor',
    'TimeSeriesMetrics',
    'TimeSeriesVisualizer'
]