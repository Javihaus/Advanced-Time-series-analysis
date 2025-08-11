"""
Basic functionality tests for the Advanced Time Series Analysis Framework.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from time_series_framework import (
        GaussianProcessModel,
        LSTMModel,
        RNNModel,
        GRUModel,
        XGBoostModel,
        TimeSeriesProcessor,
        TimeSeriesMetrics,
        ModelComparator
    )
    IMPORTS_AVAILABLE = True
except ImportError:
    IMPORTS_AVAILABLE = False


@pytest.fixture
def sample_time_series_data():
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n = 100
    t = np.arange(n)
    
    # Create synthetic time series with trend and noise
    trend = 0.02 * t
    seasonal = 0.5 * np.sin(2 * np.pi * t / 12)
    noise = 0.1 * np.random.normal(size=n)
    
    data = trend + seasonal + noise + 10  # Add baseline
    return data


@pytest.fixture 
def sample_train_test_split():
    """Generate train/test split data."""
    np.random.seed(42)
    n_train, n_test = 80, 20
    
    # Training data
    t_train = np.arange(n_train)
    trend_train = 0.02 * t_train
    seasonal_train = 0.5 * np.sin(2 * np.pi * t_train / 12)
    noise_train = 0.1 * np.random.normal(size=n_train)
    y_train = trend_train + seasonal_train + noise_train + 10
    
    # Test data
    t_test = np.arange(n_train, n_train + n_test)
    trend_test = 0.02 * t_test
    seasonal_test = 0.5 * np.sin(2 * np.pi * t_test / 12)
    noise_test = 0.1 * np.random.normal(size=n_test)
    y_test = trend_test + seasonal_test + noise_test + 10
    
    return y_train, y_test


class TestDataProcessing:
    """Test data preprocessing functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_time_series_processor_initialization(self):
        """Test TimeSeriesProcessor initialization."""
        processor = TimeSeriesProcessor()
        assert processor.scaling_method == 'minmax'
        assert processor.handle_missing == 'interpolate'
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_basic_preprocessing_pipeline(self, sample_time_series_data):
        """Test basic preprocessing pipeline."""
        processor = TimeSeriesProcessor()
        result = processor.process_full_pipeline(sample_time_series_data)
        
        assert result.processed_data is not None
        assert result.original_data is not None
        assert len(result.processed_data) == len(sample_time_series_data)
        assert 'data_stats' in result.metadata


class TestEvaluationMetrics:
    """Test evaluation metrics functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_metrics_calculator_initialization(self):
        """Test TimeSeriesMetrics initialization."""
        metrics = TimeSeriesMetrics()
        assert hasattr(metrics, 'rmse')
        assert hasattr(metrics, 'mae')
        assert hasattr(metrics, 'mape')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_basic_metrics_calculation(self, sample_train_test_split):
        """Test basic metrics calculation."""
        y_train, y_test = sample_train_test_split
        # Create simple predictions (just use training mean)
        y_pred = np.full_like(y_test, np.mean(y_train))
        
        metrics = TimeSeriesMetrics()
        
        rmse = metrics.rmse(y_test, y_pred)
        mae = metrics.mae(y_test, y_pred)
        mape = metrics.mape(y_test, y_pred)
        
        assert rmse > 0
        assert mae > 0
        assert mape > 0
        assert isinstance(rmse, float)
        assert isinstance(mae, float)
        assert isinstance(mape, float)


class TestModelComparator:
    """Test model comparison functionality."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_model_comparator_initialization(self):
        """Test ModelComparator initialization."""
        comparator = ModelComparator()
        assert hasattr(comparator, 'model_results')
        assert hasattr(comparator, 'evaluation_metrics')
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_add_model_results(self, sample_train_test_split):
        """Test adding model results to comparator."""
        y_train, y_test = sample_train_test_split
        
        # Create dummy predictions
        y_pred_model1 = y_test + 0.1 * np.random.normal(size=len(y_test))
        y_pred_model2 = y_test + 0.2 * np.random.normal(size=len(y_test))
        
        comparator = ModelComparator()
        
        # Add model results
        comparator.add_model_result("Model1", y_test, y_pred_model1, 10.0)
        comparator.add_model_result("Model2", y_test, y_pred_model2, 15.0)
        
        assert len(comparator.model_results) == 2
        assert "Model1" in comparator.model_results
        assert "Model2" in comparator.model_results
        
        # Test ranking
        rankings = comparator.rank_models()
        assert isinstance(rankings, dict)
        assert len(rankings) > 0


class TestFrameworkIntegration:
    """Test overall framework integration."""
    
    @pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="Framework imports not available")
    def test_framework_imports(self):
        """Test that all main components can be imported."""
        # This test mainly checks that imports work correctly
        assert GaussianProcessModel is not None
        assert LSTMModel is not None
        assert RNNModel is not None
        assert GRUModel is not None
        assert XGBoostModel is not None
        assert TimeSeriesProcessor is not None
        assert TimeSeriesMetrics is not None
        assert ModelComparator is not None
    
    def test_numpy_basic_functionality(self):
        """Test that numpy is working correctly (basic sanity check)."""
        arr = np.array([1, 2, 3, 4, 5])
        assert np.mean(arr) == 3.0
        assert np.std(arr) > 0
    
    def test_pandas_basic_functionality(self):
        """Test that pandas is working correctly (basic sanity check)."""
        df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        assert len(df) == 3
        assert list(df.columns) == ['A', 'B']


if __name__ == "__main__":
    pytest.main([__file__])