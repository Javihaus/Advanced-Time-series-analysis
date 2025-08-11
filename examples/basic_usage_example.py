"""
Basic usage example for the Advanced Time Series Analysis Framework.

This script demonstrates how to use the framework to compare different
time series forecasting models on synthetic data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path for example
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import framework components
from time_series_framework import (
    GaussianProcessModel,
    LSTMModel,
    RNNModel,
    GRUModel,
    XGBoostModel,
    TimeSeriesProcessor,
    TimeSeriesMetrics,
    ModelComparator,
    TimeSeriesVisualizer
)


def generate_sample_data(n_points=200, noise_level=0.1):
    """Generate synthetic time series data for demonstration."""
    np.random.seed(42)
    
    t = np.arange(n_points)
    
    # Components of the time series
    trend = 0.02 * t
    seasonal_yearly = 0.8 * np.sin(2 * np.pi * t / 50)  # Annual cycle
    seasonal_weekly = 0.3 * np.sin(2 * np.pi * t / 7)   # Weekly cycle  
    noise = noise_level * np.random.normal(size=n_points)
    
    # Combine components
    data = trend + seasonal_yearly + seasonal_weekly + noise + 10
    
    # Create timestamps
    timestamps = pd.date_range('2020-01-01', periods=n_points, freq='D')
    
    return data, timestamps


def preprocess_data(data, train_ratio=0.8):
    """Preprocess the data using the framework's processor."""
    processor = TimeSeriesProcessor(
        scaling_method='minmax',
        handle_missing='interpolate',
        detect_outliers=True
    )
    
    # Process the data
    result = processor.process_full_pipeline(data)
    
    # Split into train/test
    split_point = int(len(result.processed_data) * train_ratio)
    train_data = result.processed_data[:split_point]
    test_data = result.processed_data[split_point:]
    
    print(f"Data preprocessing completed:")
    print(f"  - Original data shape: {result.metadata['original_shape']}")
    print(f"  - Processed data shape: {result.metadata['processed_shape']}")
    print(f"  - Scaling method: {result.metadata['scaling_method']}")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Test samples: {len(test_data)}")
    
    return result.processed_data, result.scaler, split_point


def train_models(train_data, test_data, original_test_data):
    """Train different models and collect results."""
    print("\n" + "="*60)
    print("TRAINING MODELS")
    print("="*60)
    
    models = {}
    model_results = {}
    
    # Model Comparator
    comparator = ModelComparator()
    
    try:
        # 1. XGBoost Model (most likely to work without heavy dependencies)
        print("\nTraining XGBoost Model...")
        xgb_model = XGBoostModel(
            n_estimators=50,
            learning_rate=0.1,
            max_depth=4,
            sequence_length=10
        )
        
        # Combine train and test for XGBoost (it handles splitting internally)
        full_data = np.concatenate([train_data, test_data])
        xgb_metrics = xgb_model.fit(full_data)
        
        # Predict on test data
        xgb_results = xgb_model.predict(train_data, steps_ahead=len(test_data))
        xgb_predictions = xgb_results.predictions
        
        comparator.add_model_result(
            "XGBoost", 
            original_test_data, 
            xgb_predictions[:len(original_test_data)],
            xgb_metrics['training_time']
        )
        
        models['XGBoost'] = xgb_model
        print(f"  - Training completed in {xgb_metrics['training_time']:.2f}s")
        
    except Exception as e:
        print(f"  - XGBoost failed: {e}")
    
    try:
        # 2. RNN Model (PyTorch - if available)
        print("\nTraining RNN Model...")
        rnn_model = RNNModel(
            hidden_dim=32,
            num_layers=1,
            epochs=50,
            learning_rate=0.01
        )
        
        # Combine for RNN (it handles splitting)
        full_data = np.concatenate([train_data, test_data])
        rnn_metrics = rnn_model.fit(full_data)
        
        # Predict
        rnn_results = rnn_model.predict(train_data, steps_ahead=len(test_data))
        rnn_predictions = rnn_results.predictions
        
        comparator.add_model_result(
            "RNN",
            original_test_data,
            rnn_predictions[:len(original_test_data)],
            rnn_metrics['training_time']
        )
        
        models['RNN'] = rnn_model
        print(f"  - Training completed in {rnn_metrics['training_time']:.2f}s")
        
    except Exception as e:
        print(f"  - RNN failed: {e}")
    
    try:
        # 3. LSTM Model (Keras/TensorFlow - if available)
        print("\nTraining LSTM Model...")
        lstm_model = LSTMModel(
            hidden_size=32,
            num_layers=1,
            epochs=50,
            learning_rate=0.01
        )
        
        # Combine for LSTM
        full_data = np.concatenate([train_data, test_data])
        lstm_metrics = lstm_model.fit(full_data)
        
        # Predict
        lstm_results = lstm_model.predict(train_data, steps_ahead=len(test_data))
        lstm_predictions = lstm_results.predictions
        
        comparator.add_model_result(
            "LSTM",
            original_test_data,
            lstm_predictions[:len(original_test_data)],
            lstm_metrics['training_time']
        )
        
        models['LSTM'] = lstm_model
        print(f"  - Training completed in {lstm_metrics['training_time']:.2f}s")
        
    except Exception as e:
        print(f"  - LSTM failed: {e}")
    
    # Simple baseline model
    print("\nCreating Baseline Model (Simple Mean)...")
    baseline_pred = np.full(len(test_data), np.mean(train_data))
    
    comparator.add_model_result(
        "Baseline",
        original_test_data,
        baseline_pred[:len(original_test_data)],
        0.001  # Minimal training time
    )
    
    print(f"  - Baseline model created")
    
    return comparator, models


def analyze_results(comparator):
    """Analyze and display model comparison results."""
    print("\n" + "="*60)
    print("MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Generate comparison report
    comparison_results = comparator.generate_comparison_report()
    
    # Display rankings
    print("\nModel Rankings:")
    print("-" * 30)
    for metric, ranking in comparison_results.rankings.items():
        print(f"{metric.upper()}:")
        for i, model in enumerate(ranking, 1):
            model_result = comparison_results.model_results[model]
            metric_value = model_result.metrics.get(metric, 'N/A')
            if isinstance(metric_value, float):
                print(f"  {i}. {model}: {metric_value:.4f}")
            else:
                print(f"  {i}. {model}: {metric_value}")
        print()
    
    # Display recommendations
    print("Recommendations:")
    print("-" * 20)
    for rec in comparison_results.recommendations:
        print(f"• {rec}")
    
    # Display performance summary
    print(f"\nPerformance Summary:")
    print("-" * 20)
    for model_name, result in comparison_results.model_results.items():
        print(f"{model_name}:")
        print(f"  RMSE: {result.metrics.get('rmse', 'N/A'):.4f}")
        print(f"  MAE: {result.metrics.get('mae', 'N/A'):.4f}")
        print(f"  R²: {result.metrics.get('r2', 'N/A'):.4f}")
        print(f"  Training Time: {result.training_time:.3f}s")
        print()
    
    return comparison_results


def visualize_results(original_data, timestamps, comparator, split_point):
    """Create visualizations of the results."""
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    try:
        visualizer = TimeSeriesVisualizer()
        
        # Extract predictions for visualization
        predictions = {}
        for model_name, result in comparator.model_results.items():
            predictions[model_name] = result.predictions
        
        # Plot comparison
        test_data = original_data[split_point:]
        
        print("Generating comparison plots...")
        
        # Create comparison dashboard
        comparator.plot_comparison_dashboard(
            y_true=test_data,
            timestamps=timestamps[split_point:split_point+len(test_data)] if timestamps is not None else None,
            title="Time Series Model Comparison"
        )
        
        print("Visualizations completed!")
        
    except Exception as e:
        print(f"Visualization failed: {e}")
        print("Note: This might be due to missing matplotlib or display issues in some environments")


def main():
    """Main execution function."""
    print("Advanced Time Series Analysis Framework - Basic Example")
    print("="*60)
    
    # Generate sample data
    print("Generating synthetic time series data...")
    data, timestamps = generate_sample_data(n_points=200, noise_level=0.15)
    
    # Preprocess data
    print("\nPreprocessing data...")
    processed_data, scaler, split_point = preprocess_data(data)
    
    # Prepare original test data for comparison
    original_test_data = data[split_point:]
    test_data = processed_data[split_point:]
    train_data = processed_data[:split_point]
    
    # Train models
    comparator, models = train_models(train_data, test_data, original_test_data)
    
    # Check if any models were trained successfully
    if not comparator.model_results:
        print("\nNo models were trained successfully. This might be due to missing dependencies.")
        print("Please install the required packages: pip install -r requirements.txt")
        return
    
    # Analyze results
    comparison_results = analyze_results(comparator)
    
    # Create visualizations (optional)
    try:
        visualize_results(data, timestamps, comparator, split_point)
    except Exception as e:
        print(f"Skipping visualizations due to: {e}")
    
    # Save results
    try:
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        # Export to CSV
        comparator.export_results_to_csv(output_dir / "model_comparison_results.csv")
        
        print(f"\nResults saved to {output_dir}/")
        
    except Exception as e:
        print(f"Could not save results: {e}")
    
    print("\n" + "="*60)
    print("EXAMPLE COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nThis example demonstrated:")
    print("• Synthetic time series data generation")
    print("• Data preprocessing with the framework")
    print("• Training multiple forecasting models")
    print("• Comprehensive model comparison and evaluation")
    print("• Statistical analysis and recommendations")
    
    if len(comparator.model_results) > 0:
        best_model = min(comparator.model_results.keys(), 
                        key=lambda k: comparator.model_results[k].metrics.get('rmse', float('inf')))
        print(f"\nBest performing model: {best_model}")
    
    print("\nFor more advanced usage, see the notebooks/ directory and documentation.")


if __name__ == "__main__":
    main()