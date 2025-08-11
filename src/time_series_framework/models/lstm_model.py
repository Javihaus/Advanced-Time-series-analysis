"""
LSTM implementation for time series forecasting.

This module implements Long Short-Term Memory networks using Keras/TensorFlow
for sequential time series prediction with advanced training techniques.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class LSTMResults:
    """Results container for LSTM predictions."""
    predictions: np.ndarray
    training_history: Dict[str, List[float]]
    model_summary: str
    training_time: float


class LSTMModel:
    """
    LSTM model for time series forecasting.
    
    This class implements LSTM networks with configurable architecture,
    regularization techniques, and comprehensive evaluation metrics.
    """
    
    def __init__(self,
                 hidden_size: int = 64,
                 num_layers: int = 2,
                 dropout_rate: float = 0.2,
                 sequence_length: int = 10,
                 batch_size: int = 32,
                 epochs: int = 100,
                 learning_rate: float = 0.001,
                 early_stopping: bool = True,
                 patience: int = 10,
                 validation_split: float = 0.2):
        """
        Initialize LSTM model.
        
        Args:
            hidden_size: Number of LSTM units in each layer
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            sequence_length: Length of input sequences
            batch_size: Training batch size
            epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            early_stopping: Whether to use early stopping
            patience: Early stopping patience
            validation_split: Fraction of data for validation
        """
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.patience = patience
        self.validation_split = validation_split
        
        # Model components
        self.model = None
        self.scaler = None
        self.history = None
        
        # Initialize TensorFlow/Keras
        self._init_tensorflow()
    
    def _init_tensorflow(self):
        """Initialize TensorFlow/Keras components."""
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.callbacks import EarlyStopping
            from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
            from sklearn.preprocessing import MinMaxScaler
            
            self.tf = tf
            self.Sequential = Sequential
            self.LSTM = LSTM
            self.Dense = Dense
            self.Dropout = Dropout
            self.Adam = Adam
            self.EarlyStopping = EarlyStopping
            self.TimeseriesGenerator = TimeseriesGenerator
            self.MinMaxScaler = MinMaxScaler
            self.available = True
            
        except ImportError:
            print("Warning: TensorFlow/Keras not available. Install with: pip install tensorflow")
            self.available = False
    
    def _create_model(self, input_shape: Tuple[int, int]) -> Any:
        """Create LSTM model architecture."""
        if not self.available:
            raise ImportError("TensorFlow/Keras required for LSTM model")
        
        model = self.Sequential()
        
        # First LSTM layer
        model.add(self.LSTM(
            self.hidden_size,
            return_sequences=True if self.num_layers > 1 else False,
            input_shape=input_shape
        ))
        model.add(self.Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, self.num_layers):
            return_sequences = i < self.num_layers - 1
            model.add(self.LSTM(
                self.hidden_size,
                return_sequences=return_sequences
            ))
            model.add(self.Dropout(self.dropout_rate))
        
        # Output layer
        model.add(self.Dense(1))
        
        # Compile model
        optimizer = self.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare time series data for LSTM training."""
        # Scale data
        self.scaler = self.MinMaxScaler()
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length])
        
        X = np.array(X).reshape(-1, self.sequence_length, 1)
        y = np.array(y)
        
        return X, y
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Train LSTM model on time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary containing training metrics and history
        """
        if not self.available:
            raise ImportError("TensorFlow/Keras required for LSTM model")
        
        import time
        start_time = time.time()
        
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Split data
        split_idx = int(len(X) * (1 - self.validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create model
        self.model = self._create_model((self.sequence_length, 1))
        
        # Setup callbacks
        callbacks = []
        if self.early_stopping:
            early_stop = self.EarlyStopping(
                monitor='val_loss',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]),
            'epochs_trained': len(self.history.history['loss']),
            'model_parameters': self.model.count_params()
        }
    
    def predict(self, data: np.ndarray, steps_ahead: int = 1) -> LSTMResults:
        """
        Make predictions using trained LSTM model.
        
        Args:
            data: Input time series data
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            LSTMResults containing predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Scale input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:]
        
        for _ in range(steps_ahead):
            # Prepare input
            X_input = current_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            pred = self.model.predict(X_input, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence for next prediction
            current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return LSTMResults(
            predictions=predictions,
            training_history=self.history.history if self.history else {},
            model_summary=str(self.model.summary()) if self.model else "",
            training_time=0.0  # Set during fit
        )
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation")
        
        # Prepare test data
        X_test_scaled = []
        for i in range(len(X_test) - self.sequence_length + 1):
            sequence = self.scaler.transform(
                X_test[i:i+self.sequence_length].reshape(-1, 1)
            ).flatten()
            X_test_scaled.append(sequence)
        
        X_test_scaled = np.array(X_test_scaled).reshape(-1, self.sequence_length, 1)
        y_test_subset = y_test[self.sequence_length-1:]
        
        # Make predictions
        y_pred_scaled = self.model.predict(X_test_scaled, verbose=0)
        y_pred = self.scaler.inverse_transform(y_pred_scaled).flatten()
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test_subset, y_pred)),
            'mae': mean_absolute_error(y_test_subset, y_pred),
            'r2': r2_score(y_test_subset, y_pred),
            'mape': np.mean(np.abs((y_test_subset - y_pred) / y_test_subset)) * 100
        }
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            raise ValueError("Model must be trained first")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot loss
            ax1.plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Model Loss')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot MAE
            if 'mae' in self.history.history:
                ax2.plot(self.history.history['mae'], label='Training MAE')
                if 'val_mae' in self.history.history:
                    ax2.plot(self.history.history['val_mae'], label='Validation MAE')
                ax2.set_title('Model MAE')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('MAE')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")
    
    def get_model_architecture(self) -> Dict[str, Any]:
        """Get detailed model architecture information."""
        if self.model is None:
            raise ValueError("Model must be created first")
        
        architecture = {
            'total_params': self.model.count_params(),
            'trainable_params': int(np.sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])),
            'layers': []
        }
        
        for i, layer in enumerate(self.model.layers):
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': layer.output_shape,
                'params': layer.count_params()
            }
            
            # Add layer-specific information
            if hasattr(layer, 'units'):
                layer_info['units'] = layer.units
            if hasattr(layer, 'rate'):
                layer_info['dropout_rate'] = layer.rate
            if hasattr(layer, 'return_sequences'):
                layer_info['return_sequences'] = layer.return_sequences
            
            architecture['layers'].append(layer_info)
        
        return architecture
    
    def save_model(self, filepath: str):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file."""
        if not self.available:
            raise ImportError("TensorFlow/Keras required to load model")
        
        from tensorflow.keras.models import load_model
        self.model = load_model(filepath)
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
        
        # Use the last sequence_length points as initial input
        current_data = data[-self.sequence_length:]
        forecasts = []
        
        for _ in range(n_steps):
            # Scale current sequence
            scaled_sequence = self.scaler.transform(current_data.reshape(-1, 1)).flatten()
            
            # Reshape for model input
            X_input = scaled_sequence.reshape(1, self.sequence_length, 1)
            
            # Make prediction
            pred_scaled = self.model.predict(X_input, verbose=0)[0, 0]
            
            # Inverse transform prediction
            pred = self.scaler.inverse_transform([[pred_scaled]])[0, 0]
            forecasts.append(pred)
            
            # Update current sequence (remove first element, add prediction)
            current_data = np.append(current_data[1:], pred)
        
        return np.array(forecasts)