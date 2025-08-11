"""
RNN, GRU, and Autoregressive implementations for time series forecasting.

This module implements various recurrent neural network architectures using PyTorch
for sequential time series prediction with configurable architectures.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import time
from dataclasses import dataclass


@dataclass
class RNNResults:
    """Results container for RNN predictions."""
    predictions: np.ndarray
    training_history: Dict[str, List[float]]
    model_summary: str
    training_time: float


class BaseRNNModel:
    """Base class for PyTorch RNN models."""
    
    def __init__(self,
                 input_dim: int = 1,
                 hidden_dim: int = 32,
                 output_dim: int = 1,
                 num_layers: int = 1,
                 sequence_length: int = 10,
                 batch_size: int = 32,
                 epochs: int = 500,
                 learning_rate: float = 0.01,
                 train_split: float = 0.7):
        """
        Initialize base RNN model.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Number of hidden units
            output_dim: Number of output features
            num_layers: Number of RNN layers
            sequence_length: Length of input sequences
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            train_split: Fraction of data for training
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.train_split = train_split
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        
        # Initialize PyTorch components
        self._init_pytorch()
    
    def _init_pytorch(self):
        """Initialize PyTorch components."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.preprocessing import MinMaxScaler
            
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self.MinMaxScaler = MinMaxScaler
            self.available = True
            
        except ImportError:
            print("Warning: PyTorch not available. Install with: pip install torch")
            self.available = False
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def _split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into train and test sets."""
        N = len(X)
        split_idx = int(N * self.train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[Any, Any, Any, Any]:
        """Prepare data for training."""
        # Scale data
        self.scaler = self.MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Split data
        X_train, X_test, y_train, y_test = self._split_data(X, y)
        
        # Convert to tensors
        X_train = self.torch.from_numpy(X_train.astype(np.float32)).unsqueeze(-1)
        X_test = self.torch.from_numpy(X_test.astype(np.float32)).unsqueeze(-1)
        y_train = self.torch.from_numpy(y_train.astype(np.float32)).unsqueeze(-1)
        y_test = self.torch.from_numpy(y_test.astype(np.float32)).unsqueeze(-1)
        
        return X_train, X_test, y_train, y_test
    
    def _train_epoch(self, X_train, y_train, X_test, y_test):
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        predictions = self.model(X_train)
        train_loss = self.criterion(predictions, y_train)
        
        # Backward pass
        train_loss.backward()
        self.optimizer.step()
        
        # Evaluation
        self.model.eval()
        with self.torch.no_grad():
            test_predictions = self.model(X_test)
            test_loss = self.criterion(test_predictions, y_test)
        
        return train_loss.item(), test_loss.item()
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Train the model on time series data.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary containing training metrics
        """
        if not self.available:
            raise ImportError("PyTorch required for RNN models")
        
        start_time = time.time()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(data)
        
        # Initialize model, optimizer, and loss
        self.model = self._create_model()
        self.optimizer = self.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = self.nn.MSELoss()
        
        # Training loop
        self.train_losses = []
        self.test_losses = []
        
        for epoch in range(self.epochs):
            train_loss, test_loss = self._train_epoch(X_train, y_train, X_test, y_test)
            
            self.train_losses.append(train_loss)
            self.test_losses.append(test_loss)
            
            if (epoch + 1) % (self.epochs // 10) == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}: Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}')
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'final_train_loss': float(self.train_losses[-1]),
            'final_test_loss': float(self.test_losses[-1]),
            'epochs_trained': self.epochs,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def predict(self, data: np.ndarray, steps_ahead: int = 1) -> RNNResults:
        """
        Make predictions using trained model.
        
        Args:
            data: Input time series data
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            RNNResults containing predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Scale input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:]
        
        with self.torch.no_grad():
            for _ in range(steps_ahead):
                # Prepare input tensor
                input_tensor = self.torch.from_numpy(current_sequence.astype(np.float32))
                input_tensor = input_tensor.unsqueeze(0).unsqueeze(-1)  # Add batch and feature dimensions
                
                # Make prediction
                pred = self.model(input_tensor).item()
                predictions.append(pred)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return RNNResults(
            predictions=predictions,
            training_history={'train_loss': self.train_losses, 'test_loss': self.test_losses},
            model_summary=str(self.model),
            training_time=0.0  # Set during fit
        )
    
    def evaluate(self, data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        results = self.predict(data, steps_ahead=len(target))
        predictions = results.predictions[:len(target)]
        
        return {
            'rmse': np.sqrt(mean_squared_error(target, predictions)),
            'mae': mean_absolute_error(target, predictions),
            'r2': r2_score(target, predictions),
            'mape': np.mean(np.abs((target - predictions) / target)) * 100
        }


class RNNModel(BaseRNNModel):
    """Simple RNN model for time series forecasting."""
    
    def _create_model(self):
        """Create simple RNN architecture."""
        if not self.available:
            raise ImportError("PyTorch required for RNN model")
        
        class SimpleRNN(self.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                self.rnn = self.nn.RNN(
                    input_dim, hidden_dim, num_layers, 
                    nonlinearity='tanh', batch_first=True
                )
                self.fc = self.nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                h0 = self.torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                out, _ = self.rnn(x, h0)
                out = self.fc(out[:, -1, :])  # Take last timestep
                return out
        
        return SimpleRNN(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)


class GRUModel(BaseRNNModel):
    """GRU model for time series forecasting."""
    
    def _create_model(self):
        """Create GRU architecture."""
        if not self.available:
            raise ImportError("PyTorch required for GRU model")
        
        class GRU(self.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                self.gru = self.nn.GRU(
                    input_dim, hidden_dim, num_layers, batch_first=True
                )
                self.fc = self.nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                h0 = self.torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                out, _ = self.gru(x, h0)
                out = self.fc(out[:, -1, :])  # Take last timestep
                return out
        
        return GRU(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)


class LSTMPyTorchModel(BaseRNNModel):
    """LSTM model implemented in PyTorch for comparison."""
    
    def _create_model(self):
        """Create LSTM architecture."""
        if not self.available:
            raise ImportError("PyTorch required for LSTM model")
        
        class LSTM(self.nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers
                
                self.lstm = self.nn.LSTM(
                    input_dim, hidden_dim, num_layers, batch_first=True
                )
                self.fc = self.nn.Linear(hidden_dim, output_dim)
            
            def forward(self, x):
                h0 = self.torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                c0 = self.torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
                out, (_, _) = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])  # Take last timestep
                return out
        
        return LSTM(self.input_dim, self.hidden_dim, self.output_dim, self.num_layers)


class AutoregressiveModel:
    """Simple autoregressive model using linear regression."""
    
    def __init__(self,
                 sequence_length: int = 10,
                 learning_rate: float = 0.01,
                 epochs: int = 500,
                 train_split: float = 0.7):
        """
        Initialize autoregressive model.
        
        Args:
            sequence_length: Number of previous timesteps to use
            learning_rate: Learning rate for optimizer
            epochs: Number of training epochs
            train_split: Fraction of data for training
        """
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.train_split = train_split
        
        # Model components
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.scaler = None
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        
        # Initialize PyTorch components
        self._init_pytorch()
    
    def _init_pytorch(self):
        """Initialize PyTorch components."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from sklearn.preprocessing import MinMaxScaler
            
            self.torch = torch
            self.nn = nn
            self.optim = optim
            self.MinMaxScaler = MinMaxScaler
            self.available = True
            
        except ImportError:
            print("Warning: PyTorch not available. Install with: pip install torch")
            self.available = False
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for autoregressive training."""
        X, y = [], []
        for i in range(len(data) - self.sequence_length - 1):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def _prepare_data(self, data: np.ndarray) -> Tuple[Any, Any, Any, Any]:
        """Prepare data for training."""
        # Scale data
        self.scaler = self.MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X, y = self._create_sequences(scaled_data)
        
        # Split data
        N = len(X)
        split_idx = int(N * self.train_split)
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Convert to tensors
        X_train = self.torch.from_numpy(X_train.astype(np.float32))
        X_test = self.torch.from_numpy(X_test.astype(np.float32))
        y_train = self.torch.from_numpy(y_train.astype(np.float32)).unsqueeze(-1)
        y_test = self.torch.from_numpy(y_test.astype(np.float32)).unsqueeze(-1)
        
        return X_train, X_test, y_train, y_test
    
    def fit(self, data: np.ndarray) -> Dict[str, Any]:
        """
        Train the autoregressive model.
        
        Args:
            data: Time series data
            
        Returns:
            Dictionary containing training metrics
        """
        if not self.available:
            raise ImportError("PyTorch required for autoregressive model")
        
        start_time = time.time()
        
        # Prepare data
        X_train, X_test, y_train, y_test = self._prepare_data(data)
        
        # Create linear model
        self.model = self.nn.Linear(self.sequence_length, 1)
        self.optimizer = self.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = self.nn.MSELoss()
        
        # Training loop
        self.train_losses = []
        self.test_losses = []
        
        for epoch in range(self.epochs):
            # Training
            self.model.train()
            self.optimizer.zero_grad()
            
            train_pred = self.model(X_train)
            train_loss = self.criterion(train_pred, y_train)
            
            train_loss.backward()
            self.optimizer.step()
            
            # Evaluation
            self.model.eval()
            with self.torch.no_grad():
                test_pred = self.model(X_test)
                test_loss = self.criterion(test_pred, y_test)
            
            self.train_losses.append(train_loss.item())
            self.test_losses.append(test_loss.item())
            
            if (epoch + 1) % (self.epochs // 10) == 0:
                print(f'Epoch {epoch + 1}/{self.epochs}: Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}')
        
        training_time = time.time() - start_time
        
        return {
            'training_time': training_time,
            'final_train_loss': float(self.train_losses[-1]),
            'final_test_loss': float(self.test_losses[-1]),
            'epochs_trained': self.epochs,
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
    
    def predict(self, data: np.ndarray, steps_ahead: int = 1) -> RNNResults:
        """
        Make predictions using trained autoregressive model.
        
        Args:
            data: Input time series data
            steps_ahead: Number of steps to predict ahead
            
        Returns:
            RNNResults containing predictions and metadata
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        self.model.eval()
        
        # Scale input data
        scaled_data = self.scaler.transform(data.reshape(-1, 1)).flatten()
        
        predictions = []
        current_sequence = scaled_data[-self.sequence_length:]
        
        with self.torch.no_grad():
            for _ in range(steps_ahead):
                # Prepare input tensor
                input_tensor = self.torch.from_numpy(current_sequence.astype(np.float32)).unsqueeze(0)
                
                # Make prediction
                pred = self.model(input_tensor).item()
                predictions.append(pred)
                
                # Update sequence for next prediction
                current_sequence = np.append(current_sequence[1:], pred)
        
        # Inverse transform predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions).flatten()
        
        return RNNResults(
            predictions=predictions,
            training_history={'train_loss': self.train_losses, 'test_loss': self.test_losses},
            model_summary=str(self.model),
            training_time=0.0  # Set during fit
        )
    
    def evaluate(self, data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        results = self.predict(data, steps_ahead=len(target))
        predictions = results.predictions[:len(target)]
        
        return {
            'rmse': np.sqrt(mean_squared_error(target, predictions)),
            'mae': mean_absolute_error(target, predictions),
            'r2': r2_score(target, predictions),
            'mape': np.mean(np.abs((target - predictions) / target)) * 100
        }
    
    def plot_training_history(self):
        """Plot training history."""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss')
            plt.plot(self.test_losses, label='Test Loss')
            plt.title('Autoregressive Model Training History')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")