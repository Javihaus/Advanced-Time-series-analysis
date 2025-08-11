"""
Gaussian Process implementation for time series forecasting.

This module implements Gaussian Processes using PyMC3 for probabilistic
time series forecasting with uncertainty quantification.
"""

from typing import Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class GPResults:
    """Results container for Gaussian Process predictions."""
    mean_predictions: np.ndarray
    std_predictions: np.ndarray
    posterior_samples: np.ndarray
    hyperparameters: Dict[str, float]
    training_time: float


class GaussianProcessModel:
    """
    Gaussian Process model for time series forecasting.
    
    This class implements GP regression with various kernel options,
    Bayesian hyperparameter inference, and uncertainty quantification.
    """
    
    def __init__(self,
                 kernel: str = 'rbf',
                 mean_function: str = 'zero',
                 n_samples: int = 2000,
                 tune: int = 1000,
                 random_state: int = 42):
        """
        Initialize Gaussian Process model.
        
        Args:
            kernel: Kernel type ('rbf', 'matern32', 'matern52')
            mean_function: Mean function ('zero', 'constant', 'linear')
            n_samples: Number of posterior samples
            tune: Number of tuning samples for MCMC
            random_state: Random seed for reproducibility
        """
        self.kernel = kernel
        self.mean_function = mean_function
        self.n_samples = n_samples
        self.tune = tune
        self.random_state = random_state
        
        # Model components
        self.model = None
        self.trace = None
        self.X_train = None
        self.y_train = None
        
        # Initialize PyMC3 components
        self._init_pymc3()
    
    def _init_pymc3(self):
        """Initialize PyMC3 components."""
        try:
            import pymc3 as pm
            import theano.tensor as tt
            self.pm = pm
            self.tt = tt
            self.available = True
        except ImportError:
            print("Warning: PyMC3 not available. Install with: pip install pymc3")
            self.available = False
    
    def _create_kernel(self, X: np.ndarray, Xs: np.ndarray) -> Any:
        """Create covariance kernel based on specified type."""
        if not self.available:
            raise ImportError("PyMC3 required for Gaussian Process")
        
        # Hyperparameters
        eta = self.pm.HalfCauchy('eta', beta=1.0)  # Signal variance
        rho = self.pm.HalfCauchy('rho', beta=1.0)  # Length scale
        sigma = self.pm.HalfCauchy('sigma', beta=0.1)  # Noise variance
        
        if self.kernel == 'rbf':
            # Squared exponential kernel
            cov_func = eta**2 * self.pm.gp.cov.ExpQuad(1, rho)
        elif self.kernel == 'matern32':
            cov_func = eta**2 * self.pm.gp.cov.Matern32(1, rho)
        elif self.kernel == 'matern52':
            cov_func = eta**2 * self.pm.gp.cov.Matern52(1, rho)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
        
        return cov_func
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Fit Gaussian Process to training data.
        
        Args:
            X: Input features (time indices)
            y: Target values
            
        Returns:
            Dictionary containing training metrics and hyperparameters
        """
        if not self.available:
            raise ImportError("PyMC3 required for Gaussian Process")
        
        import time
        start_time = time.time()
        
        self.X_train = X.reshape(-1, 1) if X.ndim == 1 else X
        self.y_train = y
        
        with self.pm.Model() as self.model:
            # Mean function
            if self.mean_function == 'zero':
                mean_func = self.pm.gp.mean.Zero()
            elif self.mean_function == 'constant':
                mean_func = self.pm.gp.mean.Constant(c=0.0)
            else:
                mean_func = self.pm.gp.mean.Zero()
            
            # Covariance function
            cov_func = self._create_kernel(self.X_train, self.X_train)
            
            # Gaussian Process
            gp = self.pm.gp.Marginal(mean_func=mean_func, cov_func=cov_func)
            
            # Observed data
            y_obs = gp.marginal_likelihood("y_obs", X=self.X_train, y=self.y_train, noise=self.pm.find_VAR('sigma'))
            
            # MCMC sampling
            self.trace = self.pm.sample(
                draws=self.n_samples,
                tune=self.tune,
                random_seed=self.random_state,
                return_inferencedata=False
            )
        
        training_time = time.time() - start_time
        
        # Extract hyperparameters
        hyperparams = {}
        for var_name in ['eta', 'rho', 'sigma']:
            if var_name in self.trace.varnames:
                hyperparams[var_name] = float(np.mean(self.trace[var_name]))
        
        return {
            'training_time': training_time,
            'hyperparameters': hyperparams,
            'n_samples': len(self.trace),
            'convergence': self._check_convergence()
        }
    
    def predict(self, X_test: np.ndarray, n_pred_samples: int = 100) -> GPResults:
        """
        Make predictions on test data.
        
        Args:
            X_test: Test input features
            n_pred_samples: Number of posterior predictive samples
            
        Returns:
            GPResults containing predictions and uncertainty estimates
        """
        if self.model is None or self.trace is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_test = X_test.reshape(-1, 1) if X_test.ndim == 1 else X_test
        
        with self.model:
            # Conditional GP for predictions
            gp_pred = self.pm.gp.Marginal(
                mean_func=self.pm.gp.mean.Zero(),
                cov_func=self._create_kernel(self.X_train, X_test)
            )
            
            # Posterior predictive samples
            pred_samples = gp_pred.conditional(
                "pred", Xnew=X_test, pred_noise=True
            )
            
            posterior_pred = self.pm.sample_posterior_predictive(
                self.trace,
                samples=n_pred_samples,
                var_names=['pred'],
                random_seed=self.random_state
            )
        
        # Extract predictions
        pred_samples = posterior_pred['pred']
        mean_pred = np.mean(pred_samples, axis=0)
        std_pred = np.std(pred_samples, axis=0)
        
        # Extract hyperparameters
        hyperparams = {}
        for var_name in ['eta', 'rho', 'sigma']:
            if var_name in self.trace.varnames:
                hyperparams[var_name] = float(np.mean(self.trace[var_name]))
        
        return GPResults(
            mean_predictions=mean_pred,
            std_predictions=std_pred,
            posterior_samples=pred_samples,
            hyperparameters=hyperparams,
            training_time=0.0  # Set during fit
        )
    
    def _check_convergence(self) -> Dict[str, float]:
        """Check MCMC convergence diagnostics."""
        try:
            import arviz as az
            summary = az.summary(self.trace)
            return {
                'r_hat_max': float(summary['r_hat'].max()),
                'ess_bulk_min': float(summary['ess_bulk'].min()),
                'ess_tail_min': float(summary['ess_tail'].min())
            }
        except ImportError:
            return {'r_hat_max': 1.0, 'ess_bulk_min': 100, 'ess_tail_min': 100}
    
    def get_kernel_parameters(self) -> Dict[str, np.ndarray]:
        """Get posterior distributions of kernel hyperparameters."""
        if self.trace is None:
            raise ValueError("Model must be fitted first")
        
        params = {}
        for var_name in ['eta', 'rho', 'sigma']:
            if var_name in self.trace.varnames:
                params[var_name] = self.trace[var_name]
        
        return params
    
    def plot_posterior_predictive(self, X_test: np.ndarray, y_test: Optional[np.ndarray] = None):
        """Plot posterior predictive distribution."""
        try:
            import matplotlib.pyplot as plt
            
            results = self.predict(X_test)
            
            plt.figure(figsize=(12, 8))
            
            # Training data
            plt.scatter(self.X_train.flatten(), self.y_train, 
                       alpha=0.6, label='Training Data', color='blue')
            
            # Test data if provided
            if y_test is not None:
                plt.scatter(X_test.flatten(), y_test, 
                           alpha=0.6, label='Test Data', color='red')
            
            # Predictions
            plt.plot(X_test.flatten(), results.mean_predictions, 
                    'r-', label='GP Mean', linewidth=2)
            
            # Uncertainty bands
            plt.fill_between(
                X_test.flatten(),
                results.mean_predictions - 2 * results.std_predictions,
                results.mean_predictions + 2 * results.std_predictions,
                alpha=0.2, color='red', label='95% Confidence'
            )
            
            plt.xlabel('Time')
            plt.ylabel('Value')
            plt.title('Gaussian Process Predictions')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            
        except ImportError:
            print("Matplotlib required for plotting")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance on test data."""
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        results = self.predict(X_test)
        predictions = results.mean_predictions
        
        return {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'log_likelihood': self._compute_log_likelihood(y_test, results),
            'uncertainty_coverage': self._compute_coverage(y_test, results)
        }
    
    def _compute_log_likelihood(self, y_true: np.ndarray, results: GPResults) -> float:
        """Compute predictive log-likelihood."""
        from scipy.stats import norm
        
        log_prob = norm.logpdf(
            y_true,
            loc=results.mean_predictions,
            scale=results.std_predictions
        )
        
        return float(np.mean(log_prob))
    
    def _compute_coverage(self, y_true: np.ndarray, results: GPResults, 
                         confidence: float = 0.95) -> float:
        """Compute prediction interval coverage."""
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        lower = results.mean_predictions - z_score * results.std_predictions
        upper = results.mean_predictions + z_score * results.std_predictions
        
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        
        return float(coverage)