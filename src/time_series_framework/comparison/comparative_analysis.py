"""
Comprehensive comparative analysis framework for time series models.

This module provides tools for comparing different time series forecasting models,
including statistical tests, performance benchmarking, and comprehensive reporting.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
from dataclasses import dataclass
import time
from pathlib import Path


@dataclass
class ModelResult:
    """Container for individual model results."""
    model_name: str
    predictions: np.ndarray
    metrics: Dict[str, float]
    training_time: float
    model_info: Dict[str, Any]
    residuals: Optional[np.ndarray] = None


@dataclass
class ComparisonResults:
    """Container for comprehensive comparison results."""
    model_results: Dict[str, ModelResult]
    rankings: Dict[str, List[str]]
    statistical_tests: Dict[str, Any]
    summary_statistics: Dict[str, Any]
    recommendations: List[str]


class ModelComparator:
    """
    Comprehensive model comparison framework.
    
    This class provides tools for comparing multiple time series forecasting models
    across various dimensions including accuracy, efficiency, and statistical properties.
    """
    
    def __init__(self, 
                 evaluation_metrics: List[str] = None,
                 significance_level: float = 0.05):
        """
        Initialize model comparator.
        
        Args:
            evaluation_metrics: List of metrics to evaluate
            significance_level: Significance level for statistical tests
        """
        self.evaluation_metrics = evaluation_metrics or [
            'rmse', 'mae', 'mape', 'r2', 'directional_accuracy'
        ]
        self.significance_level = significance_level
        self.model_results = {}
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize required components."""
        try:
            from scipy import stats
            from ..utils.evaluation_metrics import TimeSeriesMetrics
            from ..utils.visualization import TimeSeriesVisualizer
            
            self.stats = stats
            self.metrics_calculator = TimeSeriesMetrics()
            self.visualizer = TimeSeriesVisualizer()
            self.scipy_available = True
            
        except ImportError:
            print("Warning: Some components not available for advanced analysis")
            self.scipy_available = False
    
    def add_model_result(self,
                        model_name: str,
                        y_true: np.ndarray,
                        y_pred: np.ndarray,
                        training_time: float,
                        model_info: Dict[str, Any] = None,
                        y_train: Optional[np.ndarray] = None) -> None:
        """
        Add a model's results to the comparison.
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            training_time: Training time in seconds
            model_info: Additional model information
            y_train: Training data for MASE calculation
        """
        # Calculate metrics
        if hasattr(self, 'metrics_calculator'):
            evaluation_result = self.metrics_calculator.calculate_all_metrics(
                y_true, y_pred, y_train
            )
            metrics = evaluation_result.point_metrics
        else:
            # Basic metrics calculation
            metrics = self._calculate_basic_metrics(y_true, y_pred)
        
        # Calculate residuals
        residuals = y_true - y_pred
        
        # Store result
        self.model_results[model_name] = ModelResult(
            model_name=model_name,
            predictions=y_pred,
            metrics=metrics,
            training_time=training_time,
            model_info=model_info or {},
            residuals=residuals
        )
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic metrics when advanced calculator not available."""
        return {
            'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
            'mae': np.mean(np.abs(y_true - y_pred)),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'r2': 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
        }
    
    def rank_models(self, 
                   ranking_metrics: List[str] = None,
                   weights: Dict[str, float] = None) -> Dict[str, List[str]]:
        """
        Rank models based on performance metrics.
        
        Args:
            ranking_metrics: Metrics to use for ranking
            weights: Weights for different metrics in composite ranking
            
        Returns:
            Dictionary of rankings for each metric
        """
        if not self.model_results:
            raise ValueError("No model results available for ranking")
        
        ranking_metrics = ranking_metrics or self.evaluation_metrics
        rankings = {}
        
        for metric in ranking_metrics:
            # Get metric values for all models
            metric_values = {}
            for model_name, result in self.model_results.items():
                if metric in result.metrics:
                    metric_values[model_name] = result.metrics[metric]
            
            if not metric_values:
                continue
            
            # Sort based on metric (lower is better for error metrics, higher for R²)
            reverse = metric in ['r2', 'directional_accuracy']
            sorted_models = sorted(metric_values.items(), 
                                 key=lambda x: x[1], 
                                 reverse=reverse)
            
            rankings[metric] = [model for model, _ in sorted_models]
        
        # Create composite ranking if weights provided
        if weights:
            composite_scores = self._calculate_composite_scores(weights)
            sorted_composite = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
            rankings['composite'] = [model for model, _ in sorted_composite]
        
        return rankings
    
    def _calculate_composite_scores(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate composite scores based on weighted metrics."""
        composite_scores = {}
        
        # Normalize metrics to 0-1 scale
        normalized_metrics = {}
        for metric in weights.keys():
            values = [result.metrics.get(metric, 0) for result in self.model_results.values()]
            if values:
                min_val, max_val = min(values), max(values)
                range_val = max_val - min_val
                
                for model_name, result in self.model_results.items():
                    if model_name not in normalized_metrics:
                        normalized_metrics[model_name] = {}
                    
                    if range_val > 0:
                        # For error metrics, lower is better (invert)
                        if metric in ['rmse', 'mae', 'mape']:
                            normalized_metrics[model_name][metric] = 1 - (result.metrics.get(metric, 0) - min_val) / range_val
                        else:
                            normalized_metrics[model_name][metric] = (result.metrics.get(metric, 0) - min_val) / range_val
                    else:
                        normalized_metrics[model_name][metric] = 0.5
        
        # Calculate weighted composite scores
        for model_name in self.model_results.keys():
            score = 0
            total_weight = 0
            for metric, weight in weights.items():
                if metric in normalized_metrics.get(model_name, {}):
                    score += normalized_metrics[model_name][metric] * weight
                    total_weight += weight
            
            composite_scores[model_name] = score / total_weight if total_weight > 0 else 0
        
        return composite_scores
    
    def perform_statistical_tests(self) -> Dict[str, Any]:
        """
        Perform statistical tests for model comparison.
        
        Returns:
            Dictionary containing various statistical test results
        """
        if not self.scipy_available:
            print("Warning: SciPy required for statistical tests")
            return {}
        
        test_results = {}
        
        # Pairwise comparison tests
        model_names = list(self.model_results.keys())
        if len(model_names) >= 2:
            test_results['pairwise_tests'] = self._perform_pairwise_tests(model_names)
        
        # Friedman test for multiple model comparison
        if len(model_names) > 2:
            test_results['friedman_test'] = self._perform_friedman_test(model_names)
        
        # Residual analysis tests
        test_results['residual_tests'] = self._perform_residual_tests()
        
        return test_results
    
    def _perform_pairwise_tests(self, model_names: List[str]) -> Dict[str, Any]:
        """Perform pairwise statistical tests between models."""
        pairwise_results = {}
        
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                
                # Diebold-Mariano test
                dm_result = self._diebold_mariano_test(
                    self.model_results[model1].residuals,
                    self.model_results[model2].residuals
                )
                
                # Paired t-test on absolute errors
                errors1 = np.abs(self.model_results[model1].residuals)
                errors2 = np.abs(self.model_results[model2].residuals)
                
                t_stat, t_pval = self.stats.ttest_rel(errors1, errors2)
                
                pairwise_results[f"{model1}_vs_{model2}"] = {
                    'diebold_mariano': dm_result,
                    'paired_t_test': {'statistic': float(t_stat), 'p_value': float(t_pval)}
                }
        
        return pairwise_results
    
    def _diebold_mariano_test(self, residuals1: np.ndarray, residuals2: np.ndarray) -> Dict[str, float]:
        """Perform Diebold-Mariano test for forecast accuracy."""
        # Calculate loss differential
        loss_diff = residuals1**2 - residuals2**2
        
        # Mean of loss differential
        mean_diff = np.mean(loss_diff)
        
        # Standard error (accounting for autocorrelation)
        n = len(loss_diff)
        gamma0 = np.var(loss_diff)
        
        # Simple version without autocorrelation correction
        se = np.sqrt(gamma0 / n)
        
        if se > 0:
            dm_stat = mean_diff / se
            p_value = 2 * (1 - self.stats.norm.cdf(np.abs(dm_stat)))
        else:
            dm_stat, p_value = 0.0, 1.0
        
        return {'statistic': float(dm_stat), 'p_value': float(p_value)}
    
    def _perform_friedman_test(self, model_names: List[str]) -> Dict[str, float]:
        """Perform Friedman test for multiple model comparison."""
        # Create rankings for each observation
        n_models = len(model_names)
        n_obs = len(next(iter(self.model_results.values())).residuals)
        
        rankings = np.zeros((n_obs, n_models))
        
        for i in range(n_obs):
            # Get absolute errors for this observation
            errors = [np.abs(self.model_results[model].residuals[i]) for model in model_names]
            
            # Rank errors (1 = best, n_models = worst)
            ranks = self.stats.rankdata(errors)
            rankings[i, :] = ranks
        
        # Perform Friedman test
        try:
            statistic, p_value = self.stats.friedmanchisquare(*rankings.T)
            return {'statistic': float(statistic), 'p_value': float(p_value)}
        except:
            return {'statistic': np.nan, 'p_value': np.nan}
    
    def _perform_residual_tests(self) -> Dict[str, Any]:
        """Perform residual analysis tests for each model."""
        residual_tests = {}
        
        for model_name, result in self.model_results.items():
            residuals = result.residuals
            
            # Normality test (Shapiro-Wilk)
            if len(residuals) <= 5000:  # Shapiro-Wilk limitation
                shapiro_stat, shapiro_pval = self.stats.shapiro(residuals)
            else:
                shapiro_stat, shapiro_pval = np.nan, np.nan
            
            # Ljung-Box test for autocorrelation
            ljung_box = self._ljung_box_test(residuals)
            
            residual_tests[model_name] = {
                'normality_test': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_pval)},
                'autocorrelation_test': ljung_box
            }
        
        return residual_tests
    
    def _ljung_box_test(self, residuals: np.ndarray, lags: int = 10) -> Dict[str, float]:
        """Ljung-Box test for autocorrelation."""
        try:
            n = len(residuals)
            acf_values = []
            
            for lag in range(1, min(lags + 1, n)):
                if lag < n:
                    acf = np.corrcoef(residuals[:-lag], residuals[lag:])[0, 1]
                    acf_values.append(acf if not np.isnan(acf) else 0.0)
            
            # Ljung-Box statistic
            statistic = n * (n + 2) * np.sum([acf**2 / (n - lag) for lag, acf in enumerate(acf_values, 1)])
            p_value = 1 - self.stats.chi2.cdf(statistic, lags)
            
            return {'statistic': float(statistic), 'p_value': float(p_value)}
        except:
            return {'statistic': np.nan, 'p_value': np.nan}
    
    def generate_comparison_report(self, 
                                  output_path: Optional[str] = None) -> ComparisonResults:
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            ComparisonResults containing all analysis
        """
        if not self.model_results:
            raise ValueError("No model results available for comparison")
        
        # Perform rankings
        rankings = self.rank_models()
        
        # Perform statistical tests
        statistical_tests = self.perform_statistical_tests()
        
        # Calculate summary statistics
        summary_stats = self._calculate_summary_statistics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(rankings, summary_stats)
        
        # Create comparison results
        comparison_results = ComparisonResults(
            model_results=self.model_results,
            rankings=rankings,
            statistical_tests=statistical_tests,
            summary_statistics=summary_stats,
            recommendations=recommendations
        )
        
        # Save report if path provided
        if output_path:
            self._save_report(comparison_results, output_path)
        
        return comparison_results
    
    def _calculate_summary_statistics(self) -> Dict[str, Any]:
        """Calculate summary statistics across all models."""
        summary = {
            'n_models': len(self.model_results),
            'metrics_summary': {},
            'efficiency_summary': {},
            'model_complexity': {}
        }
        
        # Metrics summary
        for metric in self.evaluation_metrics:
            values = [result.metrics.get(metric, np.nan) for result in self.model_results.values()]
            values = [v for v in values if not np.isnan(v)]
            
            if values:
                summary['metrics_summary'][metric] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'range': float(np.max(values) - np.min(values))
                }
        
        # Efficiency summary (training times)
        training_times = [result.training_time for result in self.model_results.values()]
        summary['efficiency_summary'] = {
            'mean_training_time': float(np.mean(training_times)),
            'std_training_time': float(np.std(training_times)),
            'fastest_model': min(self.model_results.keys(), 
                               key=lambda k: self.model_results[k].training_time),
            'slowest_model': max(self.model_results.keys(), 
                               key=lambda k: self.model_results[k].training_time)
        }
        
        # Model complexity (if available)
        for model_name, result in self.model_results.items():
            model_info = result.model_info
            complexity_info = {}
            
            # Extract complexity measures
            if 'model_parameters' in model_info:
                complexity_info['parameters'] = model_info['model_parameters']
            if 'n_features' in model_info:
                complexity_info['features'] = model_info['n_features']
            if 'n_estimators' in model_info:
                complexity_info['estimators'] = model_info['n_estimators']
            
            if complexity_info:
                summary['model_complexity'][model_name] = complexity_info
        
        return summary
    
    def _generate_recommendations(self, 
                                rankings: Dict[str, List[str]], 
                                summary_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Best overall performer
        if 'composite' in rankings:
            best_overall = rankings['composite'][0]
            recommendations.append(f"Best overall performer: {best_overall}")
        else:
            # Find most frequently top-ranked model
            top_positions = {}
            for metric_ranking in rankings.values():
                if metric_ranking:
                    top_model = metric_ranking[0]
                    top_positions[top_model] = top_positions.get(top_model, 0) + 1
            
            if top_positions:
                best_overall = max(top_positions.keys(), key=lambda k: top_positions[k])
                recommendations.append(f"Most consistently top performer: {best_overall}")
        
        # Efficiency recommendations
        efficiency = summary_stats.get('efficiency_summary', {})
        if 'fastest_model' in efficiency:
            fastest = efficiency['fastest_model']
            recommendations.append(f"Fastest training model: {fastest}")
        
        # Accuracy recommendations
        if 'rmse' in rankings:
            most_accurate = rankings['rmse'][0]
            recommendations.append(f"Most accurate model (RMSE): {most_accurate}")
        
        # Balanced recommendation
        if len(self.model_results) >= 3:
            # Find model that's consistently in top 50%
            consistent_performers = []
            n_models = len(self.model_results)
            threshold = n_models // 2
            
            for model in self.model_results.keys():
                avg_rank = 0
                count = 0
                for ranking in rankings.values():
                    if model in ranking:
                        avg_rank += ranking.index(model)
                        count += 1
                
                if count > 0 and avg_rank / count <= threshold:
                    consistent_performers.append(model)
            
            if consistent_performers:
                recommendations.append(f"Consistent performers: {', '.join(consistent_performers)}")
        
        return recommendations
    
    def _save_report(self, results: ComparisonResults, output_path: str):
        """Save comparison report to file."""
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("TIME SERIES MODEL COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Models compared: {len(results.model_results)}")
        report_lines.append("")
        
        # Model Performance Summary
        report_lines.append("MODEL PERFORMANCE SUMMARY")
        report_lines.append("-" * 40)
        
        for model_name, result in results.model_results.items():
            report_lines.append(f"\n{model_name}:")
            for metric, value in result.metrics.items():
                report_lines.append(f"  {metric.upper()}: {value:.4f}")
            report_lines.append(f"  Training Time: {result.training_time:.2f}s")
        
        # Rankings
        report_lines.append("\n\nMODEL RANKINGS")
        report_lines.append("-" * 40)
        
        for metric, ranking in results.rankings.items():
            report_lines.append(f"\n{metric.upper()} Ranking:")
            for i, model in enumerate(ranking, 1):
                report_lines.append(f"  {i}. {model}")
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDAT IONS")
        report_lines.append("-" * 40)
        
        for recommendation in results.recommendations:
            report_lines.append(f"• {recommendation}")
        
        # Statistical Tests Summary
        if results.statistical_tests:
            report_lines.append("\n\nSTATISTICAL TESTS")
            report_lines.append("-" * 40)
            
            pairwise = results.statistical_tests.get('pairwise_tests', {})
            for comparison, tests in pairwise.items():
                report_lines.append(f"\n{comparison}:")
                dm_test = tests.get('diebold_mariano', {})
                if 'p_value' in dm_test:
                    significant = "significant" if dm_test['p_value'] < 0.05 else "not significant"
                    report_lines.append(f"  Diebold-Mariano: {significant} (p={dm_test['p_value']:.4f})")
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"Comparison report saved to {output_path}")
    
    def plot_comparison_dashboard(self, 
                                y_true: np.ndarray,
                                timestamps: Optional[np.ndarray] = None,
                                title: str = "Model Comparison Dashboard"):
        """
        Create comprehensive comparison dashboard.
        
        Args:
            y_true: True values for plotting
            timestamps: Optional timestamps
            title: Dashboard title
        """
        if not hasattr(self, 'visualizer'):
            print("Visualizer not available")
            return
        
        # Prepare data for visualization
        predictions = {name: result.predictions for name, result in self.model_results.items()}
        metrics = {name: result.metrics for name, result in self.model_results.items()}
        
        # Create dashboard
        self.visualizer.create_model_comparison_dashboard(
            y_true=y_true,
            predictions=predictions,
            metrics=metrics,
            timestamps=timestamps,
            title=title
        )
    
    def export_results_to_csv(self, output_path: str):
        """
        Export comparison results to CSV format.
        
        Args:
            output_path: Path for CSV output
        """
        # Create DataFrame with all metrics
        data = []
        for model_name, result in self.model_results.items():
            row = {'model_name': model_name, 'training_time': result.training_time}
            row.update(result.metrics)
            data.append(row)
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Results exported to {output_path}")
    
    def clear_results(self):
        """Clear all stored results."""
        self.model_results.clear()
        print("All results cleared")