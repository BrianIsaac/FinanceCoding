"""
Training Convergence and Hyperparameter Validation System.

This script implements comprehensive training metrics tracking, hyperparameter optimization,
training stability validation, and diagnostic reporting for all three ML approaches,
following Story 5.2 Task 5 requirements.
"""

import logging
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid

warnings.filterwarnings("ignore", category=UserWarning)


class TrainingConvergenceValidator:
    """
    Comprehensive training convergence and hyperparameter validation system.

    Implements all subtasks from Story 5.2 Task 5:
    - Comprehensive training metrics tracking (loss convergence, validation performance, early stopping)
    - Hyperparameter optimization using grid search across all model types
    - Training stability and performance consistency across multiple random seeds
    - Training diagnostic reports with convergence analysis and performance validation metrics
    """

    def __init__(self, results_dir: str = "logs/training/convergence"):
        """
        Initialize training convergence validator.

        Args:
            results_dir: Directory for storing validation results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Training metrics storage
        self.convergence_results: dict[str, Any] = {}
        self.hyperparameter_results: dict[str, Any] = {}
        self.stability_results: dict[str, Any] = {}

    def _setup_logging(self):
        """Setup logging for convergence validator."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.results_dir / "convergence_validation.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Training Convergence Validator initialized")

    def track_training_metrics(self,
                             model_type: str,
                             training_history: dict[str, list[float]],
                             validation_history: dict[str, list[float]],
                             early_stopping_info: dict[str, Any],
                             config_id: str) -> dict[str, Any]:
        """
        Track comprehensive training metrics for convergence analysis.

        Args:
            model_type: Type of model (hrp, lstm, gat)
            training_history: Training loss and metrics history
            validation_history: Validation metrics history
            early_stopping_info: Early stopping configuration and results
            config_id: Configuration identifier

        Returns:
            Convergence analysis results
        """
        self.logger.info(f"Tracking training metrics for {model_type} - {config_id}")

        # Analyze training convergence
        convergence_analysis = self._analyze_convergence(
            training_history, validation_history
        )

        # Early stopping analysis
        early_stopping_analysis = self._analyze_early_stopping(
            early_stopping_info, training_history, validation_history
        )

        # Validation performance analysis
        validation_analysis = self._analyze_validation_performance(
            validation_history
        )

        results = {
            'model_type': model_type,
            'config_id': config_id,
            'convergence_analysis': convergence_analysis,
            'early_stopping_analysis': early_stopping_analysis,
            'validation_analysis': validation_analysis,
            'training_epochs': len(training_history.get('loss', [])),
            'final_training_loss': training_history.get('loss', [])[-1] if training_history.get('loss') else None,
            'best_validation_metric': self._get_best_validation_metric(validation_history),
            'convergence_quality': self._assess_convergence_quality(convergence_analysis)
        }

        # Store results
        if model_type not in self.convergence_results:
            self.convergence_results[model_type] = {}
        self.convergence_results[model_type][config_id] = results

        return results

    def _analyze_convergence(self,
                           training_history: dict[str, list[float]],
                           validation_history: dict[str, list[float]]) -> dict[str, Any]:
        """Analyze training convergence patterns."""
        analysis = {}

        # Training loss convergence
        if 'loss' in training_history and training_history['loss']:
            losses = training_history['loss']
            analysis['loss_convergence'] = {
                'final_loss': losses[-1],
                'min_loss': min(losses),
                'loss_reduction': losses[0] - losses[-1] if len(losses) > 1 else 0,
                'convergence_rate': self._calculate_convergence_rate(losses),
                'plateau_detection': self._detect_plateau(losses),
                'oscillation_measure': self._calculate_oscillation(losses)
            }

        # Validation metric convergence
        for metric_name, values in validation_history.items():
            if values:
                analysis[f'{metric_name}_convergence'] = {
                    'final_value': values[-1],
                    'best_value': max(values) if 'accuracy' in metric_name or 'sharpe' in metric_name else min(values),
                    'improvement_trend': self._calculate_improvement_trend(values),
                    'stability_measure': np.std(values[-10:]) if len(values) >= 10 else np.std(values)
                }

        return analysis

    def _analyze_early_stopping(self,
                              early_stopping_info: dict[str, Any],
                              training_history: dict[str, list[float]],
                              validation_history: dict[str, list[float]]) -> dict[str, Any]:
        """Analyze early stopping effectiveness."""
        return {
            'triggered': early_stopping_info.get('triggered', False),
            'patience': early_stopping_info.get('patience', None),
            'best_epoch': early_stopping_info.get('best_epoch', None),
            'epochs_without_improvement': early_stopping_info.get('epochs_without_improvement', None),
            'prevented_overfitting': self._assess_overfitting_prevention(
                training_history, validation_history
            )
        }

    def _analyze_validation_performance(self,
                                      validation_history: dict[str, list[float]]) -> dict[str, Any]:
        """Analyze validation performance characteristics."""
        analysis = {}

        for metric_name, values in validation_history.items():
            if values:
                analysis[metric_name] = {
                    'best_performance': max(values) if 'accuracy' in metric_name or 'sharpe' in metric_name else min(values),
                    'final_performance': values[-1],
                    'performance_stability': np.std(values),
                    'improvement_epochs': len([i for i in range(1, len(values)) if values[i] > values[i-1]]),
                    'degradation_epochs': len([i for i in range(1, len(values)) if values[i] < values[i-1]])
                }

        return analysis

    def optimize_hyperparameters(self,
                                model_type: str,
                                hyperparameter_space: dict[str, list[Any]],
                                training_function: callable,
                                validation_function: callable,
                                n_trials: Optional[int] = None) -> dict[str, Any]:
        """
        Execute hyperparameter optimization using grid search.

        Args:
            model_type: Type of model to optimize
            hyperparameter_space: Dictionary defining hyperparameter search space
            training_function: Function to train model with given hyperparameters
            validation_function: Function to validate trained model
            n_trials: Maximum number of trials (None for full grid search)

        Returns:
            Hyperparameter optimization results
        """
        self.logger.info(f"Starting hyperparameter optimization for {model_type}")

        # Generate parameter grid
        param_grid = list(ParameterGrid(hyperparameter_space))
        if n_trials and n_trials < len(param_grid):
            # Random sample if n_trials specified
            np.random.shuffle(param_grid)
            param_grid = param_grid[:n_trials]

        optimization_results = {
            'model_type': model_type,
            'search_space': hyperparameter_space,
            'total_trials': len(param_grid),
            'trial_results': [],
            'best_config': None,
            'best_performance': None,
            'performance_distribution': {}
        }

        for trial_idx, params in enumerate(param_grid):
            self.logger.info(f"Trial {trial_idx + 1}/{len(param_grid)}: {params}")

            try:
                # Train model with current parameters
                model, training_history = training_function(params)

                # Validate model performance
                validation_results = validation_function(model)

                trial_result = {
                    'trial_id': trial_idx,
                    'parameters': params,
                    'training_history': training_history,
                    'validation_results': validation_results,
                    'performance_metric': validation_results.get('primary_metric', 0),
                    'training_time': validation_results.get('training_time', 0),
                    'convergence_quality': self._assess_convergence_quality(
                        self._analyze_convergence(training_history, validation_results.get('validation_history', {}))
                    )
                }

                optimization_results['trial_results'].append(trial_result)

                # Update best configuration
                current_performance = trial_result['performance_metric']
                if (optimization_results['best_performance'] is None or
                    current_performance > optimization_results['best_performance']):
                    optimization_results['best_config'] = params
                    optimization_results['best_performance'] = current_performance

            except Exception as e:
                self.logger.error(f"Trial {trial_idx} failed: {e}")
                continue

        # Analyze performance distribution
        if optimization_results['trial_results']:
            performances = [r['performance_metric'] for r in optimization_results['trial_results']]
            optimization_results['performance_distribution'] = {
                'mean': np.mean(performances),
                'std': np.std(performances),
                'min': np.min(performances),
                'max': np.max(performances),
                'median': np.median(performances)
            }

        # Store results
        self.hyperparameter_results[model_type] = optimization_results

        self.logger.info(f"Hyperparameter optimization completed for {model_type}")
        return optimization_results

    def validate_training_stability(self,
                                  model_type: str,
                                  config: dict[str, Any],
                                  training_function: callable,
                                  validation_function: callable,
                                  n_seeds: int = 5) -> dict[str, Any]:
        """
        Validate training stability across multiple random seeds.

        Args:
            model_type: Type of model to test
            config: Model configuration
            training_function: Function to train model
            validation_function: Function to validate model
            n_seeds: Number of random seeds to test

        Returns:
            Training stability analysis results
        """
        self.logger.info(f"Validating training stability for {model_type} across {n_seeds} seeds")

        stability_results = {
            'model_type': model_type,
            'config': config,
            'n_seeds': n_seeds,
            'seed_results': [],
            'stability_metrics': {},
            'consistency_analysis': {}
        }

        # Test across different random seeds
        seed_performances = []
        seed_convergences = []

        for seed in range(n_seeds):
            self.logger.info(f"Testing seed {seed + 1}/{n_seeds}")

            try:
                # Set random seed
                np.random.seed(seed)

                # Train model
                model, training_history = training_function(config, random_seed=seed)

                # Validate model
                validation_results = validation_function(model)

                seed_result = {
                    'seed': seed,
                    'training_history': training_history,
                    'validation_results': validation_results,
                    'final_performance': validation_results.get('primary_metric', 0),
                    'convergence_epochs': len(training_history.get('loss', [])),
                    'final_loss': training_history.get('loss', [])[-1] if training_history.get('loss') else None
                }

                stability_results['seed_results'].append(seed_result)
                seed_performances.append(seed_result['final_performance'])
                seed_convergences.append(seed_result['convergence_epochs'])

            except Exception as e:
                self.logger.error(f"Seed {seed} failed: {e}")
                continue

        # Calculate stability metrics
        if seed_performances:
            stability_results['stability_metrics'] = {
                'performance_mean': np.mean(seed_performances),
                'performance_std': np.std(seed_performances),
                'performance_cv': np.std(seed_performances) / np.mean(seed_performances) if np.mean(seed_performances) != 0 else float('inf'),
                'convergence_mean': np.mean(seed_convergences),
                'convergence_std': np.std(seed_convergences),
                'success_rate': len(seed_performances) / n_seeds
            }

            # Consistency analysis
            stability_results['consistency_analysis'] = self._analyze_consistency(
                seed_performances, seed_convergences
            )

        # Store results
        if model_type not in self.stability_results:
            self.stability_results[model_type] = {}
        self.stability_results[model_type]['stability_test'] = stability_results

        return stability_results

    def generate_training_diagnostic_report(self, model_type: str) -> dict[str, Any]:
        """
        Generate comprehensive training diagnostic report.

        Args:
            model_type: Type of model to report on

        Returns:
            Comprehensive diagnostic report
        """
        self.logger.info(f"Generating training diagnostic report for {model_type}")

        report = {
            'model_type': model_type,
            'generated_at': pd.Timestamp.now().isoformat(),
            'convergence_summary': {},
            'hyperparameter_summary': {},
            'stability_summary': {},
            'recommendations': [],
            'quality_assessment': {}
        }

        # Convergence summary
        if model_type in self.convergence_results:
            convergence_data = self.convergence_results[model_type]
            report['convergence_summary'] = {
                'total_configurations': len(convergence_data),
                'successful_convergences': sum(1 for r in convergence_data.values()
                                             if r['convergence_quality'] >= 0.7),
                'average_convergence_quality': np.mean([r['convergence_quality']
                                                      for r in convergence_data.values()]),
                'best_configuration': max(convergence_data.keys(),
                                        key=lambda k: convergence_data[k]['convergence_quality'])
            }

        # Hyperparameter summary
        if model_type in self.hyperparameter_results:
            hp_data = self.hyperparameter_results[model_type]
            report['hyperparameter_summary'] = {
                'total_trials': hp_data.get('total_trials', 0),
                'best_performance': hp_data.get('best_performance'),
                'best_config': hp_data.get('best_config'),
                'performance_distribution': hp_data.get('performance_distribution', {})
            }

        # Stability summary
        if model_type in self.stability_results:
            stability_data = self.stability_results[model_type].get('stability_test', {})
            report['stability_summary'] = {
                'seeds_tested': stability_data.get('n_seeds', 0),
                'stability_metrics': stability_data.get('stability_metrics', {}),
                'consistency_rating': self._rate_consistency(stability_data.get('consistency_analysis', {}))
            }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(model_type, report)

        # Overall quality assessment
        report['quality_assessment'] = self._assess_overall_quality(report)

        # Save report
        report_path = self.results_dir / f"{model_type}_diagnostic_report.yaml"
        with open(report_path, 'w') as f:
            yaml.dump(report, f, default_flow_style=False)

        self.logger.info(f"Diagnostic report saved: {report_path}")
        return report

    def _calculate_convergence_rate(self, losses: list[float]) -> float:
        """Calculate convergence rate from loss history."""
        if len(losses) < 2:
            return 0.0

        # Calculate exponential decay rate
        x = np.arange(len(losses))
        y = np.log(np.maximum(losses, 1e-10))  # Avoid log(0)

        try:
            slope = np.polyfit(x, y, 1)[0]
            return -slope  # Negative slope indicates convergence
        except:
            return 0.0

    def _detect_plateau(self, values: list[float], window: int = 10, threshold: float = 0.001) -> bool:
        """Detect if training has plateaued."""
        if len(values) < window:
            return False

        recent_values = values[-window:]
        return np.std(recent_values) < threshold

    def _calculate_oscillation(self, values: list[float]) -> float:
        """Calculate oscillation measure in training."""
        if len(values) < 3:
            return 0.0

        differences = np.diff(values)
        sign_changes = np.sum(np.diff(np.sign(differences)) != 0)
        return sign_changes / len(differences)

    def _calculate_improvement_trend(self, values: list[float]) -> float:
        """Calculate improvement trend (positive = improving)."""
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope

    def _assess_overfitting_prevention(self,
                                     training_history: dict[str, list[float]],
                                     validation_history: dict[str, list[float]]) -> bool:
        """Assess if early stopping prevented overfitting."""
        train_losses = training_history.get('loss', [])
        val_losses = validation_history.get('loss', [])

        if not train_losses or not val_losses:
            return False

        # Check if validation loss started increasing while training loss decreased
        min_len = min(len(train_losses), len(val_losses))
        if min_len < 10:
            return False

        # Look at last 10 epochs
        recent_train = train_losses[-10:]
        recent_val = val_losses[-10:]

        train_trend = self._calculate_improvement_trend(recent_train)
        val_trend = self._calculate_improvement_trend(recent_val)

        # Overfitting prevented if training improved but validation degraded
        return train_trend < 0 and val_trend > 0  # train loss decreasing, val loss increasing

    def _get_best_validation_metric(self, validation_history: dict[str, list[float]]) -> dict[str, float]:
        """Get best validation metric values."""
        best_metrics = {}
        for metric_name, values in validation_history.items():
            if values:
                if 'accuracy' in metric_name or 'sharpe' in metric_name:
                    best_metrics[metric_name] = max(values)
                else:
                    best_metrics[metric_name] = min(values)
        return best_metrics

    def _assess_convergence_quality(self, convergence_analysis: dict[str, Any]) -> float:
        """Assess overall convergence quality (0-1 score)."""
        quality_score = 0.0
        factors = 0

        if 'loss_convergence' in convergence_analysis:
            loss_conv = convergence_analysis['loss_convergence']

            # Convergence rate factor
            if loss_conv.get('convergence_rate', 0) > 0:
                quality_score += min(loss_conv['convergence_rate'] / 0.1, 1.0) * 0.3
            factors += 0.3

            # Loss reduction factor
            if loss_conv.get('loss_reduction', 0) > 0:
                quality_score += min(loss_conv['loss_reduction'] / 1.0, 1.0) * 0.2
            factors += 0.2

            # Low oscillation factor
            oscillation = loss_conv.get('oscillation_measure', 1.0)
            quality_score += (1.0 - min(oscillation / 0.5, 1.0)) * 0.2
            factors += 0.2

            # Plateau detection (good if not plateaued early)
            if not loss_conv.get('plateau_detection', False):
                quality_score += 0.3
            factors += 0.3

        return quality_score / factors if factors > 0 else 0.0

    def _analyze_consistency(self, performances: list[float], convergences: list[int]) -> dict[str, Any]:
        """Analyze consistency across different seeds."""
        return {
            'performance_consistency': 'high' if np.std(performances) / np.mean(performances) < 0.1 else
                                     'medium' if np.std(performances) / np.mean(performances) < 0.2 else 'low',
            'convergence_consistency': 'high' if np.std(convergences) / np.mean(convergences) < 0.2 else
                                     'medium' if np.std(convergences) / np.mean(convergences) < 0.4 else 'low',
            'outlier_count': len([p for p in performances if abs(p - np.mean(performances)) > 2 * np.std(performances)])
        }

    def _rate_consistency(self, consistency_analysis: dict[str, Any]) -> str:
        """Rate overall consistency."""
        perf_rating = consistency_analysis.get('performance_consistency', 'low')
        conv_rating = consistency_analysis.get('convergence_consistency', 'low')

        rating_scores = {'high': 3, 'medium': 2, 'low': 1}
        avg_score = (rating_scores.get(perf_rating, 1) + rating_scores.get(conv_rating, 1)) / 2

        if avg_score >= 2.5:
            return 'excellent'
        elif avg_score >= 2.0:
            return 'good'
        elif avg_score >= 1.5:
            return 'fair'
        else:
            return 'poor'

    def _generate_recommendations(self, model_type: str, report: dict[str, Any]) -> list[str]:
        """Generate recommendations based on diagnostic report."""
        recommendations = []

        # Convergence recommendations
        conv_summary = report.get('convergence_summary', {})
        if conv_summary.get('average_convergence_quality', 0) < 0.5:
            recommendations.append("Consider adjusting learning rate or optimizer settings to improve convergence quality")

        # Stability recommendations
        stability_summary = report.get('stability_summary', {})
        consistency_rating = stability_summary.get('consistency_rating', 'poor')
        if consistency_rating in ['fair', 'poor']:
            recommendations.append("Training shows inconsistency across seeds - consider regularization or different initialization")

        # Hyperparameter recommendations
        hp_summary = report.get('hyperparameter_summary', {})
        if hp_summary.get('total_trials', 0) > 0:
            best_performance = hp_summary.get('best_performance', 0)
            if best_performance < 0.1:  # Assuming higher is better
                recommendations.append("Consider expanding hyperparameter search space or trying different architectures")

        if not recommendations:
            recommendations.append("Training performance looks good - consider production deployment")

        return recommendations

    def _assess_overall_quality(self, report: dict[str, Any]) -> dict[str, Any]:
        """Assess overall training quality."""
        quality_factors = []

        # Convergence quality
        conv_quality = report.get('convergence_summary', {}).get('average_convergence_quality', 0)
        quality_factors.append(conv_quality)

        # Stability quality
        stability_rating = report.get('stability_summary', {}).get('consistency_rating', 'poor')
        stability_score = {'excellent': 1.0, 'good': 0.8, 'fair': 0.6, 'poor': 0.4}.get(stability_rating, 0.4)
        quality_factors.append(stability_score)

        # Performance quality
        hp_summary = report.get('hyperparameter_summary', {})
        if 'best_performance' in hp_summary:
            perf_quality = min(hp_summary['best_performance'], 1.0)  # Normalize if needed
            quality_factors.append(perf_quality)

        overall_score = np.mean(quality_factors) if quality_factors else 0.0

        return {
            'overall_score': overall_score,
            'rating': 'excellent' if overall_score >= 0.8 else
                     'good' if overall_score >= 0.6 else
                     'fair' if overall_score >= 0.4 else 'poor',
            'quality_factors': quality_factors
        }


def main():
    """Main execution function for training convergence validation."""

    # Initialize convergence validator
    validator = TrainingConvergenceValidator()

    # Demonstrate comprehensive validation for each model type
    model_types = ['hrp', 'lstm', 'gat']

    for model_type in model_types:

        # Simulate training metrics tracking
        sample_training_history = {
            'loss': [1.0 - 0.1*i + 0.01*np.sin(i) for i in range(50)],
            'accuracy': [0.5 + 0.01*i for i in range(50)]
        }

        sample_validation_history = {
            'loss': [0.9 - 0.08*i + 0.02*np.sin(i) for i in range(50)],
            'accuracy': [0.6 + 0.008*i for i in range(50)],
            'sharpe': [0.1 + 0.002*i for i in range(50)]
        }

        sample_early_stopping = {
            'triggered': True,
            'patience': 10,
            'best_epoch': 42,
            'epochs_without_improvement': 8
        }

        # Track training metrics
        validator.track_training_metrics(
            model_type=model_type,
            training_history=sample_training_history,
            validation_history=sample_validation_history,
            early_stopping_info=sample_early_stopping,
            config_id=f"{model_type}_default_config"
        )


        # Generate diagnostic report
        validator.generate_training_diagnostic_report(model_type)





if __name__ == "__main__":
    main()
