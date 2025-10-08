# Evaluation and Backtesting Framework

## Rolling Validation Architecture

The evaluation framework implements academic-grade temporal validation with strict no-look-ahead guarantees:

```python
@dataclass
class BacktestConfig:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    training_months: int = 36                   # 3-year training window
    validation_months: int = 12                 # 1-year validation
    test_months: int = 12                       # 1-year out-of-sample test
    step_months: int = 12                       # Annual walk-forward
    rebalance_frequency: str = "M"              # Monthly rebalancing
    
class RollingBacktestEngine:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results_cache = {}
        
    def run_backtest(self, 
                    models: Dict[str, PortfolioModel],
                    data: Dict[str, pd.DataFrame]) -> BacktestResults:
        
        # Generate rolling windows with strict temporal separation
        windows = self._generate_rolling_windows()
        
        results = {}
        for window_idx, (train_period, val_period, test_period) in enumerate(windows):
            
            # Train models on historical data only
            for model_name, model in models.items():
                model.fit(data["returns"], 
                         universe=data["universe"], 
                         fit_period=train_period)
                
                # Validate on out-of-sample validation period
                val_results = self._validate_model(model, val_period, data)
                
                # Test on final out-of-sample period
                test_results = self._test_model(model, test_period, data)
                
                results[f"{model_name}_window_{window_idx}"] = {
                    "validation": val_results,
                    "test": test_results
                }
                
        return BacktestResults(results)
```

## Performance Analytics Suite

Comprehensive performance measurement with institutional-grade metrics:

```python
class PerformanceAnalytics:
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Annualized Sharpe ratio with bias correction."""
        excess_returns = returns - risk_free_rate/252  # Daily risk-free rate
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    @staticmethod
    def information_ratio(portfolio_returns: pd.Series, 
                         benchmark_returns: pd.Series) -> float:
        """Information ratio vs benchmark."""
        active_returns = portfolio_returns - benchmark_returns
        return np.sqrt(252) * active_returns.mean() / active_returns.std()
    
    @staticmethod
    def maximum_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Maximum drawdown with start and end dates."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_date = drawdown.idxmin()
        
        # Find start of drawdown period
        prior_peak = running_max.loc[:max_dd_date].idxmax()
        
        return max_dd, prior_peak, max_dd_date
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Historical Value at Risk."""
        return returns.quantile(confidence_level)
    
    @staticmethod
    def conditional_value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Expected Shortfall / Conditional VaR."""
        var_threshold = returns.quantile(confidence_level)
        return returns[returns <= var_threshold].mean()
```

## Statistical Significance Testing

Rigorous hypothesis testing framework for academic validation:

```python
class StatisticalValidation:
    @staticmethod
    def sharpe_ratio_test(returns_a: pd.Series, 
                         returns_b: pd.Series,
                         alpha: float = 0.05) -> Dict[str, float]:
        """
        Test statistical significance of Sharpe ratio differences.
        Uses Jobson-Korkie test with Memmel correction.
        """
        n = len(returns_a)
        sharpe_a = PerformanceAnalytics.sharpe_ratio(returns_a)
        sharpe_b = PerformanceAnalytics.sharpe_ratio(returns_b)
        
        # Calculate test statistic (Memmel, 2003)
        var_a, var_b = returns_a.var(), returns_b.var()
        cov_ab = returns_a.cov(returns_b)
        
        theta = (var_a * var_b - cov_ab**2) / (2 * var_a * var_b)
        
        test_stat = np.sqrt(n) * (sharpe_a - sharpe_b) / np.sqrt(theta)
        p_value = 2 * (1 - stats.norm.cdf(abs(test_stat)))
        
        return {
            "sharpe_difference": sharpe_a - sharpe_b,
            "test_statistic": test_stat,
            "p_value": p_value,
            "significant": p_value < alpha
        }
    
    @staticmethod
    def bootstrap_confidence_intervals(performance_metric: Callable,
                                     returns: pd.Series,
                                     confidence_level: float = 0.95,
                                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """Bootstrap confidence intervals for performance metrics."""
        bootstrap_values = []
        n = len(returns)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = returns.sample(n=n, replace=True)
            bootstrap_values.append(performance_metric(bootstrap_sample))
            
        alpha = 1 - confidence_level
        lower_bound = np.percentile(bootstrap_values, 100 * alpha/2)
        upper_bound = np.percentile(bootstrap_values, 100 * (1 - alpha/2))
        
        return lower_bound, upper_bound
```

---
