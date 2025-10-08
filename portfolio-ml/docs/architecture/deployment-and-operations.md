# Deployment and Operations

## Production Deployment Guide

```python
# deployment/production_config.py
@dataclass
class ProductionConfig:
    # Environment settings
    environment: str = "production"
    log_level: str = "INFO"
    enable_monitoring: bool = True
    
    # Data refresh settings  
    daily_data_update_time: str = "06:00"  # UTC
    monthly_rebalance_day: int = -1        # Last business day
    
    # Model serving settings
    model_inference_timeout_seconds: int = 300
    batch_size_limit: int = 1000
    
    # Risk management
    position_limit_checks: bool = True
    turnover_limit_enforcement: bool = True
    realtime_risk_monitoring: bool = True
    
    # Performance monitoring
    performance_tracking: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "max_daily_loss": -0.05,
        "max_drawdown": -0.25,
        "min_sharpe_ratio": 0.5
    })

class ProductionPortfolioManager:
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.models = self._load_trained_models()
        self.risk_manager = RealTimeRiskManager(config)
        
    def generate_monthly_portfolio(self, 
                                 rebalance_date: pd.Timestamp) -> Dict[str, float]:
        """Generate portfolio for monthly rebalancing."""
        
        # Get current universe and market data
        universe = self._get_active_universe(rebalance_date)
        market_data = self._get_latest_market_data(universe, rebalance_date)
        
        # Generate model predictions
        model_portfolios = {}
        for model_name, model in self.models.items():
            try:
                weights = model.predict_weights(rebalance_date, universe)
                model_portfolios[model_name] = weights
            except Exception as e:
                logging.error(f"Model {model_name} failed: {e}")
                continue
        
        # Select best performing model or use ensemble
        final_portfolio = self._select_portfolio(model_portfolios)
        
        # Risk management checks
        validated_portfolio = self.risk_manager.validate_portfolio(final_portfolio)
        
        return validated_portfolio
```

## Monitoring and Alerting

```python
class PerformanceMonitor:
    def __init__(self, alert_config: Dict[str, float]):
        self.alert_thresholds = alert_config
        self.performance_history = []
        
    def update_performance(self, date: pd.Timestamp, metrics: Dict[str, float]):
        """Update performance tracking and check alerts."""
        self.performance_history.append({
            'date': date,
            **metrics
        })
        
        # Check alert conditions
        self._check_alerts(metrics)
        
    def _check_alerts(self, metrics: Dict[str, float]):
        """Check if any metrics breach alert thresholds."""
        for metric_name, value in metrics.items():
            if metric_name in self.alert_thresholds:
                threshold = self.alert_thresholds[metric_name]
                if value < threshold:
                    self._send_alert(f"Performance alert: {metric_name} = {value:.4f} below threshold {threshold:.4f}")
    
    def _send_alert(self, message: str):
        """Send performance alert (email, Slack, etc.)."""
        logging.critical(f"PERFORMANCE ALERT: {message}")
        # Integration with alerting systems would go here
```

---
