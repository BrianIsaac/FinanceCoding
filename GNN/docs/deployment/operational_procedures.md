# Operational Procedures Documentation

## Overview

This document provides comprehensive operational procedures for managing the GNN Portfolio Optimization System in production. These procedures ensure consistent, reliable operations with institutional-grade risk management and compliance requirements.

## Daily Operations

### Morning System Check (Pre-Market)

#### System Health Verification
```bash
#!/bin/bash
# Daily health check script: scripts/daily_health_check.sh

echo "=== Daily Pre-Market System Check $(date) ==="

# 1. Service Status Check
echo "1. Checking service status..."
systemctl status gnn-portfolio.service --no-pager
if ! systemctl is-active --quiet gnn-portfolio.service; then
    echo "ALERT: GNN Portfolio service is not running"
    # Send alert to operations team
    curl -X POST "$SLACK_WEBHOOK" -d '{"text":"🚨 GNN Portfolio service is DOWN"}'
fi

# 2. GPU Health Check
echo -e "\n2. GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader
GPU_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
if [[ $GPU_TEMP -gt 85 ]]; then
    echo "WARNING: GPU temperature high ($GPU_TEMP°C)"
fi

# 3. Data Pipeline Status
echo -e "\n3. Data Pipeline Status:"
python src/utils/data_health_check.py
DATA_AGE=$(python -c "
import pandas as pd
from pathlib import Path
data_path = Path('/opt/gnn-portfolio/data/latest')
if data_path.exists():
    import time
    age_hours = (time.time() - data_path.stat().st_mtime) / 3600
    print(f'{age_hours:.1f}')
else:
    print('999')
")

if (( $(echo "$DATA_AGE > 24" | bc -l) )); then
    echo "WARNING: Data is $DATA_AGE hours old"
fi

# 4. Model Status Check
echo -e "\n4. Model Status:"
python src/utils/model_health_check.py --check-all

# 5. Disk Space Check
echo -e "\n5. Storage Status:"
df -h /opt/gnn-portfolio | tail -1
DISK_USAGE=$(df /opt/gnn-portfolio | tail -1 | awk '{print $5}' | sed 's/%//')
if [[ $DISK_USAGE -gt 80 ]]; then
    echo "WARNING: Disk usage high ($DISK_USAGE%)"
fi

echo -e "\n=== Pre-Market Check Complete ==="
```

#### Data Refresh Verification
```bash
# Verify data refresh completion
echo "=== Data Refresh Status ==="

# Check last update time
python << EOF
import pandas as pd
from src.data.loaders.portfolio_data import PortfolioDataLoader
from datetime import datetime, timedelta

loader = PortfolioDataLoader()
last_update = loader.get_last_update_time()
market_date = loader.get_last_market_date()

print(f"Last data update: {last_update}")
print(f"Last market date: {market_date}")

# Check if data is current
if (datetime.now() - last_update).total_seconds() > 86400:  # 24 hours
    print("❌ Data is stale")
    exit(1)
else:
    print("✅ Data is current")
EOF
```

### Market Data Collection

#### Automated Data Collection Process
```python
# src/operations/daily_data_collection.py
"""Daily market data collection and validation."""

import logging
from datetime import datetime, timedelta
from pathlib import Path

from src.data.collectors.yfinance import YFinanceCollector
from src.data.processors.data_quality_validator import DataQualityValidator
from src.data.processors.gap_filling import GapFiller
from src.utils.notifications import send_alert

logger = logging.getLogger(__name__)

class DailyDataCollector:
    """Manages daily data collection and validation process."""
    
    def __init__(self, config_path: str = "configs/production/data_config.yaml"):
        """Initialize daily data collector."""
        self.config = self._load_config(config_path)
        self.collector = YFinanceCollector()
        self.validator = DataQualityValidator()
        self.gap_filler = GapFiller()
    
    def run_daily_collection(self) -> bool:
        """Execute complete daily data collection process."""
        try:
            logger.info("Starting daily data collection")
            
            # 1. Collect latest market data
            collection_result = self._collect_market_data()
            if not collection_result:
                send_alert("Data collection failed", level="error")
                return False
            
            # 2. Validate data quality
            validation_result = self._validate_data_quality()
            if not validation_result:
                send_alert("Data validation failed", level="warning")
                
            # 3. Fill gaps if necessary
            gap_fill_result = self._fill_data_gaps()
            
            # 4. Generate data quality report
            self._generate_quality_report()
            
            logger.info("Daily data collection completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Daily data collection failed: {str(e)}")
            send_alert(f"Data collection error: {str(e)}", level="error")
            return False
    
    def _collect_market_data(self) -> bool:
        """Collect market data for current universe."""
        try:
            universe_symbols = self._get_current_universe()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)  # Get last week's data
            
            for symbol in universe_symbols:
                data = self.collector.collect_data(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is None or data.empty:
                    logger.warning(f"No data collected for {symbol}")
                    continue
                
                # Save to daily storage
                self._save_daily_data(symbol, data)
            
            return True
            
        except Exception as e:
            logger.error(f"Market data collection failed: {str(e)}")
            return False
    
    def _validate_data_quality(self) -> bool:
        """Validate quality of collected data."""
        validation_results = {}
        
        for data_file in Path("data/daily").glob("*.parquet"):
            data = pd.read_parquet(data_file)
            
            result = self.validator.validate_dataframe(data)
            validation_results[data_file.name] = result
            
            if not result.is_valid:
                logger.warning(f"Data quality issues in {data_file.name}: {result.issues}")
        
        return all(result.is_valid for result in validation_results.values())
```

#### Data Quality Monitoring
```bash
# Create data quality monitoring script
cat > scripts/monitor_data_quality.sh << EOF
#!/bin/bash

REPORT_DATE=\$(date +%Y-%m-%d)
REPORT_FILE="/opt/gnn-portfolio/reports/data_quality_\$REPORT_DATE.json"

echo "Generating data quality report for \$REPORT_DATE"

python << END_PYTHON
import json
from datetime import datetime
from src.data.processors.data_quality_validator import DataQualityValidator
from src.data.loaders.portfolio_data import PortfolioDataLoader

# Initialize components
validator = DataQualityValidator()
loader = PortfolioDataLoader()

# Load latest data
data = loader.load_latest_data()

# Run comprehensive validation
report = {
    "date": "$REPORT_DATE",
    "timestamp": datetime.now().isoformat(),
    "universe_size": len(data.columns),
    "data_coverage": {},
    "quality_metrics": {},
    "issues": []
}

# Validate each symbol
for symbol in data.columns:
    symbol_data = data[symbol].dropna()
    
    if len(symbol_data) == 0:
        report["issues"].append(f"No data for {symbol}")
        continue
    
    # Coverage metrics
    coverage = len(symbol_data) / len(data)
    report["data_coverage"][symbol] = coverage
    
    # Quality checks
    if coverage < 0.9:
        report["issues"].append(f"Low coverage for {symbol}: {coverage:.2%}")
    
    # Check for stale data
    last_update = symbol_data.index[-1]
    days_since_update = (datetime.now() - last_update).days
    
    if days_since_update > 3:
        report["issues"].append(f"Stale data for {symbol}: {days_since_update} days")

# Summary metrics
report["quality_metrics"] = {
    "avg_coverage": sum(report["data_coverage"].values()) / len(report["data_coverage"]),
    "symbols_with_issues": len([s for s, c in report["data_coverage"].items() if c < 0.9]),
    "total_issues": len(report["issues"])
}

# Save report
import os
os.makedirs(os.path.dirname("$REPORT_FILE"), exist_ok=True)
with open("$REPORT_FILE", "w") as f:
    json.dump(report, f, indent=2)

print(f"Data quality report saved to $REPORT_FILE")
print(f"Issues found: {report['quality_metrics']['total_issues']}")
END_PYTHON
EOF

chmod +x scripts/monitor_data_quality.sh
```

## Monthly Operations

### Model Retraining Workflow

#### Complete Model Retraining Process
```python
# src/operations/monthly_retraining.py
"""Monthly model retraining and validation workflow."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from src.config.models import ModelConfig
from src.evaluation.backtest.engine import BacktestEngine
from src.models.model_registry import ModelRegistry
from src.utils.notifications import send_alert

logger = logging.getLogger(__name__)

class MonthlyRetrainingWorkflow:
    """Manages monthly model retraining process."""
    
    def __init__(self, config_path: str = "configs/production/monthly_retraining.yaml"):
        """Initialize retraining workflow."""
        self.config = self._load_config(config_path)
        self.model_registry = ModelRegistry()
        self.backtest_engine = BacktestEngine()
    
    def execute_monthly_retraining(self) -> bool:
        """Execute complete monthly retraining workflow."""
        try:
            logger.info("Starting monthly retraining workflow")
            
            # 1. Create backup of current models
            backup_result = self._backup_current_models()
            if not backup_result:
                return False
            
            # 2. Retrain all models
            retraining_results = self._retrain_all_models()
            
            # 3. Validate new models
            validation_results = self._validate_new_models()
            
            # 4. Performance comparison
            comparison_results = self._compare_model_performance()
            
            # 5. Deployment decision
            deployment_decision = self._make_deployment_decision(
                validation_results, 
                comparison_results
            )
            
            # 6. Deploy or rollback
            if deployment_decision["deploy"]:
                deployment_result = self._deploy_new_models()
            else:
                deployment_result = self._rollback_to_previous_models()
                
            # 7. Generate monthly report
            self._generate_monthly_report(
                retraining_results,
                validation_results, 
                comparison_results,
                deployment_decision
            )
            
            logger.info("Monthly retraining workflow completed")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Monthly retraining failed: {str(e)}")
            send_alert(f"Monthly retraining error: {str(e)}", level="critical")
            return False
    
    def _retrain_all_models(self) -> Dict[str, bool]:
        """Retrain all production models."""
        results = {}
        
        for model_type in ["hrp", "lstm", "gat"]:
            try:
                logger.info(f"Retraining {model_type} model")
                
                # Load model configuration
                config = self._get_model_config(model_type)
                
                # Create model instance
                model = self.model_registry.create_model(model_type, config)
                
                # Train model
                training_result = model.train(
                    data_loader=self._get_training_data(),
                    validation_split=0.2
                )
                
                if training_result.success:
                    # Save new model
                    model_path = f"models/monthly_{datetime.now().strftime('%Y%m')}/{model_type}_model.pkl"
                    model.save(model_path)
                    results[model_type] = True
                    logger.info(f"{model_type} model training completed successfully")
                else:
                    results[model_type] = False
                    logger.error(f"{model_type} model training failed: {training_result.error}")
                    
            except Exception as e:
                logger.error(f"Error training {model_type} model: {str(e)}")
                results[model_type] = False
        
        return results
    
    def _validate_new_models(self) -> Dict[str, Dict]:
        """Validate newly trained models."""
        validation_results = {}
        
        for model_type in ["hrp", "lstm", "gat"]:
            try:
                model_path = f"models/monthly_{datetime.now().strftime('%Y%m')}/{model_type}_model.pkl"
                
                if not Path(model_path).exists():
                    validation_results[model_type] = {"valid": False, "reason": "Model file not found"}
                    continue
                
                # Load model
                model = self.model_registry.load_model(model_path)
                
                # Run validation tests
                validation_result = self._run_model_validation_tests(model)
                validation_results[model_type] = validation_result
                
            except Exception as e:
                validation_results[model_type] = {
                    "valid": False, 
                    "reason": f"Validation error: {str(e)}"
                }
        
        return validation_results
```

#### Monthly Portfolio Rebalancing
```bash
# Create monthly rebalancing script
cat > scripts/monthly_rebalancing.sh << EOF
#!/bin/bash

REBALANCE_DATE=\$(date +%Y-%m-%d)
REBALANCE_DIR="/opt/gnn-portfolio/rebalancing/\$REBALANCE_DATE"

echo "Starting monthly rebalancing for \$REBALANCE_DATE"

# Create rebalancing directory
mkdir -p "\$REBALANCE_DIR"

# Generate new portfolio weights
python << END_PYTHON
import pandas as pd
from datetime import datetime
from src.models.model_registry import ModelRegistry
from src.operations.portfolio_generation import ProductionPortfolioGenerator

# Initialize components
registry = ModelRegistry()
generator = ProductionPortfolioGenerator()

# Load production models
models = {
    'hrp': registry.load_production_model('hrp'),
    'lstm': registry.load_production_model('lstm'),
    'gat': registry.load_production_model('gat')
}

# Generate portfolio weights
portfolio_weights = generator.generate_ensemble_portfolio(
    models=models,
    date=datetime.now(),
    ensemble_method='weighted_average',
    model_weights={'hrp': 0.4, 'lstm': 0.3, 'gat': 0.3}
)

# Save portfolio weights
portfolio_weights.to_csv("$REBALANCE_DIR/new_portfolio_weights.csv")

# Generate trade list
current_portfolio = pd.read_csv("/opt/gnn-portfolio/current_portfolio.csv", index_col=0)
trades = generator.generate_trade_list(
    current_weights=current_portfolio['weight'],
    target_weights=portfolio_weights['weight']
)

trades.to_csv("$REBALANCE_DIR/trade_list.csv")

# Risk analysis
risk_report = generator.generate_risk_report(portfolio_weights)
risk_report.to_json("$REBALANCE_DIR/risk_analysis.json")

print(f"Rebalancing files generated in $REBALANCE_DIR")
print(f"Number of trades required: {len(trades)}")
print(f"Estimated turnover: {trades['trade_value'].abs().sum() / portfolio_weights['value'].sum():.2%}")
END_PYTHON

# Generate rebalancing report
python src/operations/generate_rebalancing_report.py \
    --current-portfolio /opt/gnn-portfolio/current_portfolio.csv \
    --new-portfolio "\$REBALANCE_DIR/new_portfolio_weights.csv" \
    --output "\$REBALANCE_DIR/rebalancing_report.pdf"

echo "Monthly rebalancing completed. Review files in \$REBALANCE_DIR"
EOF

chmod +x scripts/monthly_rebalancing.sh
```

### Performance Review Process

#### Monthly Performance Analysis
```python
# src/operations/monthly_performance_review.py
"""Monthly performance review and analysis."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

from src.evaluation.metrics.performance import PerformanceAnalytics
from src.evaluation.reporting.comprehensive_report import ComprehensiveReportGenerator
from src.utils.notifications import send_report

logger = logging.getLogger(__name__)

class MonthlyPerformanceReview:
    """Manages monthly performance review process."""
    
    def __init__(self):
        """Initialize performance review system."""
        self.analytics = PerformanceAnalytics()
        self.report_generator = ComprehensiveReportGenerator()
    
    def generate_monthly_review(self, review_date: datetime = None) -> Dict:
        """Generate comprehensive monthly performance review."""
        if review_date is None:
            review_date = datetime.now()
        
        # Define review period
        end_date = review_date.replace(day=1) - timedelta(days=1)  # Last day of previous month
        start_date = end_date.replace(day=1)  # First day of previous month
        
        logger.info(f"Generating monthly review for {start_date.strftime('%Y-%m')}")
        
        review_data = {
            "period": {
                "start_date": start_date,
                "end_date": end_date,
                "month": start_date.strftime("%B %Y")
            },
            "portfolio_performance": self._analyze_portfolio_performance(start_date, end_date),
            "model_performance": self._analyze_model_performance(start_date, end_date),
            "risk_analysis": self._analyze_risk_metrics(start_date, end_date),
            "attribution_analysis": self._analyze_performance_attribution(start_date, end_date),
            "market_comparison": self._compare_to_benchmarks(start_date, end_date),
            "operational_metrics": self._analyze_operational_metrics(start_date, end_date),
            "recommendations": self._generate_recommendations()
        }
        
        # Generate formatted reports
        self._generate_formatted_reports(review_data)
        
        return review_data
    
    def _analyze_portfolio_performance(self, start_date: datetime, end_date: datetime) -> Dict:
        """Analyze portfolio performance for the month."""
        portfolio_data = self._load_portfolio_returns(start_date, end_date)
        
        return {
            "monthly_return": self.analytics.calculate_period_return(portfolio_data),
            "volatility": self.analytics.calculate_volatility(portfolio_data, annualized=True),
            "sharpe_ratio": self.analytics.calculate_sharpe_ratio(portfolio_data),
            "max_drawdown": self.analytics.calculate_max_drawdown(portfolio_data),
            "var_95": self.analytics.calculate_var(portfolio_data, confidence_level=0.95),
            "total_trades": self._count_trades(start_date, end_date),
            "turnover": self._calculate_turnover(start_date, end_date),
            "transaction_costs": self._calculate_transaction_costs(start_date, end_date)
        }
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate operational and strategic recommendations."""
        recommendations = []
        
        # Model performance recommendations
        model_metrics = self._get_recent_model_metrics()
        
        for model_type, metrics in model_metrics.items():
            if metrics["sharpe_ratio"] < 1.0:
                recommendations.append({
                    "type": "model_performance",
                    "priority": "medium",
                    "description": f"{model_type} model underperforming (Sharpe: {metrics['sharpe_ratio']:.2f})",
                    "action": f"Consider retraining {model_type} model with updated parameters",
                    "timeline": "next_month"
                })
        
        # Risk management recommendations
        portfolio_metrics = self._get_current_portfolio_metrics()
        
        if portfolio_metrics["max_drawdown"] > 0.15:
            recommendations.append({
                "type": "risk_management",
                "priority": "high",
                "description": f"High drawdown detected: {portfolio_metrics['max_drawdown']:.2%}",
                "action": "Review position sizing and risk constraints",
                "timeline": "immediate"
            })
        
        return recommendations
```

## System Maintenance

### Weekly Maintenance Tasks

#### Database Maintenance
```bash
# Weekly database maintenance script
cat > scripts/weekly_maintenance.sh << EOF
#!/bin/bash

echo "=== Weekly Maintenance $(date) ==="

# 1. Database vacuum and analyze
echo "1. Database maintenance..."
sudo -u postgres psql gnn_portfolio << SQL
VACUUM ANALYZE;
REINDEX DATABASE gnn_portfolio;
SQL

# 2. Log rotation
echo "2. Log rotation..."
find /opt/gnn-portfolio/logs -name "*.log" -mtime +30 -delete
find /opt/gnn-portfolio/logs -name "*.log.*" -mtime +7 -delete

# 3. Cache cleanup
echo "3. Cache cleanup..."
find /opt/gnn-portfolio/cache -mtime +7 -delete

# 4. Model checkpoint cleanup
echo "4. Model checkpoint cleanup..."
find /opt/gnn-portfolio/models -name "checkpoint_*.pkl" -mtime +14 -delete

# 5. System updates (if approved)
echo "5. Security updates check..."
apt list --upgradable | grep -i security

# 6. GPU driver check
echo "6. GPU driver status..."
nvidia-smi -q -d temperature,power,clock

# 7. Performance metrics collection
echo "7. Collecting performance metrics..."
python src/utils/collect_weekly_metrics.py --output /opt/gnn-portfolio/reports/weekly_metrics.json

echo "=== Weekly Maintenance Complete ==="
EOF

chmod +x scripts/weekly_maintenance.sh

# Schedule weekly maintenance
(crontab -l 2>/dev/null; echo "0 2 * * 0 /opt/gnn-portfolio/gnn-portfolio-system/scripts/weekly_maintenance.sh") | crontab -
```

### Emergency Procedures

#### Service Recovery Procedures
```bash
# Emergency service recovery script
cat > scripts/emergency_recovery.sh << EOF
#!/bin/bash

SERVICE_NAME="gnn-portfolio"
RECOVERY_LOG="/opt/gnn-portfolio/logs/recovery_\$(date +%Y%m%d_%H%M%S).log"

echo "=== EMERGENCY RECOVERY PROCEDURE ===" | tee \$RECOVERY_LOG

# 1. Check service status
echo "1. Checking service status..." | tee -a \$RECOVERY_LOG
if systemctl is-active --quiet \$SERVICE_NAME; then
    echo "Service is running - no recovery needed" | tee -a \$RECOVERY_LOG
    exit 0
fi

echo "Service is down - initiating recovery" | tee -a \$RECOVERY_LOG

# 2. Check for obvious issues
echo "2. Diagnosing issues..." | tee -a \$RECOVERY_LOG

# Check disk space
DISK_USAGE=\$(df /opt/gnn-portfolio | tail -1 | awk '{print \$5}' | sed 's/%//')
if [[ \$DISK_USAGE -gt 90 ]]; then
    echo "CRITICAL: Disk space full (\$DISK_USAGE%)" | tee -a \$RECOVERY_LOG
    # Emergency cleanup
    find /opt/gnn-portfolio/cache -type f -delete
    find /opt/gnn-portfolio/logs -name "*.log.*" -delete
fi

# Check GPU status
if ! nvidia-smi &>/dev/null; then
    echo "CRITICAL: GPU not accessible" | tee -a \$RECOVERY_LOG
    # Try to reset GPU
    sudo nvidia-smi -r || echo "GPU reset failed" | tee -a \$RECOVERY_LOG
fi

# Check database
if ! sudo -u postgres psql gnn_portfolio -c "SELECT 1;" &>/dev/null; then
    echo "CRITICAL: Database not accessible" | tee -a \$RECOVERY_LOG
    sudo systemctl restart postgresql
    sleep 10
fi

# 3. Attempt service restart
echo "3. Attempting service restart..." | tee -a \$RECOVERY_LOG
sudo systemctl restart \$SERVICE_NAME
sleep 30

# 4. Verify recovery
if systemctl is-active --quiet \$SERVICE_NAME; then
    echo "SUCCESS: Service recovered" | tee -a \$RECOVERY_LOG
    
    # Test API endpoint
    if curl -f -s http://localhost:8000/health >/dev/null; then
        echo "SUCCESS: API is responding" | tee -a \$RECOVERY_LOG
    else
        echo "WARNING: API not responding" | tee -a \$RECOVERY_LOG
    fi
else
    echo "FAILURE: Service recovery failed" | tee -a \$RECOVERY_LOG
    echo "Manual intervention required" | tee -a \$RECOVERY_LOG
    
    # Send critical alert
    curl -X POST "\$SLACK_WEBHOOK" -d '{"text":"🔥 CRITICAL: GNN Portfolio service recovery FAILED - manual intervention required"}'
fi

echo "Recovery log saved to \$RECOVERY_LOG"
EOF

chmod +x scripts/emergency_recovery.sh
```

## Monitoring and Alerting

### Alert Configuration

#### Comprehensive Monitoring Setup
```yaml
# configs/monitoring/alerts.yaml
alerts:
  system:
    - name: "service_down"
      condition: "service_status != 'active'"
      severity: "critical"
      notification: ["email", "slack", "pager"]
      cooldown_minutes: 5
    
    - name: "high_gpu_temperature"
      condition: "gpu_temperature > 85"
      severity: "warning"
      notification: ["email", "slack"]
      cooldown_minutes: 15
    
    - name: "low_disk_space"
      condition: "disk_usage_percent > 85"
      severity: "warning"
      notification: ["email"]
      cooldown_minutes: 60
  
  performance:
    - name: "high_daily_drawdown"
      condition: "daily_drawdown < -0.05"
      severity: "high"
      notification: ["email", "slack"]
      cooldown_minutes: 0
    
    - name: "low_sharpe_ratio"
      condition: "rolling_30d_sharpe < 0.5"
      severity: "medium"
      notification: ["email"]
      cooldown_minutes: 1440  # Daily
  
  data:
    - name: "stale_data"
      condition: "last_data_update_hours > 25"
      severity: "high"
      notification: ["email", "slack"]
      cooldown_minutes: 60
    
    - name: "data_quality_issues"
      condition: "data_quality_score < 0.9"
      severity: "medium"
      notification: ["email"]
      cooldown_minutes: 240
```

### Custom Monitoring Scripts

#### Performance Monitoring Dashboard
```python
# src/monitoring/performance_dashboard.py
"""Real-time performance monitoring dashboard."""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List

import pandas as pd
from dataclasses import dataclass

from src.evaluation.metrics.performance import PerformanceAnalytics
from src.utils.gpu_monitor import GPUMonitor
from src.utils.notifications import send_alert

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: datetime
    gpu_memory_used: float
    gpu_temperature: int
    cpu_usage: float
    disk_usage: float
    active_connections: int

@dataclass
class PortfolioMetrics:
    """Portfolio performance metrics."""
    timestamp: datetime
    current_value: float
    daily_pnl: float
    daily_pnl_pct: float
    mtd_return: float
    ytd_return: float
    sharpe_ratio: float
    max_drawdown: float

class RealTimeMonitoringDashboard:
    """Real-time monitoring and alerting system."""
    
    def __init__(self, config_path: str = "configs/monitoring/dashboard_config.yaml"):
        """Initialize monitoring dashboard."""
        self.config = self._load_config(config_path)
        self.analytics = PerformanceAnalytics()
        self.gpu_monitor = GPUMonitor()
        self.alert_thresholds = self._load_alert_thresholds()
        self.last_alert_times = {}
    
    def start_monitoring(self, update_interval: int = 60):
        """Start real-time monitoring loop."""
        logger.info("Starting real-time monitoring dashboard")
        
        while True:
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                
                # Collect portfolio metrics
                portfolio_metrics = self._collect_portfolio_metrics()
                
                # Check alert conditions
                self._check_alert_conditions(system_metrics, portfolio_metrics)
                
                # Update dashboard
                self._update_dashboard(system_metrics, portfolio_metrics)
                
                # Log metrics
                self._log_metrics(system_metrics, portfolio_metrics)
                
            except Exception as e:
                logger.error(f"Monitoring error: {str(e)}")
                send_alert(f"Monitoring system error: {str(e)}", level="warning")
            
            time.sleep(update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        return SystemMetrics(
            timestamp=datetime.now(),
            gpu_memory_used=gpu_info["memory_used_gb"],
            gpu_temperature=gpu_info["temperature"],
            cpu_usage=self._get_cpu_usage(),
            disk_usage=self._get_disk_usage(),
            active_connections=self._get_active_connections()
        )
    
    def _check_alert_conditions(
        self, 
        system_metrics: SystemMetrics, 
        portfolio_metrics: PortfolioMetrics
    ):
        """Check for alert conditions and send notifications."""
        
        # System alerts
        if system_metrics.gpu_temperature > self.alert_thresholds["gpu_temp_warning"]:
            self._send_alert_if_cooldown_expired(
                "gpu_temperature_high",
                f"GPU temperature high: {system_metrics.gpu_temperature}°C",
                level="warning"
            )
        
        if system_metrics.disk_usage > self.alert_thresholds["disk_usage_warning"]:
            self._send_alert_if_cooldown_expired(
                "disk_usage_high",
                f"Disk usage high: {system_metrics.disk_usage:.1f}%",
                level="warning"
            )
        
        # Portfolio alerts
        if portfolio_metrics.daily_pnl_pct < self.alert_thresholds["daily_loss_threshold"]:
            self._send_alert_if_cooldown_expired(
                "daily_loss_threshold",
                f"Daily loss threshold exceeded: {portfolio_metrics.daily_pnl_pct:.2%}",
                level="high"
            )
        
        if portfolio_metrics.max_drawdown < self.alert_thresholds["max_drawdown_threshold"]:
            self._send_alert_if_cooldown_expired(
                "max_drawdown_threshold", 
                f"Maximum drawdown threshold exceeded: {portfolio_metrics.max_drawdown:.2%}",
                level="critical"
            )
```

This comprehensive operational procedures document provides institutional-grade operational workflows with detailed automation, monitoring, and emergency response capabilities for production environments.