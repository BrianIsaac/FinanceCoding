# REST API Specification

## API Overview

The GNN Portfolio Optimization System provides a comprehensive REST API for model training, portfolio generation, performance monitoring, and system management. The API follows RESTful principles and returns JSON responses for programmatic integration with existing portfolio management systems.

## Base Configuration

### API Base URL
```
Production: https://api.gnn-portfolio.com/v1
Development: http://localhost:8000/v1
```

### Authentication
```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
Accept: application/json
```

### Response Format
All API responses follow this standard format:

```json
{
  "success": true,
  "data": {
    // Response payload
  },
  "message": "Operation completed successfully",
  "timestamp": "2025-09-09T12:00:00Z",
  "request_id": "req_12345"
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter: lookback_days must be positive",
    "details": {
      "parameter": "lookback_days", 
      "value": -10
    }
  },
  "timestamp": "2025-09-09T12:00:00Z",
  "request_id": "req_12345"
}
```

## Model Training API

### HRP Model Training

#### Train HRP Model
```http
POST /models/hrp/train
```

**Request Body:**
```json
{
  "config": {
    "lookback_days": 756,
    "clustering_config": {
      "linkage_method": "ward",
      "distance_metric": "correlation",
      "min_observations": 252,
      "correlation_method": "pearson"
    },
    "allocation_config": {
      "risk_measure": "variance",
      "rebalance_threshold": 0.05
    }
  },
  "data_config": {
    "universe": "sp_midcap_400",
    "start_date": "2016-01-01",
    "end_date": "2025-09-09",
    "features": ["returns", "volatility", "volume"]
  },
  "experiment_id": "hrp_experiment_001"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "hrp_model_20250909_120000",
    "training_status": "completed",
    "training_metrics": {
      "training_time_seconds": 120,
      "memory_usage_gb": 2.4,
      "validation_sharpe": 1.42,
      "validation_sortino": 1.89
    },
    "model_artifacts": {
      "checkpoint_path": "/models/hrp_model_20250909_120000.pkl",
      "config_path": "/configs/hrp_model_20250909_120000.yaml",
      "training_log": "/logs/hrp_training_20250909_120000.log"
    }
  }
}
```

### LSTM Model Training

#### Train LSTM Model
```http
POST /models/lstm/train
```

**Request Body:**
```json
{
  "config": {
    "sequence_length": 60,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "early_stopping_patience": 10
  },
  "data_config": {
    "universe": "sp_midcap_400",
    "start_date": "2016-01-01", 
    "end_date": "2025-09-09",
    "features": ["returns", "volatility", "volume", "market_cap"]
  },
  "training_config": {
    "use_gpu": true,
    "mixed_precision": true,
    "gradient_clipping": 1.0
  },
  "experiment_id": "lstm_experiment_001"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "lstm_model_20250909_120000",
    "training_status": "in_progress",
    "estimated_completion": "2025-09-09T16:00:00Z",
    "training_progress": {
      "epoch": 25,
      "total_epochs": 100,
      "current_loss": 0.0342,
      "best_validation_loss": 0.0289,
      "gpu_memory_usage": "8.2 GB / 11.0 GB"
    }
  }
}
```

### GAT Model Training

#### Train GAT Model
```http
POST /models/gat/train
```

**Request Body:**
```json
{
  "config": {
    "attention_heads": 4,
    "hidden_dim": 128,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "graph_construction": "k_nn",
    "k_neighbors": 10,
    "edge_threshold": 0.3
  },
  "data_config": {
    "universe": "sp_midcap_400",
    "start_date": "2016-01-01",
    "end_date": "2025-09-09",
    "features": ["returns", "volatility", "sector", "market_cap"]
  },
  "graph_config": {
    "method": "k_nn",
    "k_neighbors": 10,
    "edge_features": ["correlation", "sector_similarity"],
    "dynamic_graphs": true
  },
  "experiment_id": "gat_experiment_001"
}
```

## Portfolio Generation API

### Generate Portfolio Weights

#### Single Model Portfolio
```http
POST /portfolios/generate
```

**Request Body:**
```json
{
  "model_id": "hrp_model_20250909_120000",
  "date": "2025-09-09",
  "constraints": {
    "long_only": true,
    "top_k_positions": 50,
    "max_position_weight": 0.10,
    "max_monthly_turnover": 0.20,
    "transaction_cost_bps": 10.0,
    "sector_constraints": {
      "technology": {"min": 0.05, "max": 0.25},
      "healthcare": {"min": 0.05, "max": 0.20}
    }
  },
  "universe_config": {
    "universe": "sp_midcap_400",
    "exclude_symbols": ["AAPL", "MSFT"],
    "min_market_cap": 1000000000
  }
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "portfolio_20250909_120000",
    "generation_date": "2025-09-09",
    "model_id": "hrp_model_20250909_120000",
    "weights": {
      "NVDA": 0.08,
      "AMD": 0.06,
      "QCOM": 0.05,
      // ... additional holdings
    },
    "portfolio_metrics": {
      "total_positions": 47,
      "concentration": 0.32,
      "sector_allocation": {
        "technology": 0.22,
        "healthcare": 0.18,
        "financial": 0.15
      },
      "estimated_turnover": 0.18,
      "transaction_costs_bps": 8.5
    },
    "constraint_validation": {
      "all_constraints_satisfied": true,
      "violations": [],
      "warnings": [
        "Position count (47) below target (50) due to liquidity constraints"
      ]
    }
  }
}
```

#### Ensemble Portfolio (Multiple Models)
```http
POST /portfolios/ensemble
```

**Request Body:**
```json
{
  "models": [
    {
      "model_id": "hrp_model_20250909_120000",
      "weight": 0.4
    },
    {
      "model_id": "lstm_model_20250909_120000", 
      "weight": 0.3
    },
    {
      "model_id": "gat_model_20250909_120000",
      "weight": 0.3
    }
  ],
  "ensemble_method": "weighted_average",
  "date": "2025-09-09",
  "constraints": {
    "long_only": true,
    "top_k_positions": 50,
    "max_position_weight": 0.10
  }
}
```

## Performance Monitoring API

### Real-time Performance Tracking

#### Get Portfolio Performance
```http
GET /portfolios/{portfolio_id}/performance
```

**Query Parameters:**
- `start_date`: Start date for performance calculation (ISO format)
- `end_date`: End date for performance calculation (ISO format)  
- `benchmark`: Benchmark for comparison (default: "equal_weight")
- `metrics`: Comma-separated list of metrics to include

**Response:**
```json
{
  "success": true,
  "data": {
    "portfolio_id": "portfolio_20250909_120000",
    "performance_period": {
      "start_date": "2025-08-01",
      "end_date": "2025-09-09"
    },
    "returns": {
      "total_return": 0.0842,
      "annualized_return": 0.2156,
      "benchmark_return": 0.0623,
      "excess_return": 0.0219
    },
    "risk_metrics": {
      "sharpe_ratio": 1.45,
      "sortino_ratio": 1.92,
      "maximum_drawdown": -0.0432,
      "volatility": 0.1487,
      "var_95": -0.0234,
      "cvar_95": -0.0356
    },
    "attribution": {
      "sector_contribution": {
        "technology": 0.0156,
        "healthcare": 0.0087,
        "financial": -0.0023
      },
      "stock_contribution_top_5": {
        "NVDA": 0.0067,
        "AMD": 0.0043,
        "QCOM": 0.0039,
        "AVGO": 0.0034,
        "GOOGL": 0.0031
      }
    }
  }
}
```

#### Live Performance Dashboard
```http
GET /performance/dashboard
```

**Response:**
```json
{
  "success": true,
  "data": {
    "timestamp": "2025-09-09T12:00:00Z",
    "active_portfolios": [
      {
        "portfolio_id": "portfolio_20250909_120000",
        "model_type": "ensemble",
        "current_value": 1084200.56,
        "daily_pnl": 8420.34,
        "daily_pnl_pct": 0.0082,
        "mtd_return": 0.0234,
        "ytd_return": 0.1567,
        "sharpe_ratio": 1.45,
        "max_drawdown": -0.0432
      }
    ],
    "system_health": {
      "gpu_memory_usage": "7.2 GB / 11.0 GB",
      "cpu_usage": 45,
      "disk_usage": "234 GB / 500 GB",
      "data_pipeline_status": "healthy",
      "last_data_update": "2025-09-09T11:30:00Z"
    },
    "alerts": [
      {
        "level": "warning",
        "message": "Daily drawdown exceeded -2% threshold",
        "portfolio_id": "portfolio_20250909_120000",
        "timestamp": "2025-09-09T10:15:00Z"
      }
    ]
  }
}
```

## Model Management API

### Model Information and Status

#### List Available Models
```http
GET /models
```

**Query Parameters:**
- `model_type`: Filter by model type (hrp, lstm, gat)
- `status`: Filter by status (training, completed, deployed)
- `limit`: Maximum number of results (default: 50)

**Response:**
```json
{
  "success": true,
  "data": {
    "models": [
      {
        "model_id": "hrp_model_20250909_120000",
        "model_type": "hrp",
        "status": "deployed",
        "created_date": "2025-09-09T12:00:00Z",
        "training_metrics": {
          "validation_sharpe": 1.42,
          "training_time_hours": 0.03,
          "memory_usage_gb": 2.4
        },
        "deployment_metrics": {
          "portfolios_generated": 156,
          "avg_generation_time_ms": 234,
          "success_rate": 0.998
        }
      }
    ],
    "total_count": 12,
    "page": 1,
    "limit": 50
  }
}
```

#### Get Model Details
```http
GET /models/{model_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "model_id": "hrp_model_20250909_120000",
    "model_type": "hrp", 
    "status": "deployed",
    "config": {
      "lookback_days": 756,
      "clustering_config": {
        "linkage_method": "ward",
        "distance_metric": "correlation"
      }
    },
    "training_history": {
      "start_time": "2025-09-09T12:00:00Z",
      "end_time": "2025-09-09T12:02:00Z",
      "validation_metrics": {
        "sharpe_ratio": 1.42,
        "max_drawdown": -0.0856,
        "annual_return": 0.1834
      }
    },
    "deployment_info": {
      "deployment_date": "2025-09-09T12:05:00Z",
      "version": "1.0.0",
      "resource_usage": {
        "memory_mb": 2400,
        "cpu_cores": 2
      }
    }
  }
}
```

## Backtesting and Analysis API

### Rolling Backtest Execution

#### Run Backtest
```http
POST /backtest/run
```

**Request Body:**
```json
{
  "models": ["hrp_model_20250909_120000", "lstm_model_20250909_120000"],
  "backtest_config": {
    "start_date": "2016-01-01",
    "end_date": "2024-12-31",
    "rebalance_frequency": "monthly",
    "initial_capital": 1000000,
    "universe": "sp_midcap_400"
  },
  "analysis_config": {
    "benchmarks": ["equal_weight", "market_cap_weight"],
    "metrics": ["sharpe_ratio", "max_drawdown", "annual_return"],
    "statistical_tests": ["jobson_korkie", "bootstrap_confidence"]
  },
  "experiment_id": "backtest_comparison_001"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "backtest_id": "backtest_20250909_120000",
    "status": "running",
    "estimated_completion": "2025-09-09T18:00:00Z",
    "progress": {
      "current_date": "2020-06-15",
      "completion_percent": 45.2,
      "models_completed": 0,
      "models_total": 2
    }
  }
}
```

#### Get Backtest Results
```http
GET /backtest/{backtest_id}/results
```

**Response:**
```json
{
  "success": true,
  "data": {
    "backtest_id": "backtest_20250909_120000",
    "status": "completed",
    "results": {
      "hrp_model_20250909_120000": {
        "annual_return": 0.1834,
        "sharpe_ratio": 1.42,
        "max_drawdown": -0.0856,
        "volatility": 0.1291,
        "statistical_significance": {
          "vs_equal_weight": {
            "p_value": 0.0023,
            "confidence_interval": [0.021, 0.087]
          }
        }
      },
      "benchmarks": {
        "equal_weight": {
          "annual_return": 0.1456,
          "sharpe_ratio": 1.08,
          "max_drawdown": -0.1243
        }
      }
    }
  }
}
```

## System Management API

### Health and Status Monitoring

#### System Health Check
```http
GET /health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "status": "healthy",
    "timestamp": "2025-09-09T12:00:00Z",
    "version": "1.0.0",
    "components": {
      "database": "healthy",
      "gpu": {
        "status": "healthy", 
        "memory_usage": "7.2 GB / 11.0 GB",
        "temperature": "72°C"
      },
      "data_pipeline": {
        "status": "healthy",
        "last_update": "2025-09-09T11:30:00Z",
        "symbols_updated": 400
      },
      "model_serving": "healthy"
    },
    "uptime_seconds": 86400
  }
}
```

### Configuration Management

#### Update System Configuration
```http
PUT /config/system
```

**Request Body:**
```json
{
  "gpu_memory_limit_gb": 10.5,
  "batch_size_limit": 128,
  "alert_thresholds": {
    "max_daily_loss": -0.05,
    "max_drawdown": -0.25,
    "min_sharpe_ratio": 0.5
  },
  "data_refresh_schedule": "0 6 * * 1-5"  // 6 AM weekdays
}
```

## Error Codes Reference

### Model Training Errors
- `MODEL_TRAINING_FAILED`: Training process failed
- `INSUFFICIENT_GPU_MEMORY`: Not enough GPU memory for training
- `INVALID_MODEL_CONFIG`: Model configuration validation failed
- `DATA_INSUFFICIENT`: Insufficient data for training period

### Portfolio Generation Errors  
- `CONSTRAINT_VIOLATION`: Portfolio constraints cannot be satisfied
- `MODEL_NOT_FOUND`: Specified model ID does not exist
- `UNIVERSE_INVALID`: Invalid universe configuration
- `LIQUIDITY_INSUFFICIENT`: Insufficient liquidity for required positions

### System Errors
- `GPU_MEMORY_EXCEEDED`: GPU memory limit exceeded
- `DATA_PIPELINE_ERROR`: Error in data pipeline processing
- `AUTHENTICATION_FAILED`: Invalid or expired authentication token
- `RATE_LIMIT_EXCEEDED`: API rate limit exceeded

## Rate Limiting

### API Rate Limits
- **Training endpoints**: 10 requests per hour per API key
- **Portfolio generation**: 100 requests per hour per API key  
- **Performance monitoring**: 1000 requests per hour per API key
- **System health**: No rate limiting

### Headers
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1694268000
```

## Webhooks

### Model Training Completion
```json
{
  "event": "model.training.completed",
  "data": {
    "model_id": "hrp_model_20250909_120000", 
    "status": "completed",
    "metrics": {
      "validation_sharpe": 1.42
    }
  },
  "timestamp": "2025-09-09T12:00:00Z"
}
```

### Portfolio Alert
```json
{
  "event": "portfolio.alert",
  "data": {
    "portfolio_id": "portfolio_20250909_120000",
    "alert_type": "drawdown_exceeded",
    "threshold": -0.02,
    "current_value": -0.0234
  },
  "timestamp": "2025-09-09T10:15:00Z"
}
```

This REST API specification provides comprehensive coverage for all aspects of the GNN Portfolio Optimization System, enabling full programmatic control and integration with existing institutional systems.