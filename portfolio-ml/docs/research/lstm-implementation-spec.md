# LSTM Implementation Specification
## Portfolio Optimization with Machine Learning Techniques

**Version:** 1.0  
**Date:** September 5, 2025  
**Dependencies:** Existing data pipeline, GAT framework, unified constraint system

---

## Overview

The LSTM (Long Short-Term Memory) module implements temporal pattern recognition for portfolio allocation using sequence-to-sequence networks with 60-day lookback windows. This module leverages the existing S&P MidCap 400 dynamic universe and parquet-based data pipeline to capture temporal dependencies that traditional approaches miss.

## Data Integration

### Input Data Sources
- **returns_daily.parquet**: Daily returns matrix from existing data pipeline
- **universe_calendar.parquet**: Dynamic S&P MidCap 400 membership 
- **prices.parquet**: Price levels for additional feature engineering
- **volume.parquet**: Volume data for liquidity filtering

### Feature Engineering Pipeline
```python
@dataclass
class LSTMDataConfig:
    sequence_length: int = 60                    # 60-day lookback window
    forecast_horizon: int = 21                   # 21-day forward prediction
    feature_set: List[str] = field(default_factory=lambda: [
        'returns',                               # Daily returns (primary)
        'volatility_20d',                       # 20-day rolling volatility
        'momentum_20d',                         # 20-day momentum
        'rsi_14d',                              # 14-day RSI
        'volume_ratio_20d'                      # Volume ratio vs 20-day average
    ])
    min_history_days: int = 100                 # Minimum history for inclusion
    standardize_features: bool = True           # Z-score normalization
    handle_missing: str = 'forward_fill'       # Missing data strategy
```

### Sequence Generation
```python
class LSTMSequenceGenerator:
    def __init__(self, config: LSTMDataConfig):
        self.config = config
        
    def create_sequences(self, 
                        returns_df: pd.DataFrame,
                        universe_calendar: pd.DataFrame,
                        start_date: pd.Timestamp,
                        end_date: pd.Timestamp) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Generate LSTM training sequences from historical data.
        
        Returns:
            features: [N_samples, sequence_length, n_features] 
            targets: [N_samples, n_assets] - 21-day forward returns
            asset_names: List of asset tickers for each sample
        """
        
    def _engineer_features(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators from returns data."""
        features_df = pd.DataFrame(index=returns_df.index)
        
        for asset in returns_df.columns:
            asset_returns = returns_df[asset]
            
            # Core features
            features_df[f'{asset}_returns'] = asset_returns
            features_df[f'{asset}_vol_20d'] = asset_returns.rolling(20).std()
            features_df[f'{asset}_momentum_20d'] = asset_returns.rolling(20).mean()
            
            # Technical indicators
            features_df[f'{asset}_rsi_14d'] = self._calculate_rsi(asset_returns, 14)
            
            # Volume features (if available)
            if hasattr(self, 'volume_df'):
                vol_ratio = self.volume_df[asset] / self.volume_df[asset].rolling(20).mean()
                features_df[f'{asset}_vol_ratio_20d'] = vol_ratio
        
        return features_df
    
    def _calculate_rsi(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate Relative Strength Index from returns."""
        delta = returns.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

## Model Architecture

### Core LSTM Network
```python
class LSTMPortfolioNetwork(nn.Module):
    def __init__(self, config: LSTMModelConfig):
        super().__init__()
        self.config = config
        
        # Multi-layer LSTM backbone
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional
        )
        
        # Attention mechanism for focusing on relevant time steps
        lstm_output_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            batch_first=True
        )
        
        # Return prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(lstm_output_size, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(), 
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.forecast_horizon)
        )
        
        # Portfolio weight generation layer
        self.portfolio_head = nn.Linear(config.forecast_horizon, 1)
        
    def forward(self, 
                sequences: torch.Tensor,      # [batch, seq_len, features]
                asset_mask: torch.Tensor      # [batch, n_assets] - valid assets
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass generating return forecasts and portfolio weights.
        
        Returns:
            return_forecasts: [batch, n_assets, forecast_horizon]
            portfolio_weights: [batch, n_assets]
        """
        batch_size, n_assets, seq_len, n_features = sequences.shape
        
        # Reshape for LSTM processing: [batch*assets, seq_len, features]
        lstm_input = sequences.reshape(batch_size * n_assets, seq_len, n_features)
        
        # LSTM processing
        lstm_output, (hidden, cell) = self.lstm(lstm_input)
        
        # Attention over time steps
        attended_output, attention_weights = self.attention(
            lstm_output, lstm_output, lstm_output
        )
        
        # Use final time step with attention
        final_representation = attended_output[:, -1, :]  # [batch*assets, hidden_size]
        
        # Generate return forecasts
        return_forecasts = self.prediction_head(final_representation)  # [batch*assets, forecast_horizon]
        return_forecasts = return_forecasts.reshape(batch_size, n_assets, self.config.forecast_horizon)
        
        # Convert forecasts to portfolio weights via expected return ranking
        expected_returns = return_forecasts.mean(dim=2)  # [batch, n_assets]
        
        # Apply softmax with temperature for weight generation
        portfolio_weights = F.softmax(expected_returns / self.config.temperature, dim=1)
        
        # Apply asset mask (zero weights for unavailable assets)
        portfolio_weights = portfolio_weights * asset_mask.float()
        portfolio_weights = portfolio_weights / portfolio_weights.sum(dim=1, keepdim=True)
        
        return return_forecasts, portfolio_weights
```

### Training Configuration
```python
@dataclass 
class LSTMModelConfig:
    # Architecture parameters
    input_features: int = 5                      # Number of features per asset
    hidden_size: int = 128                       # LSTM hidden dimension
    num_layers: int = 3                          # Stacked LSTM layers
    dropout: float = 0.3                         # Dropout rate
    bidirectional: bool = True                   # Bidirectional LSTM
    
    # Attention parameters
    attention_heads: int = 8                     # Multi-head attention
    attention_dropout: float = 0.1               # Attention dropout
    
    # Output parameters
    forecast_horizon: int = 21                   # Days to forecast
    temperature: float = 1.0                     # Softmax temperature for weights
    
    # Training parameters
    learning_rate: float = 0.001                 # Initial learning rate
    weight_decay: float = 1e-5                   # L2 regularization
    batch_size: int = 32                         # Training batch size
    max_epochs: int = 100                        # Maximum training epochs
    patience: int = 15                           # Early stopping patience
    
    # Memory optimization
    gradient_accumulation_steps: int = 4         # For large effective batch size
    mixed_precision: bool = True                 # FP16 training
    gradient_clip_norm: float = 1.0              # Gradient clipping
```

### Loss Functions and Training
```python
class SharpeRatioLoss(nn.Module):
    def __init__(self, risk_free_rate: float = 0.0):
        super().__init__()
        self.risk_free_rate = risk_free_rate
        
    def forward(self, 
                return_forecasts: torch.Tensor,    # [batch, assets, horizon]
                actual_returns: torch.Tensor,      # [batch, assets, horizon]  
                portfolio_weights: torch.Tensor    # [batch, assets]
                ) -> torch.Tensor:
        """
        Compute negative Sharpe ratio as loss (minimize for maximization).
        """
        # Calculate portfolio returns for each forecast horizon
        portfolio_returns = (portfolio_weights.unsqueeze(2) * actual_returns).sum(dim=1)  # [batch, horizon]
        
        # Calculate mean and std of portfolio returns
        mean_return = portfolio_returns.mean(dim=1)  # [batch]
        std_return = portfolio_returns.std(dim=1) + 1e-8  # [batch], add epsilon for stability
        
        # Sharpe ratio (annualized assuming daily returns)
        sharpe_ratio = (mean_return - self.risk_free_rate/252) / std_return * np.sqrt(252)
        
        # Return negative Sharpe (to minimize)
        return -sharpe_ratio.mean()

class LSTMTrainer:
    def __init__(self, 
                 model: LSTMPortfolioNetwork,
                 config: LSTMModelConfig,
                 device: torch.device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize validation Sharpe
            factor=0.5,
            patience=config.patience // 2
        )
        
        # Loss function
        self.criterion = SharpeRatioLoss()
        
        # Mixed precision training
        if config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train model for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, (sequences, targets, asset_masks) in enumerate(train_loader):
            sequences = sequences.to(self.device)
            targets = targets.to(self.device) 
            asset_masks = asset_masks.to(self.device)
            
            if self.config.mixed_precision:
                with torch.cuda.amp.autocast():
                    return_forecasts, portfolio_weights = self.model(sequences, asset_masks)
                    loss = self.criterion(return_forecasts, targets, portfolio_weights)
                    loss = loss / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                return_forecasts, portfolio_weights = self.model(sequences, asset_masks)
                loss = self.criterion(return_forecasts, targets, portfolio_weights)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
        
        return {"train_loss": total_loss / num_batches}
```

## Memory Optimization for 12GB VRAM

### Batch Size Optimization
```python
class LSTMMemoryManager:
    def __init__(self, max_memory_gb: float = 11.0):
        self.max_memory = max_memory_gb * 1024**3
        
    def calculate_optimal_batch_size(self, 
                                   model: LSTMPortfolioNetwork,
                                   sequence_length: int,
                                   n_assets: int,
                                   n_features: int) -> int:
        """
        Calculate optimal batch size for GPU memory constraints.
        """
        # Estimate memory per sample
        input_memory = sequence_length * n_assets * n_features * 4  # float32
        
        # Model parameters
        param_memory = sum(p.numel() * 4 for p in model.parameters())
        
        # LSTM hidden states (conservative estimate)
        hidden_memory = model.config.hidden_size * model.config.num_layers * n_assets * 8
        
        # Attention mechanism memory
        attention_memory = sequence_length * sequence_length * model.config.attention_heads * 4
        
        # Total memory per sample
        memory_per_sample = input_memory + hidden_memory + attention_memory
        
        # Available memory (70% of total for safety)
        available_memory = self.max_memory * 0.7 - param_memory
        
        # Calculate batch size
        optimal_batch_size = max(1, int(available_memory / memory_per_sample))
        
        return min(optimal_batch_size, 64)  # Cap at reasonable maximum
    
    def optimize_model_memory(self, model: LSTMPortfolioNetwork) -> LSTMPortfolioNetwork:
        """Apply memory optimization techniques."""
        # Enable gradient checkpointing
        model.lstm = torch.utils.checkpoint.checkpoint_wrapper(model.lstm)
        
        return model
```

## Integration with Existing Framework

### Portfolio Model Interface Implementation
```python
class LSTMPortfolioModel(PortfolioModel):
    def __init__(self, 
                 constraints: PortfolioConstraints,
                 model_config: LSTMModelConfig,
                 data_config: LSTMDataConfig):
        super().__init__(constraints)
        self.model_config = model_config
        self.data_config = data_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.sequence_generator = LSTMSequenceGenerator(data_config)
        self.memory_manager = LSTMMemoryManager()
        
        # Model will be created during fit()
        self.model = None
        self.scaler = None
        
    def fit(self, 
            returns: pd.DataFrame,
            universe: List[str], 
            fit_period: Tuple[pd.Timestamp, pd.Timestamp]) -> None:
        """Train LSTM model on historical data."""
        
        # Generate training sequences
        sequences, targets, asset_names = self.sequence_generator.create_sequences(
            returns, universe, fit_period[0], fit_period[1]
        )
        
        # Initialize model with proper input size
        self.model_config.input_features = sequences.shape[-1]
        self.model = LSTMPortfolioNetwork(self.model_config)
        
        # Optimize for memory constraints
        optimal_batch_size = self.memory_manager.calculate_optimal_batch_size(
            self.model, 
            self.data_config.sequence_length,
            len(universe),
            self.model_config.input_features
        )
        
        self.model = self.memory_manager.optimize_model_memory(self.model)
        
        # Create data loaders
        dataset = TensorDataset(sequences, targets)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=optimal_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=optimal_batch_size, shuffle=False)
        
        # Train model
        trainer = LSTMTrainer(self.model, self.model_config, self.device)
        
        best_val_sharpe = -np.inf
        patience_counter = 0
        
        for epoch in range(self.model_config.max_epochs):
            # Training
            train_metrics = trainer.train_epoch(train_loader)
            
            # Validation
            val_metrics = trainer.validate(val_loader)
            
            # Early stopping
            if val_metrics['val_sharpe'] > best_val_sharpe:
                best_val_sharpe = val_metrics['val_sharpe']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'lstm_best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.model_config.patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            trainer.scheduler.step(val_metrics['val_sharpe'])
        
        # Load best model
        self.model.load_state_dict(torch.load('lstm_best_model.pth'))
        
    def predict_weights(self, 
                       date: pd.Timestamp,
                       universe: List[str]) -> pd.Series:
        """Generate portfolio weights for rebalancing date."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get recent sequence for prediction
        end_date = date - pd.Timedelta(days=1)  # Previous trading day
        start_date = end_date - pd.Timedelta(days=self.data_config.sequence_length + 30)
        
        # Generate prediction sequence
        sequences, _, asset_names = self.sequence_generator.create_sequences(
            returns_df=None,  # Will load from data pipeline
            universe_calendar=None,
            start_date=start_date,
            end_date=end_date
        )
        
        # Create asset mask
        asset_mask = torch.ones(len(universe), dtype=torch.bool)
        
        # Predict weights
        self.model.eval()
        with torch.no_grad():
            sequences = sequences.unsqueeze(0).to(self.device)  # Add batch dimension
            asset_mask = asset_mask.unsqueeze(0).to(self.device)
            
            _, portfolio_weights = self.model(sequences, asset_mask)
            weights = portfolio_weights.squeeze(0).cpu().numpy()
        
        # Apply constraints
        weights_series = pd.Series(weights, index=universe)
        weights_series = self._apply_constraints(weights_series)
        
        return weights_series
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for analysis."""
        return {
            "model_type": "LSTM",
            "sequence_length": self.data_config.sequence_length,
            "hidden_size": self.model_config.hidden_size,
            "num_layers": self.model_config.num_layers,
            "attention_heads": self.model_config.attention_heads,
            "parameters": sum(p.numel() for p in self.model.parameters()) if self.model else 0
        }
```

## Hydra Configuration

### Default LSTM Configuration
```yaml
# configs/models/lstm_default.yaml
model:
  _target_: src.models.lstm.model.LSTMPortfolioModel
  
  model_config:
    input_features: 5
    hidden_size: 128
    num_layers: 3
    dropout: 0.3
    bidirectional: true
    attention_heads: 8
    attention_dropout: 0.1
    forecast_horizon: 21
    temperature: 1.0
    learning_rate: 0.001
    weight_decay: 0.00001
    batch_size: 32
    max_epochs: 100
    patience: 15
    gradient_accumulation_steps: 4
    mixed_precision: true
    gradient_clip_norm: 1.0

  data_config:
    sequence_length: 60
    forecast_horizon: 21
    feature_set: ['returns', 'volatility_20d', 'momentum_20d', 'rsi_14d', 'volume_ratio_20d']
    min_history_days: 100
    standardize_features: true
    handle_missing: 'forward_fill'

  constraints:
    long_only: true
    top_k_positions: 50
    max_position_weight: 0.10
    max_monthly_turnover: 0.20
    transaction_cost_bps: 10.0
```

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_models/test_lstm.py
class TestLSTMModel:
    def test_sequence_generation(self):
        """Test LSTM sequence generation from returns data."""
        # Test with sample data
        pass
        
    def test_model_forward_pass(self):
        """Test LSTM model forward pass."""
        # Test model outputs correct shapes
        pass
        
    def test_memory_optimization(self):
        """Test GPU memory usage within limits."""
        # Verify memory constraints
        pass
        
    def test_constraint_application(self):
        """Test portfolio constraint enforcement."""
        # Verify constraint compliance
        pass
```

## Implementation Timeline

- **Week 1**: Sequence generation and data pipeline integration
- **Week 2**: LSTM model architecture and training loop
- **Week 3**: Memory optimization and constraint integration  
- **Week 4**: Testing, validation, and performance tuning

This specification provides complete implementation details for the LSTM module using your existing data infrastructure, GPU constraints, and framework architecture.