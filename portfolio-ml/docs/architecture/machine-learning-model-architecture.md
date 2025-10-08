# Machine Learning Model Architecture

## Base Portfolio Model Interface

All ML approaches implement a unified interface ensuring consistent evaluation:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

@dataclass
class PortfolioConstraints:
    long_only: bool = True                      # No short positions
    top_k_positions: Optional[int] = None       # Maximum number of positions
    max_position_weight: float = 0.10           # Maximum single position
    max_monthly_turnover: float = 0.20          # Turnover limit
    transaction_cost_bps: float = 10.0          # Linear transaction costs

class PortfolioModel(ABC):
    def __init__(self, constraints: PortfolioConstraints):
        self.constraints = constraints
        
    @abstractmethod
    def fit(self, 
            returns: pd.DataFrame, 
            universe: List[str], 
            fit_period: Tuple[pd.Timestamp, pd.Timestamp]) -> None:
        """Train model on historical data."""
        
    @abstractmethod 
    def predict_weights(self, 
                       date: pd.Timestamp,
                       universe: List[str]) -> pd.Series:
        """Generate portfolio weights for rebalancing date."""
        
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Return model metadata for analysis."""
```

## Hierarchical Risk Parity (HRP) Architecture

The HRP implementation focuses on clustering-aware allocation without matrix inversion:

```python
class HRPModel(PortfolioModel):
    def __init__(self, 
                 constraints: PortfolioConstraints,
                 lookback_days: int = 756,           # 3 years
                 linkage_method: str = "single",     # Clustering linkage
                 distance_metric: str = "correlation"):
        
    def _build_correlation_distance(self, returns: pd.DataFrame) -> np.ndarray:
        """Convert correlation matrix to distance metric."""
        
    def _hierarchical_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Build asset hierarchy using correlation distances."""
        
    def _recursive_bisection(self, 
                           covariance_matrix: np.ndarray, 
                           cluster_hierarchy: np.ndarray) -> np.ndarray:
        """Allocate capital through recursive cluster bisection."""
```

**Key Features:**
- **Clustering Algorithm**: Single-linkage hierarchical clustering on correlation distance
- **Risk Budgeting**: Equal risk contribution within cluster levels
- **Matrix Stability**: Avoids unstable covariance matrix inversion
- **Parameter Sensitivity**: Robust to hyperparameter choices

## LSTM Temporal Network Architecture

The LSTM implementation captures temporal dependencies in asset returns for dynamic allocation:

```python
class LSTMPortfolioModel(PortfolioModel):
    def __init__(self, 
                 constraints: PortfolioConstraints,
                 sequence_length: int = 60,          # 60-day lookback
                 hidden_size: int = 128,            # LSTM hidden dimensions
                 num_layers: int = 2,               # Stacked LSTM layers
                 dropout: float = 0.3):
        
    class LSTMNetwork(nn.Module):
        def __init__(self, 
                     input_size: int,               # Number of features per asset
                     hidden_size: int, 
                     num_layers: int,
                     output_size: int,              # Forecast horizon
                     dropout: float):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                              dropout=dropout, batch_first=True)
            self.attention = nn.MultiheadAttention(hidden_size, num_heads=8)
            self.output_projection = nn.Linear(hidden_size, output_size)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x shape: (batch_size, sequence_length, input_size)
            lstm_out, _ = self.lstm(x)
            
            # Apply attention mechanism
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            
            # Project to return forecasts
            forecasts = self.output_projection(attn_out[:, -1, :])
            return forecasts
```

**Architecture Features:**
- **Sequence Modeling**: 60-day rolling windows capture temporal patterns
- **Multi-Head Attention**: Focus on relevant historical periods
- **Regularization**: Dropout and batch normalization prevent overfitting
- **GPU Optimization**: Batch processing within 12GB VRAM constraints

**Training Strategy:**
```python
def train_lstm_model(model: LSTMNetwork, 
                    data_loader: DataLoader,
                    validation_loader: DataLoader,
                    epochs: int = 100) -> Dict[str, float]:
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    criterion = SharpeRatioLoss()  # Custom loss function
    
    best_val_sharpe = -np.inf
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase with gradient accumulation for memory efficiency
        model.train()
        for batch_idx, (sequences, targets) in enumerate(data_loader):
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            
            # Gradient accumulation for large batch sizes
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
```

## Graph Attention Network (GAT) Architecture

The GAT implementation leverages the existing comprehensive framework with enhancements for portfolio optimization:

**Existing GAT Framework Analysis:**
- ✅ Multi-head attention with GATv2 support
- ✅ Edge attribute integration (correlation strength, sign)
- ✅ Residual connections and layer normalization
- ✅ Memory-efficient implementation within GPU constraints
- ✅ Direct Sharpe ratio optimization capability

**Enhanced Portfolio-Specific Features:**
```python
class GATPortfolioOptimized(nn.Module):
    def __init__(self,
                 input_features: int,
                 hidden_dim: int = 64,
                 num_attention_heads: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.3,
                 edge_feature_dim: int = 3,         # [ρ, |ρ|, sign]
                 constraint_layer: bool = True):    # Enforce constraints in network
        
        # Multi-layer GAT backbone
        self.gat_layers = nn.ModuleList([
            GATLayer(input_features if i == 0 else hidden_dim,
                    hidden_dim,
                    num_attention_heads,
                    dropout,
                    edge_feature_dim) 
            for i in range(num_layers)
        ])
        
        # Portfolio optimization head with constraint enforcement
        if constraint_layer:
            self.portfolio_head = ConstrainedPortfolioLayer(
                hidden_dim,
                constraint_config=self.constraints
            )
        else:
            self.portfolio_head = nn.Linear(hidden_dim, 1)  # Return forecasts
            
    def forward(self, 
                node_features: torch.Tensor,      # [N, F] asset features
                edge_index: torch.Tensor,         # [2, E] graph edges
                edge_attr: torch.Tensor,          # [E, 3] edge attributes
                asset_mask: torch.Tensor) -> torch.Tensor:  # [N] valid assets
        
        # Multi-layer GAT processing with residual connections
        x = node_features
        for layer in self.gat_layers:
            x_new = layer(x, edge_index, edge_attr)
            x = x + x_new if x.size() == x_new.size() else x_new  # Residual
            x = F.dropout(x, self.dropout, training=self.training)
        
        # Portfolio weight generation with constraint enforcement
        if isinstance(self.portfolio_head, ConstrainedPortfolioLayer):
            weights = self.portfolio_head(x, asset_mask)
        else:
            # Convert return forecasts to weights via optimization layer
            return_forecasts = self.portfolio_head(x).squeeze()
            weights = self._markowitz_optimization(return_forecasts, asset_mask)
            
        return weights
```

**Memory Optimization for 12GB VRAM:**
```python
class GPUMemoryManager:
    def __init__(self, max_vram_gb: float = 11.0):  # Conservative 11GB limit
        self.max_memory = max_vram_gb * 1024**3
        
    def optimize_batch_size(self, 
                           model: nn.Module,
                           sample_input: Tuple[torch.Tensor, ...]) -> int:
        """Determine optimal batch size for memory constraints."""
        
    def gradient_checkpointing_wrapper(self, model: nn.Module) -> nn.Module:
        """Apply gradient checkpointing to reduce memory usage."""
        
    def mixed_precision_training(self, model: nn.Module) -> Tuple[nn.Module, torch.cuda.amp.GradScaler]:
        """Enable mixed precision training for memory efficiency."""
```

---
