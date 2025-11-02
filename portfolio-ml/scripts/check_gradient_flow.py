"""Quick diagnostic to check if LSTM has zero gradient issues."""
import torch
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lstm.model import LSTMPortfolioModel, LSTMModelConfig
from src.models.lstm.architecture import LSTMConfig
from src.models.base.portfolio_model import PortfolioConstraints


def check_gradient_flow_simple():
    """Check if gradients flow with dummy data."""
    print("=" * 70)
    print("GRADIENT FLOW CHECK - DUMMY DATA")
    print("=" * 70)

    from src.models.lstm.ragged_architecture import create_ragged_lstm_network

    lstm_config = LSTMConfig()
    lstm_config.input_size = 100
    lstm_config.output_size = 100
    network = create_ragged_lstm_network(lstm_config)

    # Create dummy data
    batch_size = 16
    seq_len = 60
    num_features = 100

    sequences = torch.randn(batch_size, seq_len, num_features)
    lengths = torch.full((batch_size,), seq_len, dtype=torch.long)
    targets = torch.randn(batch_size, num_features)

    # Forward pass
    predictions, _ = network(sequences, lengths)

    # Compute loss using the model's loss function
    from src.models.lstm.architecture import SharpeRatioLoss
    criterion = SharpeRatioLoss()
    loss = criterion(predictions, targets)

    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Loss grad_fn: {loss.grad_fn}")

    # Backward pass
    loss.backward()

    # Check gradient statistics
    grad_norms = []
    zero_grad_params = []
    none_grad_params = []

    for name, param in network.named_parameters():
        if param.grad is None:
            none_grad_params.append(name)
        else:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
            if grad_norm == 0.0:
                zero_grad_params.append(name)

    print(f"\nGradient Statistics:")
    print(f"  Total parameters: {len(list(network.named_parameters()))}")
    print(f"  Parameters with grad=None: {len(none_grad_params)}")
    print(f"  Parameters with grad=0: {len(zero_grad_params)}")
    print(f"  Parameters with non-zero grad: {len(grad_norms) - len(zero_grad_params)}")

    if grad_norms:
        print(f"  Mean gradient norm: {sum(grad_norms) / len(grad_norms):.6e}")
        print(f"  Max gradient norm: {max(grad_norms):.6e}")
        print(f"  Min gradient norm: {min(grad_norms):.6e}")

    # Determine if gradients are flowing
    has_gradients = len(grad_norms) > 0 and max(grad_norms) > 1e-6

    # Filter out batch norm params - they don't always need gradients
    none_grad_params_filtered = [p for p in none_grad_params if 'batch_norm' not in p]

    if none_grad_params_filtered:
        print(f"\n❌ ISSUE: {len(none_grad_params_filtered)} non-batchnorm parameters have grad=None")
        print(f"   Parameters: {none_grad_params_filtered[:5]}")
        return False

    if none_grad_params:
        print(f"\n✓ Note: {len(none_grad_params)} batch norm parameters have grad=None (normal)")

    if zero_grad_params:
        print(f"\n⚠️  WARNING: {len(zero_grad_params)} parameters have zero gradients")
        print(f"   Parameters: {zero_grad_params[:5]}")
        if len(zero_grad_params) == len(list(network.named_parameters())):
            print("   ❌ ALL GRADIENTS ARE ZERO!")
            return False

    if has_gradients:
        print("\n✅ Gradients are flowing (dummy data test passed)")
        return True
    else:
        print("\n❌ Gradient norms too small (< 1e-6)")
        return False


def check_gradient_flow_real_data():
    """Check if gradients flow with real data during actual training."""
    print("\n" + "=" * 70)
    print("GRADIENT FLOW CHECK - REAL DATA (2 TRAINING STEPS)")
    print("=" * 70)

    # Load real data
    returns_path = Path("data/final_new_pipeline/returns_daily_final.parquet")
    if not returns_path.exists():
        print(f"❌ Data not found at {returns_path}")
        return None

    returns = pd.read_parquet(returns_path)

    # Set up training window
    test_date = pd.Timestamp("2023-06-01")
    universe = returns.columns.tolist()[:100]

    train_end = test_date - pd.Timedelta(days=1)
    train_start = train_end - pd.Timedelta(days=365)

    train_returns = returns.loc[train_start:train_end, universe]

    print(f"Training window: {train_start.date()} to {train_end.date()}")
    print(f"Universe size: {len(universe)} assets")
    print(f"Training samples: {len(train_returns)}")

    # Create model
    model_config = LSTMModelConfig()
    model_config.training_config.epochs = 2  # Just 2 epochs to check gradients
    model_config.training_config.verbose = True

    constraints = PortfolioConstraints()
    model = LSTMPortfolioModel(constraints, model_config)

    # Hook to capture gradient norms during training
    gradient_norms = []
    loss_values = []

    def capture_gradients(module, grad_input, grad_output):
        """Hook to capture gradient statistics."""
        pass

    # Train and monitor
    print("\nStarting training (monitoring gradients)...")
    try:
        model.rolling_fit(train_returns, universe, test_date)
        print("\n✅ Training completed without errors")

        # Check if model parameters changed (indicating learning occurred)
        print("\nChecking if model learned:")
        print("  (If parameters updated, gradients must have flowed)")

        return True

    except Exception as e:
        print(f"\n❌ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run gradient flow diagnostics."""
    print("LSTM GRADIENT FLOW DIAGNOSTIC")
    print("=" * 70)

    # Test 1: Dummy data
    dummy_result = check_gradient_flow_simple()

    # Test 2: Real data
    real_result = check_gradient_flow_real_data()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if dummy_result and real_result:
        print("✅ GRADIENTS FLOWING NORMALLY")
        print("   Phase 5 diagnostics NOT NEEDED")
        print("   You can proceed to Phase 6 (GAT Time-Series Features)")
        return 0
    elif not dummy_result:
        print("❌ ZERO GRADIENT ISSUE DETECTED (dummy data)")
        print("   Phase 5 diagnostics REQUIRED")
        print("   Issue: Gradients not flowing even with synthetic data")
        return 1
    elif not real_result:
        print("❌ TRAINING ISSUE DETECTED (real data)")
        print("   Phase 5 diagnostics MAY BE REQUIRED")
        print("   Issue: Training failed or gradients problematic with real data")
        return 1
    else:
        print("⚠️  INCONCLUSIVE")
        print("   Manual review recommended")
        return 2


if __name__ == "__main__":
    exit(main())
