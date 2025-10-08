# Frontend Development Guide

## Quick Start

This guide provides step-by-step instructions for developing the user-facing portfolio management interface for the GNN Portfolio Optimization system.

## Prerequisites

### System Requirements
- **Python 3.12+** (existing project requirement)
- **Node.js 16+** (if using React option in future)
- **Git** for version control
- **Modern web browser** (Chrome, Firefox, Safari, Edge)

### Existing Codebase Requirements
- Working GNN portfolio system with trained models
- Data pipeline operational with results in `data/final_new_pipeline/`
- Backtesting framework functional
- Existing visualization system in `src/evaluation/reporting/`

## Development Setup

### 1. Install Frontend Dependencies

```bash
# Navigate to project root
cd /path/to/GNN

# Install Streamlit and additional frontend dependencies
pip install streamlit>=1.28.0 plotly>=5.15.0 watchdog>=3.0.0

# Verify installation
streamlit --version
```

### 2. Create Frontend Directory Structure

```bash
# Create frontend directory structure
mkdir -p frontend/{components,styles,utils,config,tests}
mkdir -p frontend/components/{dashboard,strategy,monitoring,analytics,common}
mkdir -p frontend/tests/{unit,integration,ui}

# Create initial files
touch frontend/app.py
touch frontend/config/ui_config.yaml
touch frontend/styles/main.css
```

### 3. Basic Application Setup

**Create `frontend/app.py`:**
```python
import streamlit as st
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import existing components
from src.evaluation.reporting.interactive import InteractiveDashboard

def main():
    st.set_page_config(
        page_title="QuantEdge AI Portfolio Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🧠 QuantEdge AI Portfolio Platform")
    st.write("Professional AI-Powered Portfolio Management")
    
    # Test integration with existing codebase
    if st.button("Test Backend Integration"):
        try:
            # Test loading existing visualization components
            dashboard = InteractiveDashboard()
            st.success("✅ Backend integration successful!")
        except Exception as e:
            st.error(f"❌ Backend integration failed: {e}")

if __name__ == "__main__":
    main()
```

### 4. Test Basic Setup

```bash
# Run the basic application
cd frontend
streamlit run app.py

# Should open browser to http://localhost:8501
```

## Development Workflow

### Component Development Pattern

**1. Create Component Base Class**

```python
# frontend/components/common/base_component.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import streamlit as st
import pandas as pd

class BasePortfolioComponent(ABC):
    """Base class for all portfolio UI components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.data_cache = {}
        
    @abstractmethod
    def render(self) -> None:
        """Main rendering method - implement in subclasses"""
        pass
    
    def load_data(self) -> Dict[str, Any]:
        """Load required data - override in subclasses"""
        return {}
    
    def validate_data(self, data: Dict[str, Any]) -> bool:
        """Validate loaded data - override in subclasses"""
        return True
        
    def handle_error(self, error: Exception) -> None:
        """Standard error handling"""
        st.error(f"Component Error: {str(error)}")
        if self.config.get('debug_mode', False):
            st.exception(error)
```

**2. Implement Specific Components**

```python
# frontend/components/dashboard/portfolio_overview.py
from ..common.base_component import BasePortfolioComponent
import streamlit as st
from pathlib import Path
import json

class PortfolioOverviewComponent(BasePortfolioComponent):
    """Main portfolio overview dashboard component"""
    
    def load_data(self) -> Dict[str, Any]:
        """Load portfolio overview data from file system"""
        try:
            # Load latest portfolio data
            results_path = Path("results")
            
            data = {
                "portfolio_value": 2458392,  # Mock data for development
                "daily_change": 0.0098,
                "monthly_change": 0.0614,
                "ai_status": "active",
                "sharpe_ratio": 1.24
            }
            
            # Try to load real data if available
            if (results_path / "latest_portfolio_summary.json").exists():
                with open(results_path / "latest_portfolio_summary.json", 'r') as f:
                    real_data = json.load(f)
                    data.update(real_data)
                    
            return data
            
        except Exception as e:
            self.handle_error(e)
            return {}
    
    def render(self) -> None:
        """Render portfolio overview cards"""
        data = self.load_data()
        
        if not self.validate_data(data):
            st.warning("Portfolio data unavailable")
            return
            
        # Create metric cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Portfolio Value",
                f"${data['portfolio_value']:,.0f}",
                f"{data['daily_change']:+.2%}"
            )
            
        with col2:
            st.metric(
                "Monthly Return", 
                f"{data['monthly_change']:+.2%}",
                "vs benchmark"
            )
            
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{data['sharpe_ratio']:.2f}",
                "Risk-adjusted"
            )
            
        with col4:
            status_color = "🟢" if data['ai_status'] == 'active' else "🟡"
            st.metric(
                "AI Status",
                f"{status_color} {data['ai_status'].title()}",
                "Model confidence: 87%"
            )
```

### Navigation System Development

**Create Navigation Component:**

```python
# frontend/components/common/navigation.py
import streamlit as st
from typing import Dict, Callable

class NavigationComponent:
    """Sidebar navigation for the platform"""
    
    def __init__(self):
        self.pages = {
            "📊 Portfolio Dashboard": self._dashboard_page,
            "🎯 Strategy Selection": self._strategy_page,
            "📈 Live Monitoring": self._monitoring_page,
            "🧠 AI Insights": self._insights_page,
            "📋 Reports": self._reports_page
        }
        
    def render(self) -> str:
        """Render navigation and return selected page"""
        st.sidebar.title("QuantEdge AI")
        st.sidebar.caption("Portfolio Platform")
        
        # Navigation menu
        selected = st.sidebar.selectbox(
            "Navigation",
            options=list(self.pages.keys()),
            key="main_navigation"
        )
        
        # User info section
        with st.sidebar.expander("👤 User Info", expanded=False):
            st.write("**John Smith**")
            st.write("Portfolio Manager")
            st.write("Last login: 2 hours ago")
            
        # System status
        with st.sidebar.expander("🔧 System Status", expanded=False):
            st.success("✅ Data Pipeline: Operational")
            st.success("✅ AI Models: Active")  
            st.info("📊 Last Update: 5 min ago")
            
        return selected
        
    def _dashboard_page(self):
        """Dashboard page implementation"""
        from ..dashboard.portfolio_overview import PortfolioOverviewComponent
        overview = PortfolioOverviewComponent()
        overview.render()
        
    def _strategy_page(self):
        st.header("🎯 Strategy Selection")
        st.info("Strategy selection interface - Coming soon")
        
    def _monitoring_page(self):
        st.header("📈 Live Portfolio Monitoring") 
        st.info("Live monitoring interface - Coming soon")
        
    def _insights_page(self):
        st.header("🧠 AI Insights")
        st.info("AI insights and recommendations - Coming soon")
        
    def _reports_page(self):
        st.header("📋 Professional Reports")
        st.info("Report generation system - Coming soon")
        
    def get_page_function(self, page_name: str) -> Callable:
        """Get the function for a specific page"""
        return self.pages.get(page_name, self._dashboard_page)
```

### Styling and CSS Integration

**Create Main Stylesheet:**

```css
/* frontend/styles/main.css */

/* Import Google Fonts for professional appearance */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* CSS Variables for consistent theming */
:root {
    /* Primary colors */
    --primary-navy: #1e3a5f;
    --primary-blue: #2e5c8a;
    --primary-light: #4a90c2;
    
    /* Accent colors */
    --accent-green: #27ae60;
    --accent-red: #e74c3c;
    --accent-orange: #f39c12;
    --accent-purple: #8e44ad;
    
    /* Neutral colors */
    --gray-50: #f8f9fa;
    --gray-100: #e9ecef;
    --gray-200: #dee2e6;
    --gray-600: #6c757d;
    --gray-800: #343a40;
    --gray-900: #212529;
    
    /* Spacing scale */
    --space-xs: 0.25rem;
    --space-sm: 0.5rem;
    --space-md: 1rem;
    --space-lg: 1.5rem;
    --space-xl: 2rem;
    
    /* Border radius */
    --radius-sm: 4px;
    --radius-md: 8px;
    --radius-lg: 12px;
}

/* Global font override */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Custom metric card styling */
[data-testid="metric-container"] {
    background: white;
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
    padding: var(--space-lg);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: box-shadow 0.2s ease;
}

[data-testid="metric-container"]:hover {
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Sidebar customization */
.css-1d391kg {
    background-color: var(--primary-navy);
    color: white;
}

/* Main content area */
.main .block-container {
    padding-top: var(--space-xl);
    padding-bottom: var(--space-xl);
}

/* Button styling */
.stButton > button {
    background-color: var(--primary-blue);
    color: white;
    border: none;
    border-radius: var(--radius-sm);
    padding: var(--space-sm) var(--space-md);
    font-weight: 500;
    transition: background-color 0.2s ease;
}

.stButton > button:hover {
    background-color: var(--primary-light);
}

/* Success/Error styling */
.stSuccess {
    background-color: rgba(39, 174, 96, 0.1);
    border-left: 4px solid var(--accent-green);
}

.stError {
    background-color: rgba(231, 76, 60, 0.1);
    border-left: 4px solid var(--accent-red);
}

/* Data table styling */
.dataframe {
    border: 1px solid var(--gray-200);
    border-radius: var(--radius-md);
}

.dataframe th {
    background-color: var(--gray-50);
    font-weight: 600;
    color: var(--gray-800);
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .main .block-container {
        padding-left: var(--space-sm);
        padding-right: var(--space-sm);
    }
}
```

**Apply CSS to Streamlit App:**

```python
# frontend/utils/styling.py
import streamlit as st
from pathlib import Path

def load_css():
    """Load custom CSS styling"""
    css_file = Path(__file__).parent.parent / "styles" / "main.css"
    
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning("CSS file not found - using default styling")

def apply_custom_styling():
    """Apply additional inline styling for specific components"""
    st.markdown("""
    <style>
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom portfolio platform styling */
    .portfolio-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #2e5c8a 100%);
        color: white;
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }
    
    .metric-highlight {
        font-size: 2rem;
        font-weight: 700;
        color: #27ae60;
    }
    </style>
    """, unsafe_allow_html=True)
```

## Integration with Existing Backend

### Data Loading Utilities

```python
# frontend/utils/data_loaders.py
import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

class PortfolioDataLoader:
    """Utility class for loading portfolio data from existing file system"""
    
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(".")
        self.data_path = self.base_path / "data" / "final_new_pipeline"
        self.results_path = self.base_path / "results"
        self.models_path = self.base_path / "models"
        
    @st.cache_data(ttl=300)  # Cache for 5 minutes
    def load_latest_portfolio_weights(_self) -> pd.DataFrame:
        """Load most recent portfolio weights"""
        try:
            weights_file = _self.results_path / "latest_portfolio_weights.csv"
            if weights_file.exists():
                return pd.read_csv(weights_file, index_col=0)
            else:
                # Return mock data for development
                return _self._create_mock_weights()
        except Exception as e:
            st.warning(f"Could not load portfolio weights: {e}")
            return pd.DataFrame()
    
    @st.cache_data(ttl=600)  # Cache for 10 minutes
    def load_model_performance_history(_self) -> Dict[str, Any]:
        """Load historical model performance data"""
        try:
            performance_files = list(_self.results_path.glob("*_results.json"))
            results = {}
            
            for file in performance_files:
                model_name = file.stem.replace("_results", "")
                with open(file, 'r') as f:
                    results[model_name] = json.load(f)
                    
            if not results:
                # Return mock data for development
                results = _self._create_mock_performance()
                
            return results
        except Exception as e:
            st.warning(f"Could not load performance history: {e}")
            return {}
    
    def check_system_status(self) -> Dict[str, str]:
        """Check status of various system components"""
        status = {}
        
        # Check data availability
        if self.data_path.exists() and list(self.data_path.glob("*.parquet")):
            status["data_pipeline"] = "✅ Operational"
        else:
            status["data_pipeline"] = "❌ No Data"
            
        # Check model availability  
        model_files = list(self.models_path.glob("*.pt")) + list(self.models_path.glob("*.pkl"))
        if model_files:
            status["ai_models"] = f"✅ {len(model_files)} Models Available"
        else:
            status["ai_models"] = "⚠️ No Trained Models"
            
        # Check results availability
        result_files = list(self.results_path.glob("*.json"))
        if result_files:
            status["results"] = f"✅ {len(result_files)} Result Sets"
        else:
            status["results"] = "⚠️ No Results Available"
            
        return status
    
    def _create_mock_weights(self) -> pd.DataFrame:
        """Create mock portfolio weights for development"""
        mock_data = {
            'ticker': ['NVDA', 'META', 'AVGO', 'TSM', 'AMZN'],
            'weight': [0.048, 0.042, 0.039, 0.037, 0.034],
            'sector': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology']
        }
        return pd.DataFrame(mock_data).set_index('ticker')
    
    def _create_mock_performance(self) -> Dict[str, Any]:
        """Create mock performance data for development"""
        return {
            "gat_model": {
                "sharpe_ratio": 1.24,
                "annual_return": 0.128,
                "max_drawdown": -0.152,
                "volatility": 0.103
            },
            "hrp_model": {
                "sharpe_ratio": 1.18,
                "annual_return": 0.119,
                "max_drawdown": -0.168,
                "volatility": 0.101
            }
        }
```

### Backend Integration Service

```python
# frontend/utils/backend_integration.py
import streamlit as st
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class BackendIntegrationService:
    """Service for integrating with existing GNN backend"""
    
    def __init__(self):
        self.models_available = self._check_model_availability()
        
    def _check_model_availability(self) -> Dict[str, bool]:
        """Check which models are available for use"""
        available = {}
        
        try:
            # Test GAT model import
            from src.models.gat.model import GATPortfolioModel
            available['gat'] = True
        except ImportError:
            available['gat'] = False
            
        try:
            # Test HRP model import
            from src.models.hrp.model import HRPModel
            available['hrp'] = True
        except ImportError:
            available['hrp'] = False
            
        try:
            # Test LSTM model import
            from src.models.lstm.model import LSTMModel
            available['lstm'] = True
        except ImportError:
            available['lstm'] = False
            
        return available
    
    @st.cache_resource
    def get_interactive_dashboard(_self):
        """Get existing interactive dashboard component"""
        try:
            from src.evaluation.reporting.interactive import InteractiveDashboard
            return InteractiveDashboard()
        except ImportError as e:
            st.error(f"Could not load interactive dashboard: {e}")
            return None
    
    @st.cache_resource  
    def get_performance_tables(_self):
        """Get existing performance comparison tables"""
        try:
            from src.evaluation.reporting.tables import PerformanceComparisonTables
            return PerformanceComparisonTables()
        except ImportError as e:
            st.error(f"Could not load performance tables: {e}")
            return None
            
    def run_quick_backtest(self, strategy: str, config: Dict[str, Any]) -> Optional[Dict]:
        """Run a quick backtest using existing engine"""
        try:
            # Import existing backtesting components
            from src.evaluation.backtest.rolling_engine import RollingBacktestEngine
            from src.models.base.constraints import PortfolioConstraints
            
            # Configure constraints
            constraints = PortfolioConstraints(**config.get('constraints', {}))
            
            # This would be expanded to actually run backtests
            # For now, return mock results
            return {
                "sharpe_ratio": 1.24,
                "annual_return": 0.128,
                "max_drawdown": -0.152,
                "status": "completed"
            }
            
        except Exception as e:
            st.error(f"Backtest failed: {e}")
            return None
```

## Testing Strategy

### Unit Testing Setup

```python
# frontend/tests/unit/test_components.py
import unittest
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add frontend to path
frontend_path = Path(__file__).parent.parent.parent
sys.path.append(str(frontend_path))

from components.dashboard.portfolio_overview import PortfolioOverviewComponent

class TestPortfolioOverview(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.component = PortfolioOverviewComponent()
        
    def test_component_initialization(self):
        """Test component initializes correctly"""
        self.assertIsNotNone(self.component)
        self.assertEqual(self.component.config, {})
        
    @patch('components.dashboard.portfolio_overview.Path')
    def test_load_data_with_mock_file(self, mock_path):
        """Test data loading with mocked file system"""
        # Mock file existence
        mock_path.return_value.exists.return_value = False
        
        data = self.component.load_data()
        
        # Should return default mock data
        self.assertIn('portfolio_value', data)
        self.assertIsInstance(data['portfolio_value'], (int, float))
        
    def test_validate_data(self):
        """Test data validation"""
        valid_data = {
            'portfolio_value': 1000000,
            'daily_change': 0.02,
            'sharpe_ratio': 1.24
        }
        
        self.assertTrue(self.component.validate_data(valid_data))
        
        invalid_data = {}
        self.assertFalse(self.component.validate_data(invalid_data))

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
# frontend/tests/integration/test_backend_integration.py
import unittest
import sys
from pathlib import Path

# Add paths
frontend_path = Path(__file__).parent.parent.parent
project_root = frontend_path.parent
sys.path.extend([str(frontend_path), str(project_root)])

from utils.backend_integration import BackendIntegrationService

class TestBackendIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration service"""
        self.service = BackendIntegrationService()
        
    def test_model_availability_check(self):
        """Test that model availability checking works"""
        availability = self.service.models_available
        
        # Should return dict with boolean values
        self.assertIsInstance(availability, dict)
        for model, available in availability.items():
            self.assertIsInstance(available, bool)
            
    def test_dashboard_integration(self):
        """Test integration with existing dashboard"""
        dashboard = self.service.get_interactive_dashboard()
        
        # Should either return dashboard object or None
        if dashboard is not None:
            # Test that it has expected methods
            self.assertTrue(hasattr(dashboard, 'create_performance_dashboard'))

if __name__ == '__main__':
    unittest.main()
```

## Deployment and Running

### Local Development Server

```bash
# Create development script
# frontend/run_dev.py
import streamlit.web.cli as stcli
import sys
from pathlib import Path

def run_dev_server():
    """Run development server with proper configuration"""
    app_path = Path(__file__).parent / "app.py"
    
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.port=8501",
        "--server.address=localhost",
        "--server.headless=false",
        "--browser.gatherUsageStats=false"
    ]
    
    stcli.main()

if __name__ == "__main__":
    run_dev_server()
```

### Production Deployment Script

```bash
#!/bin/bash
# scripts/deploy_frontend.sh

echo "🚀 Deploying QuantEdge AI Portfolio Platform"

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
required_version="3.12"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.12+ required. Current version: $python_version"
    exit 1
fi

# Check if backend is operational
echo "🔍 Checking backend system..."
python3 -c "
import sys
from pathlib import Path
sys.path.append('.')

try:
    from src.models.gat.model import GATPortfolioModel
    print('✅ GAT model available')
except ImportError:
    print('⚠️ GAT model not available')

try:
    from src.evaluation.reporting.interactive import InteractiveDashboard
    print('✅ Visualization system available')
except ImportError:
    print('⚠️ Visualization system not available')
"

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
pip install -q streamlit plotly watchdog

# Check data availability
echo "📊 Checking data availability..."
if [ -d "data/final_new_pipeline" ] && [ "$(ls -A data/final_new_pipeline)" ]; then
    echo "✅ Data pipeline operational"
else
    echo "⚠️ No data found - will use mock data"
fi

# Launch application
echo "🌐 Starting QuantEdge AI Portfolio Platform..."
cd frontend
streamlit run app.py --server.port 8501 --server.headless false

echo "Platform available at: http://localhost:8501"
```

## Debugging and Troubleshooting

### Common Issues and Solutions

**1. Import Errors from Backend**
```python
# Add this to your component files if imports fail
import sys
from pathlib import Path

# Ensure project root is in Python path
project_root = Path(__file__).resolve().parents[2]  # Adjust as needed
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
```

**2. Data Loading Issues**
```python
# Add debugging to data loaders
def load_with_debug(self, file_path: str):
    """Load data with debug information"""
    print(f"DEBUG: Attempting to load {file_path}")
    print(f"DEBUG: File exists: {Path(file_path).exists()}")
    print(f"DEBUG: Current working directory: {Path.cwd()}")
    
    # Your loading logic here
```

**3. CSS Not Loading**
```python
# Force CSS reload in development
def force_css_reload():
    """Force reload CSS during development"""
    import time
    cache_buster = int(time.time())
    
    with open("styles/main.css") as f:
        css = f.read()
        
    st.markdown(
        f"<style>/* Cache buster: {cache_buster} */\n{css}</style>", 
        unsafe_allow_html=True
    )
```

### Performance Optimization

**1. Caching Strategy**
```python
# Use appropriate caching for different data types
@st.cache_data(ttl=60)    # 1 minute for real-time data
@st.cache_data(ttl=300)   # 5 minutes for portfolio data  
@st.cache_data(ttl=3600)  # 1 hour for historical data
@st.cache_resource        # For expensive objects (models, connections)
```

**2. Memory Management**
```python
# Clear caches when needed
if st.button("Refresh All Data"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.rerun()
```

## Next Steps

1. **Start with Story 6.1**: Implement core dashboard framework
2. **Add Strategy Selection**: Build Story 6.2 components
3. **Create Monitoring Interface**: Implement Story 6.3 features
4. **Add AI Insights**: Develop Story 6.4 recommendations
5. **Build Reporting**: Create Story 6.6 export functionality

## Support and Resources

- **Streamlit Documentation**: https://docs.streamlit.io/
- **Plotly Documentation**: https://plotly.com/python/
- **Project Backend Documentation**: See `docs/` directory
- **Component Examples**: Check existing `src/evaluation/reporting/` classes

---

**Ready to start developing? Begin with the basic setup above and incrementally add components following the established patterns.**