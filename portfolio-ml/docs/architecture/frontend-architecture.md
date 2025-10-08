# Frontend Architecture for GNN Portfolio Platform

## Overview

This document defines the frontend architecture for the user-facing portfolio management platform built on top of the existing GNN portfolio optimization framework. The architecture prioritizes rapid development while maintaining professional quality and extensibility.

## Architecture Principles

### Design Philosophy
- **User-First Design**: Translate complex ML concepts into intuitive financial terminology
- **Performance Optimized**: Leverage existing backend infrastructure without duplication
- **File-Based Simplicity**: No database dependencies for rapid local deployment
- **Extensible Foundation**: Architecture supports future multi-user and cloud deployment

### Technical Principles
- **Component Reusability**: Modular components supporting multiple user workflows
- **Data Consistency**: Single source of truth from existing file-based storage
- **Progressive Enhancement**: Core functionality works without JavaScript/advanced features
- **Responsive Design**: Professional experience across desktop, tablet, and mobile

## Technology Stack

### Primary Technologies

**Frontend Framework: Streamlit 1.28+**
- **Rationale**: Rapid development, native Python integration, built-in professional components
- **Benefits**: Direct integration with existing ML codebase, automatic responsive design
- **Limitations**: Single-user by default, limited customization vs full web frameworks

**Visualization: Plotly 5.15+**
- **Rationale**: Leverage existing dashboard infrastructure from `src/evaluation/reporting/`
- **Benefits**: Interactive charts, professional financial visualizations, export capabilities
- **Integration**: Direct reuse of existing chart classes without modification

**Styling: Custom CSS + Streamlit Components**
- **Professional Theme**: Financial platform color scheme (navy, white, green accents)
- **Component Library**: Reusable styled components for consistency
- **Responsive Grid**: CSS Grid and Flexbox for adaptive layouts

### Supporting Technologies

**Data Processing: Existing Python Stack**
- **Pandas**: Portfolio calculations and data manipulation
- **NumPy**: Performance metric calculations
- **PyYAML**: Configuration file management

**File System Integration**
- **Parquet Files**: Leverage existing data storage from `data/final_new_pipeline/`
- **JSON Results**: Use existing backtest results format
- **YAML Configs**: Extend existing configuration system

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│ Streamlit Application (app.py)                                 │
│ ├── Navigation Framework                                        │
│ ├── Dashboard Components                                        │  
│ ├── Strategy Configuration                                      │
│ ├── Portfolio Monitoring                                        │
│ └── Reporting & Analytics                                       │
├─────────────────────────────────────────────────────────────────┤
│                    COMPONENT LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│ Reusable UI Components                                          │
│ ├── Portfolio Cards        ├── Performance Charts              │
│ ├── Strategy Selectors     ├── Holdings Tables                 │
│ ├── Configuration Panels   ├── Risk Monitors                   │
│ └── Attribution Analysis   └── AI Activity Feeds               │
├─────────────────────────────────────────────────────────────────┤
│                    DATA ACCESS LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│ File System Integration                                         │
│ ├── Portfolio Data Loaders ├── Configuration Managers          │
│ ├── Model Status Checkers  ├── Results Processors              │
│ └── Performance Calculators └── AI Decision Trackers            │
├─────────────────────────────────────────────────────────────────┤
│                    EXISTING BACKEND                             │
├─────────────────────────────────────────────────────────────────┤
│ ML Models (GAT/HRP/LSTM) | Data Pipeline | Backtesting Engine  │
│ Performance Analytics    | Risk Framework | Reporting System   │
└─────────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
frontend/
├── app.py                          # Main Streamlit application entry point
├── config/
│   ├── ui_config.yaml             # UI-specific configuration
│   ├── styling.yaml               # Theme and styling configuration
│   └── navigation.yaml            # Navigation structure definition
├── components/
│   ├── __init__.py
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── portfolio_overview.py  # Main dashboard components
│   │   ├── performance_cards.py   # Portfolio summary cards
│   │   └── ai_status_display.py   # AI strategy status indicators
│   ├── strategy/
│   │   ├── __init__.py
│   │   ├── strategy_selector.py   # Strategy selection interface
│   │   ├── configuration_panel.py # Portfolio configuration controls
│   │   └── outcomes_calculator.py # Expected outcomes display
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── portfolio_monitor.py   # Live portfolio monitoring
│   │   ├── holdings_analysis.py   # Position analysis and breakdown
│   │   ├── performance_attribution.py # Performance attribution
│   │   └── risk_dashboard.py      # Risk monitoring and alerts
│   ├── analytics/
│   │   ├── __init__.py
│   │   ├── backtesting_interface.py # Interactive backtesting
│   │   ├── ai_insights.py         # AI recommendations and insights
│   │   └── reporting_tools.py     # Report generation and export
│   └── common/
│       ├── __init__.py
│       ├── navigation.py          # Sidebar navigation component
│       ├── data_loaders.py        # File system data loading utilities
│       ├── validators.py          # Input validation and error handling
│       └── formatters.py          # Display formatting utilities
├── styles/
│   ├── main.css                   # Primary stylesheet
│   ├── components.css             # Component-specific styles
│   ├── responsive.css             # Mobile and tablet responsiveness
│   └── theme.css                  # Color scheme and typography
├── utils/
│   ├── __init__.py
│   ├── portfolio_calculator.py    # Portfolio value and P&L calculations
│   ├── performance_analyzer.py    # Performance metric calculations
│   ├── ai_tracker.py             # AI decision logging and analysis
│   └── integration_helpers.py     # Backend integration utilities
└── tests/
    ├── __init__.py
    ├── unit/                      # Unit tests for components
    ├── integration/               # Integration tests with backend
    └── ui/                        # UI and user experience tests
```

## Component Architecture

### Core Component Design

**Component Hierarchy:**
```python
# Base component pattern for reusability
class BasePortfolioComponent:
    """Base class for all portfolio-related UI components"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader()
        
    def render(self) -> None:
        """Main rendering method - implemented by subclasses"""
        raise NotImplementedError
        
    def load_data(self) -> Dict[str, Any]:
        """Load required data from file system"""
        raise NotImplementedError
        
    def validate_inputs(self, user_inputs: Dict) -> Tuple[bool, str]:
        """Validate user inputs with helpful error messages"""
        return True, ""

# Example implementation
class PortfolioOverviewCard(BasePortfolioComponent):
    def render(self):
        data = self.load_data()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Portfolio Value", 
                     f"${data['value']:,.0f}", 
                     f"{data['daily_change']:+.2%}")
        # ... additional metrics
        
    def load_data(self):
        return self.data_loader.get_latest_portfolio_data()
```

### State Management

**Session State Architecture:**
```python
# Centralized state management for user session
class PortfolioSessionManager:
    """Manage user session state across components"""
    
    @staticmethod
    def initialize_session():
        """Initialize default session state"""
        if 'portfolio_config' not in st.session_state:
            st.session_state.portfolio_config = {}
        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = None
        if 'current_portfolio' not in st.session_state:
            st.session_state.current_portfolio = None
            
    @staticmethod
    def update_portfolio_config(updates: Dict):
        """Update portfolio configuration in session"""
        st.session_state.portfolio_config.update(updates)
        
    @staticmethod
    def get_portfolio_status():
        """Get current portfolio status and data"""
        return st.session_state.get('current_portfolio')
```

## Data Integration Layer

### File System Integration

**Data Loading Architecture:**
```python
class PortfolioDataLoader:
    """Centralized data loading from existing file system"""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.data_path = self.base_path / "data" / "final_new_pipeline"
        self.results_path = self.base_path / "results"
        self.models_path = self.base_path / "models"
        
    def get_latest_portfolio_weights(self) -> pd.DataFrame:
        """Load most recent portfolio weights"""
        weights_file = self.results_path / "latest_portfolio_weights.csv"
        if weights_file.exists():
            return pd.read_csv(weights_file, index_col=0)
        return pd.DataFrame()
        
    def get_model_performance_history(self) -> Dict[str, Any]:
        """Load historical model performance data"""
        performance_files = list(self.results_path.glob("*_results.json"))
        results = {}
        
        for file in performance_files:
            model_name = file.stem.replace("_results", "")
            with open(file, 'r') as f:
                results[model_name] = json.load(f)
                
        return results
        
    def check_model_availability(self) -> Dict[str, bool]:
        """Check which models have trained checkpoints available"""
        return {
            "gat": bool(list(self.models_path.glob("gat_*.pt"))),
            "hrp": bool(list(self.models_path.glob("hrp_*.pkl"))),
            "lstm": bool(list(self.models_path.glob("lstm_*.pt")))
        }
```

### Backend Integration

**Model Integration Layer:**
```python
class ModelIntegrationService:
    """Service layer for integration with existing ML models"""
    
    def __init__(self):
        self.gat_model = None
        self.hrp_model = None
        self.lstm_model = None
        
    def initialize_models(self, config: Dict):
        """Initialize models based on user configuration"""
        if config.get('strategy') == 'gat':
            from src.models.gat.model import GATPortfolioModel
            self.gat_model = GATPortfolioModel(
                constraints=config['constraints'],
                config=config['gat_config']
            )
            
    def run_backtest(self, strategy: str, config: Dict) -> Dict:
        """Run backtest using existing backtesting engine"""
        from src.evaluation.backtest.rolling_engine import RollingBacktestEngine
        
        # Configure and run backtest
        engine = RollingBacktestEngine(config)
        results = engine.run_rolling_backtest([self.get_model(strategy)], data)
        
        return self.format_results_for_ui(results)
        
    def format_results_for_ui(self, results: Dict) -> Dict:
        """Convert backend results to UI-friendly format"""
        return {
            "sharpe_ratio": results.get("sharpe_ratio", 0),
            "annual_return": results.get("annual_return", 0),
            "max_drawdown": results.get("max_drawdown", 0),
            "volatility": results.get("volatility", 0),
            "holdings": results.get("final_weights", {}),
            "performance_history": results.get("cumulative_returns", [])
        }
```

## User Experience Design

### Navigation Architecture

**Multi-Page Application Structure:**
```python
# Main application with page routing
def main():
    st.set_page_config(
        page_title="QuantEdge AI Portfolio Platform",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    PortfolioSessionManager.initialize_session()
    
    # Load custom styling
    load_custom_css()
    
    # Sidebar navigation
    pages = {
        "📊 Portfolio Dashboard": dashboard_page,
        "🎯 Strategy Configuration": strategy_page,
        "📈 Live Monitoring": monitoring_page,
        "🧠 AI Insights": insights_page,
        "📋 Reports": reports_page,
        "⚙️ Settings": settings_page
    }
    
    selected_page = st.sidebar.selectbox("Navigation", list(pages.keys()))
    pages[selected_page]()
```

### Responsive Design Strategy

**CSS Grid Framework:**
```css
/* Responsive grid system for portfolio layouts */
.portfolio-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: white;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border-left: 4px solid var(--primary-color);
}

/* Mobile-first responsive breakpoints */
@media (max-width: 768px) {
    .portfolio-grid {
        grid-template-columns: 1fr;
    }
    
    .metric-card {
        padding: 1rem;
    }
}
```

## Performance Optimization

### Caching Strategy

**Data Caching Architecture:**
```python
# Streamlit caching for performance optimization
@st.cache_data(ttl=300)  # 5-minute cache for portfolio data
def load_portfolio_performance_data():
    """Cache portfolio performance data for better responsiveness"""
    loader = PortfolioDataLoader()
    return loader.get_model_performance_history()

@st.cache_data(ttl=60)   # 1-minute cache for real-time data
def calculate_current_portfolio_value():
    """Cache current portfolio calculations"""
    calculator = PortfolioCalculator()
    return calculator.get_current_value_and_pnl()

# Cache invalidation for data updates
def refresh_portfolio_cache():
    """Force refresh of cached portfolio data"""
    st.cache_data.clear()
    st.rerun()
```

### Lazy Loading Implementation

**Component Loading Strategy:**
```python
class LazyComponentLoader:
    """Load components on demand for better performance"""
    
    @staticmethod
    def load_performance_charts():
        """Load performance charts only when needed"""
        if 'performance_charts_loaded' not in st.session_state:
            from components.analytics.performance_charts import PerformanceChartsComponent
            st.session_state.performance_charts = PerformanceChartsComponent()
            st.session_state.performance_charts_loaded = True
            
        return st.session_state.performance_charts
```

## Security Considerations

### Data Protection

**File System Security:**
- **Read-Only Access**: UI components only read from file system, never modify core data
- **Path Validation**: All file paths validated to prevent directory traversal
- **Error Handling**: Graceful handling of missing or corrupted files without exposing system paths

**Configuration Security:**
```python
class SecureConfigLoader:
    """Secure configuration loading with validation"""
    
    @staticmethod
    def load_portfolio_config(config_path: str) -> Dict:
        """Load configuration with security validation"""
        # Validate file path is within allowed directory
        config_path = Path(config_path).resolve()
        allowed_base = Path("configs").resolve()
        
        if not str(config_path).startswith(str(allowed_base)):
            raise SecurityError("Configuration file outside allowed directory")
            
        # Load and validate configuration structure
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        return validate_config_structure(config)
```

## Testing Strategy

### Component Testing Framework

**Unit Testing Architecture:**
```python
# Test base class for UI components
class BaseComponentTest:
    """Base test class for Streamlit components"""
    
    def setup_method(self):
        """Setup test environment"""
        self.test_data = self.create_test_data()
        self.mock_data_loader = Mock()
        
    def create_test_data(self) -> Dict:
        """Create consistent test data for components"""
        return {
            "portfolio_value": 1000000,
            "daily_change": 0.02,
            "holdings": pd.DataFrame({
                "ticker": ["AAPL", "GOOGL", "MSFT"],
                "weight": [0.1, 0.15, 0.12],
                "return": [0.02, 0.03, 0.01]
            })
        }
        
    def test_component_rendering(self):
        """Test component renders without errors"""
        # Component-specific rendering tests
        pass
```

### Integration Testing

**End-to-End Testing Framework:**
```python
class PortfolioWorkflowTests:
    """Test complete user workflows end-to-end"""
    
    def test_portfolio_creation_workflow(self):
        """Test complete portfolio creation from strategy selection to monitoring"""
        # 1. Strategy selection
        # 2. Configuration setup  
        # 3. Portfolio creation
        # 4. Monitoring dashboard access
        # 5. Performance tracking
        pass
        
    def test_backtest_execution_workflow(self):
        """Test interactive backtesting workflow"""
        # Integration with existing backtesting engine
        pass
```

## Deployment Architecture

### Local Deployment

**Single-User Streamlit Deployment:**
```bash
# Simple local deployment script
#!/bin/bash
cd /path/to/gnn/project

# Check dependencies
python -c "import streamlit, plotly, pandas" || {
    echo "Installing frontend dependencies..."
    pip install streamlit plotly
}

# Launch application
echo "Starting QuantEdge AI Portfolio Platform..."
streamlit run frontend/app.py --server.port 8501 --server.address localhost

echo "Platform available at: http://localhost:8501"
```

### Future Scalability

**Multi-User Architecture (Phase 2):**
```
┌─────────────────────────────────────────┐
│            Load Balancer                │
├─────────────────────────────────────────┤
│        FastAPI Backend Services        │
│  ├── Authentication Service            │
│  ├── Portfolio Management API          │
│  ├── Model Training API                │
│  └── Real-Time Data Service            │
├─────────────────────────────────────────┤
│         Database Layer                  │
│  ├── PostgreSQL (User/Portfolio Data)  │
│  ├── Redis (Caching/Sessions)         │
│  └── File Storage (Models/Results)     │
├─────────────────────────────────────────┤
│         Frontend Options               │
│  ├── React SPA (Advanced Users)       │
│  ├── Streamlit Apps (Quick Analysis)  │
│  └── Mobile App (Monitoring)          │
└─────────────────────────────────────────┘
```

## Change Log

| Date | Version | Description | Author |
|------|---------|-------------|---------|
| 2024-12-14 | 1.0 | Initial frontend architecture specification | James (Full Stack Developer) |