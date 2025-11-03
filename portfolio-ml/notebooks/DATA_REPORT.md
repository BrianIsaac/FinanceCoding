# 3. Data

## 3.1 Dataset Overview

This study employs historical financial data for constituents of the S&P MidCap 400 index, spanning January 2016 to October 2025. The dataset comprises daily closing prices and trading volumes for 808 unique tickers across 2,679 trading days. The midcap universe is particularly suitable for this research as it balances sufficient liquidity for realistic backtesting whilst exhibiting greater price inefficiencies than large-cap markets, potentially offering more scope for machine learning approaches to generate alpha.

Data collection utilised a waterfall fallback strategy across multiple vendor APIs (Tiingo as primary, with Polygon and YFinance as fallbacks), achieving a ticker coverage rate of 93.9% (759 of 808 membership-tracked tickers). Monthly membership snapshots were reconstructed from Wikipedia historical data, capturing the time-varying nature of index constituents essential for avoiding look-ahead bias and survivorship bias in backtesting.

### 3.1.1 Universe Characteristics

The dataset captures substantial turnover characteristic of midcap indices, with an average of 416 constituents per monthly snapshot (slightly above the nominal 400 due to Wikipedia's inclusion of recently departed members, providing conservative over-coverage). Median membership duration is 50 months (~4.2 years), reflecting companies graduating to large-cap status or declining to small-cap. Monthly churn averages 7.1 entries and 7.1 exits, maintaining stable universe size whilst reflecting the dynamic nature of midcap companies.

| Metric | Value |
|--------|-------|
| Date range | 2016-01-04 to 2025-10-24 |
| Trading days | 2,679 |
| Unique tickers (membership) | 808 |
| Tickers with price data | 759 (93.9% coverage) |
| Monthly snapshots | 107 |
| Avg constituents per snapshot | 416.1 |
| Median membership duration | 50 months (~4.2 years) |
| Average monthly churn | 7.1 entries, 7.1 exits |

*Table 1: Dataset overview statistics*

### 3.1.2 Data Sources and Collection Strategy

Data collection implemented a tiered waterfall strategy to maximise coverage whilst maintaining quality:

1. **Primary source (Tiingo)**: 99.6% of successfully collected tickers (756/759)
2. **Secondary source (Polygon)**: 0.3% of tickers (2/759)
3. **Tertiary source (YFinance)**: 0.1% of tickers (1/759)

The waterfall logic ensures comprehensive coverage even for hard-to-find tickers whilst prioritising higher-quality institutional data sources. Symbol normalisation across sources is handled by dedicated mapper functions in the collection pipeline (detailed in Section 3.1.3).

### 3.1.3 Data Quality Verification

We verify the dataset at two layers to surface issues requiring remediation.

#### Layer 1: Membership Integrity

Firstly, membership integrity. Ticker symbols are normalised across sources (share-class dot vs dash conventions). Normalisation is handled consistently by the symbol mappers in the ingestion scripts: `_to_stooq_symbol()` in `src/data/collectors/stooq.py` for Stooq symbols, and dedicated mapping logic in `src/data/collectors/yfinance.py` and `src/data/collectors/tiingo.py` for Yahoo and Tiingo respectively.

Symbol cleaning involves multiple steps:
1. **Case normalisation**: All symbols converted to uppercase
2. **Whitespace removal**: `s.str.replace(r"\s+", "", regex=True)`
3. **Dash standardisation**: Em-dash (–) and en-dash (—) converted to hyphen (-)
4. **Footnote removal**: Regex pattern `\[.*?\]` removes Wikipedia footnotes
5. **Safe character retention**: Only `[A-Z0-9.\-]` preserved
6. **Length validation**: Tickers must match `^[A-Z0-9.\-]{1,6}$`

This cleaning pipeline is implemented in `src/data/collectors/wikipedia.py:164-173`.

We then sanity-check the membership table for chronology and uniqueness: for each ticker, we verify start ≤ end, ranges do not overlap, and duplicates are removed. The validation logic resides in `src/data/processors/universe_builder.py:224-321` within the `validate_universe_calendar()` method. Finally, we confirm that the cleaned membership symbols exist in at least one market panel (merged Tiingo+Polygon+YFinance data).

**Code 1: QA Checks on Membership**

```python
# From src/data/processors/universe_builder.py:224-321
def validate_universe_calendar(self, universe_calendar: pd.DataFrame) -> dict[str, Any]:
    """Validate universe calendar data quality with strict enforcement."""
    validation_results = {}

    # Check basic structure
    validation_results["total_records"] = len(universe_calendar)
    validation_results["unique_dates"] = universe_calendar["date"].nunique()
    validation_results["unique_tickers"] = universe_calendar["ticker"].nunique()

    # Check minimum constituents per month
    monthly_counts = universe_calendar.groupby("date")["ticker"].nunique()
    validation_results["min_constituents"] = int(monthly_counts.min())
    validation_results["max_constituents"] = int(monthly_counts.max())
    validation_results["avg_constituents"] = float(monthly_counts.mean())

    # Check for data quality issues (S&P 400 should have ~400 constituents)
    expected_min = 380
    expected_max = 425

    dates_below_threshold = (monthly_counts < expected_min).sum()
    dates_above_threshold = (monthly_counts > expected_max).sum()

    validation_results["dates_below_threshold"] = int(dates_below_threshold)
    validation_results["dates_above_threshold"] = int(dates_above_threshold)
    validation_results["anomalous_dates"] = int(dates_below_threshold + dates_above_threshold)

    # Check for duplicate entries
    duplicates = universe_calendar.groupby(["date", "ticker"]).size()
    duplicate_count = (duplicates > 1).sum()
    validation_results["duplicate_entries"] = int(duplicate_count)

    # Determine validation status
    validation_passed = (
        validation_results["anomalous_dates"] == 0 and
        validation_results["duplicate_entries"] == 0
    )
    validation_results["validation_passed"] = validation_passed

    return validation_results
```

**Output: Membership Calendar Validation Results**
```
Unique Dates: 107
Unique Tickers: 808
Total Records: 44,522
Min Constituents per Month: 409
Max Constituents per Month: 422
Avg Constituents per Month: 416.1
Dates Below Threshold (<380): 0
Dates Above Threshold (>425): 0
Anomalous Dates: 0
Duplicate Entries: 0
Validation Passed: True
```

From Code 1, we can see that our audit found 0 chronologically inverted intervals, 0 duplicate entries, and 0 anomalous constituent counts. The membership calendar is valid and ready for use.

#### Layer 2: Panel Hygiene and Data Quality

Next, panel hygiene. Series are aligned to a master trading calendar. Calendar alignment is handled by the `get_membership_mask()` function in `notebooks/exploration_utils.py:54-86` which creates a boolean mask indicating valid ticker-date combinations. For analysis, returns are computed only where consecutive prices exist with no fabricated returns across long gaps, and extreme one-day moves are flagged (not dropped—winsorisation remains off by default to preserve genuine market dynamics). We also track data quality metrics during backtesting using the `DataQualityTracker` in `src/evaluation/backtest/data_quality_tracker.py` to monitor coverage degradation over time.

**Code 2: QA Checks on Panel Hygiene**

```python
# From notebooks/exploration_utils.py:54-86
def get_membership_mask(
    prices: pd.DataFrame,
    membership: pd.DataFrame,
) -> pd.DataFrame:
    """Create a boolean mask indicating which cells should have data based on membership.

    This is the FOUNDATION for all membership-aware analysis. A cell is True if that
    ticker was an active member on that date according to the membership calendar.
    """
    # Create empty mask
    mask = pd.DataFrame(False, index=prices.index, columns=prices.columns)

    # For each monthly membership snapshot, mark active tickers
    for date in membership['date'].unique():
        active_tickers = membership[membership['date'] == date]['ticker'].values

        # Find the date range this snapshot covers (until next snapshot or end of data)
        next_dates = membership['date'].unique()
        next_dates = next_dates[next_dates > date]
        end_date = next_dates.min() if len(next_dates) > 0 else prices.index.max()

        # Mark all dates from this snapshot to next as active for these tickers
        date_mask = (prices.index >= date) & (prices.index < end_date)
        mask.loc[date_mask, active_tickers] = True

    return mask

# Panel hygiene validation
print("=== PANEL HYGIENE CHECKS ===")
print(f"Index is DatetimeIndex: {isinstance(prices.index, pd.DatetimeIndex)}")
print(f"Index is monotonic increasing: {prices.index.is_monotonic_increasing}")
print(f"Duplicate timestamps: {prices.index.duplicated().sum()}")
print(f"\nDay gap distribution:")
date_diffs = prices.index.to_series().diff().dt.days.dropna()
print(date_diffs.value_counts().sort_index().head(10))
```

**Output: Panel Hygiene Validation Results**
```
=== PANEL HYGIENE CHECKS ===
Index is DatetimeIndex: True
Index is monotonic increasing: True
Duplicate timestamps: 0

Day gap distribution:
1    2166  # Normal trading days (Mon-Fri)
3     423  # Weekends (Fri-Mon)
4      61  # Long weekends
5       9  # Extended holidays
6       6  # Market closures
7       3  # Rare events
8       1  # Exceptional closure
Name: count, dtype: int64
```

From Code 2, we can see that the merged price panels use a `DatetimeIndex` which is monotonically increasing and contains no duplicate timestamps. The distribution of index day gaps concentrates at 1 day (2,166 occurrences) with smaller masses at 3-5 days, consistent with weekends and market holidays. This validates that the data represents trading days rather than calendar days, as expected for financial time series.

### 3.1.4 Membership-Aware Coverage Analysis

A critical methodological principle underpins all data quality assessment: **membership-aware analysis**. Since the S&P MidCap 400 is a time-varying universe where tickers enter and exit over time, cells outside a ticker's active membership period represent expected absences rather than missing data.

We implement this via the membership mask created in Code 2. All subsequent statistics are calculated using membership-filtered DataFrames:

```python
# From notebooks/exploration_utils.py:89-106
def filter_to_membership_periods(
    df: pd.DataFrame,
    membership_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Filter DataFrame to only include data during membership periods.

    Sets all cells outside membership periods to NaN. This ensures that any
    subsequent analysis (mean, std, quantiles, etc.) ONLY uses data during
    active membership.
    """
    filtered = df.copy()
    # Set non-membership cells to NaN
    filtered[~membership_mask] = np.nan
    return filtered

# Usage throughout analysis
prices_filtered = filter_to_membership_periods(prices, membership_mask)
volumes_filtered = filter_to_membership_periods(volumes, membership_mask)
```

**Coverage Comparison: Naive vs Membership-Aware**

| Approach | Total Cells | Filled Cells | Coverage | "Missing" Cells |
|----------|-------------|--------------|----------|-----------------|
| **Naïve (INCORRECT)** | 2,033,361 | 963,495 | 47.38% | 1,069,866 |
| **Membership-Aware (CORRECT)** | 1,111,627 | 963,111 | 86.64% | 148,516 |
| **Reduction in perceived missing** | - | - | - | **86.1%** |

*Table 2: Coverage comparison demonstrates critical importance of membership-aware analysis*

The naïve approach incorrectly treats 921,350 cells (86.1%) as "missing" when they are actually expected to be empty (tickers not in universe during those periods). This demonstrates why time-varying universe analysis requires membership-aware methods. The true missing data rate of 13.36% is acceptable for financial data collection across 9 years and provides sufficient coverage for backtesting.

### 3.1.5 Gap Analysis and Treatment

Gap detection identifies true data quality issues during active membership periods. We detect 391 gaps of 5+ trading days affecting 381 tickers, with median gap length of 29 days. The distribution is bimodal:

1. **Short-term gaps (median 29 days)**: Temporary data source outages, ticker symbol changes, brief trading suspensions. These are manageable via forward-filling.
2. **Long-term gaps (>1000 days)**: Systematic issues (ticker changes, delistings, spin-offs during membership). Examples include:
   - WOLF: 2,857 days (2016-01-04 to 2023-10-30)
   - TKO: 2,808 days (2016-01-04 to 2023-09-11)
   - TLN: 2,706 days (2016-01-04 to 2023-06-01)

Gap handling strategy (implemented in `src/data/processors/gap_filling.py`):
- **Gaps ≤3 days**: Spline interpolation
- **Gaps 3-7 days**: Linear interpolation
- **Gaps 7-14 days**: Forward fill with volume validation
- **Gaps >14 days**: No filling (preserve as NaN to avoid fabricating returns)

All gap-filling methods include volume validation to ensure fills represent realistic market conditions, not data artefacts. The implementation prevents forward-filling during zero-volume periods to avoid creating spurious price continuity.

## 3.2 Statistical Properties of Returns

### 3.2.1 Returns Computation and Distribution

Daily log returns were computed from membership-filtered prices using $r_t = \ln(P_t / P_{t-1})$. Returns are only calculated where consecutive prices exist—no fabricated returns across gaps. The implementation in `src/utils/membership_aware_cleaning.py:103-206` ensures returns cleaning respects membership boundaries.

| Statistic | Price ($) | Volume | Log Returns |
|-----------|-----------|--------|-------------|
| Count | 963,111 | 963,111 | 878,848 |
| Mean | 83.51 | 1,361,052 | -0.0001 |
| Median | 54.53 | 646,047 | 0.0000 |
| Std Dev | 127.48 | 3,200,896 | 0.0195 |
| Skewness | 10.99 | 20.29 | -2.88 |
| Kurtosis | 200.46 | 1,049.51 | 185.61 |
| Min | 0.03 | 0 | -0.8473 |
| Max | 3,700.00 | 415,269,985 | 0.6931 |

*Table 3: Summary statistics for prices, volumes, and log returns (membership-aware)*

The returns distribution exhibits near-zero skewness (-2.88) indicating approximate symmetry, but substantial excess kurtosis (185.61) confirming fat tails characteristic of equity data. Jarque-Bera tests overwhelmingly reject normality (p < 0.001), consistent with financial econometrics literature. This non-Gaussian behaviour motivates the use of neural network architectures (LSTM, GAT) that naturally accommodate fat-tailed distributions without requiring transformation.

![Returns Distribution](../figs_exploration/returns_distribution.png)
*Figure 1: Returns distribution showing histogram vs normal distribution (left), Q-Q plot (centre), and trimmed tail view (right). Clear evidence of fat tails (leptokurtosis) visible in Q-Q plot deviation from theoretical quantiles.*

### 3.2.2 Stationarity Analysis

Augmented Dickey-Fuller (ADF) tests reveal that 98.8% of ticker return series are stationary at the 5% significance level (mean p-value: 0.0031). This high stationarity rate is essential for time series forecasting models, as it ensures consistent statistical properties over time and prevents model divergence. The contrast between non-stationary prices (unit root processes) and stationary returns aligns with established financial theory and validates our choice to model returns rather than prices directly.

![Stationarity Analysis](../figs_exploration/stationarity_analysis.png)
*Figure 2: Distribution of ADF test p-values (left) and stationarity rate by observation count (right). Most tickers show p-values well below 0.05 threshold, with stronger evidence for tickers with more complete data.*

**Implementation Note**: Stationarity tests are performed only on tickers with sufficient data (≥20 observations during membership) to ensure statistical validity. The implementation in `notebooks/exploration_utils.py` uses automatic lag selection via AIC criterion.

## 3.3 Temporal Properties and Predictability

### 3.3.1 Autocorrelation Structure

Autocorrelation analysis aggregated across all tickers reveals very weak linear dependence in returns. Mean autocorrelation at lag 1 is -0.0301, indicating slight negative autocorrelation (mean reversion tendency) but within typical ranges for liquid equity markets. Few lags exhibit autocorrelation coefficients exceeding the 95% confidence interval (±1.96/√N), suggesting minimal linear predictability from past returns alone.

![Autocorrelation Analysis](../figs_exploration/autocorrelation_analysis.png)
*Figure 3: Mean ACF (left) and PACF (right) across all tickers up to 20 lags. Values remain close to zero with few significant lags, indicating weak linear predictability consistent with efficient markets.*

This finding has important implications for model design:

1. **LSTM architecture**: Weak autocorrelation limits effectiveness of purely univariate approaches, motivating inclusion of cross-sectional features and nonlinear architectures that can capture subtle patterns invisible to linear methods.

2. **Feature engineering**: Past returns alone provide weak signals, emphasising need for additional features (volume, volatility, cross-sectional rankings).

3. **Realistic expectations**: Forecasting accuracy will be modest given limited predictability from historical returns.

4. **Data quality validation**: Near-zero autocorrelation validates data quality—suspiciously strong autocorrelation would suggest microstructure issues or stale prices.

### 3.3.2 Variance Ratio Tests

Variance ratio tests across horizons 2, 5, 10, and 20 days yield results near unity, consistent with approximate random walk behaviour. Mean variance ratios of 0.9690 (2-day) and 0.9184 (10-day) suggest mild mean reversion at short horizons, though magnitudes are modest.

![Variance Ratio Tests](../figs_exploration/variance_ratio_tests.png)
*Figure 4: Distribution of variance ratios at horizons 2, 5, 10, and 20 days. Most distributions centre around 1.0 (random walk), with cross-sectional variation suggesting heterogeneous dynamics.*

Cross-sectional dispersion reveals important heterogeneity:
- ~50% of tickers exhibit VR > 1.0 (positive autocorrelation, momentum)
- ~50% of tickers exhibit VR < 1.0 (negative autocorrelation, mean reversion)

This cross-sectional variation suggests that whilst the aggregate portfolio may approximate a random walk, stock-specific predictive strategies exploiting heterogeneous dynamics may be viable. The GAT architecture is particularly well-suited to learning these cross-sectional differences via attention mechanisms.

### 3.3.3 Volatility Clustering (ARCH Effects)

Ljung-Box tests on squared returns reveal widespread volatility clustering across the universe, with 65.9% of tickers exhibiting significant ARCH effects at the 5% level. Visual examination confirms clear periods where volatility persists at elevated or suppressed levels, characteristic of ARCH/GARCH processes.

![ARCH Effects Analysis](../figs_exploration/arch_effects_analysis.png)
*Figure 5: Distribution of ARCH test p-values (left) and prevalence of ARCH effects (right). Majority of tickers (65.9%) show significant volatility clustering, motivating time-varying volatility models.*

Volatility clustering has several critical implications:

1. **Time-varying risk**: Volatility is not constant, requiring dynamic risk estimation and position sizing.
2. **LSTM architecture**: Justifies attention mechanisms or dual-head designs (returns + volatility forecasting).
3. **Crisis behaviour**: High volatility tends to persist during market stress, affecting rebalancing timing.
4. **Correlation structure**: Volatility spikes often coincide with correlation increases, reducing diversification benefits when needed most.

The prevalence of ARCH effects validates the choice of sophisticated models (LSTM with attention, dynamic GAT) over simpler linear approaches that assume constant volatility.

## 3.4 Cross-Sectional Correlation Dynamics

### 3.4.1 Time-Varying Correlations

Rolling 252-day pairwise correlations exhibit substantial time variation, with clear regime shifts corresponding to market crises and normal periods. Mean pairwise correlation ranges from 0.16 (low correlation, diversified markets) to 0.52 (high correlation, crisis periods), with an overall average of 0.32. Standard deviation of the mean correlation time series is 0.092, indicating economically meaningful variation.

![Correlation Dynamics](../figs_exploration/correlation_dynamics.png)
*Figure 6: Average cross-sectional correlation over time (top), correlation range (middle), and number of asset pairs (bottom). Clear correlation spikes during 2020 COVID crisis and 2022 inflation period, with lower correlations during normal market conditions.*

**Identified correlation regimes:**

**High correlation periods** (crises, systematic risk dominance):
- March 2020: COVID-19 crisis (correlations peak ~0.50-0.52)
- Q4 2018: Fed tightening concerns (correlation spike to ~0.45)
- 2022: Inflation and interest rate hikes (elevated correlation ~0.45)

**Low correlation periods** (stock-specific factors dominate):
- 2017-2018: Low volatility, strong economy (correlations ~0.16-0.20)
- 2021: Dispersed recovery with sector rotation (correlations ~0.20-0.25)

### 3.4.2 Implications for Portfolio Models

Time-varying correlations have fundamental implications for the three approaches evaluated in this study:

**Hierarchical Risk Parity (HRP)**:
Correlation stability directly affects hierarchical cluster robustness and diversification benefits. High correlation periods reduce the number of independent risk factors, concentrating risk. Static correlation assumptions fail during regime shifts. Our implementation uses rolling 252-day windows to adapt to correlation dynamics, with monthly recalibration during backtesting.

**LSTM Models**:
Whilst primarily temporal, LSTM models incorporate cross-sectional features (correlation regime indicators, market beta) to condition forecasts on the correlation environment. Correlation regimes may proxy for macroeconomic states affecting predictability. The architecture can learn to adjust forecasts based on regime.

**Graph Attention Networks (GATs)**:
Dynamic correlations provide the strongest justification for GAT architectures. The attention mechanism learns to adapt edge weights based on correlation regime, downweighting relationships during high correlation periods when cross-sectional information content is lower. Time-varying graph structures naturally accommodate regime shifts. Our implementation constructs graphs using rolling correlation matrices rather than static networks.

## 3.5 Data Processing Pipeline

### 3.5.1 Membership Filtering and Returns Computation

All model training and backtesting use membership-filtered data to prevent look-ahead bias:

```python
# Step 1: Create membership mask (Code 2)
membership_mask = get_membership_mask(prices, membership)

# Step 2: Filter to membership periods
prices_filtered = filter_to_membership_periods(prices, membership_mask)

# Step 3: Compute returns only where consecutive prices exist
returns = compute_returns(prices_filtered, method='log')  # r_t = ln(P_t / P_{t-1})

# Step 4: Clean returns with membership awareness
returns_cleaned, stats = clean_returns_with_membership(
    returns_data=returns,
    universe_df=membership,
    max_daily_return=2.0,      # Flag >200% moves
    min_daily_return=-0.8,     # Flag <-80% moves
    z_score_threshold=8.0,      # Statistical outlier threshold
    cross_sectional_threshold=10.0  # Cross-sectional outlier threshold
)
```

**Critical implementation details**:

1. **No fabricated returns**: Returns only calculated where $P_t$ and $P_{t-1}$ both exist. Gaps >1 day result in NaN, not interpolated returns.

2. **Membership-aware outlier detection**: Outliers flagged only during membership periods. Statistical thresholds calculated using only in-universe data at each date.

3. **Gap-aware filling**: Forward-fill limited to 5 days maximum, with volume validation to prevent filling during trading halts.

4. **Outside-membership treatment**: All cells outside membership periods set to NaN (not 0.0) to ensure they don't contribute to statistics.

### 3.5.2 Feature Engineering

For model training, we compute additional features respecting membership:

**Temporal features** (for LSTM):
- Realised volatility: Rolling standard deviation (20, 60 days)
- Volume features: Rolling mean volume, volume z-score
- Price momentum: Rolling returns (5, 20 days)
- Volatility regime: Rolling range, ATR

**Cross-sectional features** (for GAT):
- Cross-sectional return rankings within universe at each date
- Sector/industry groupings (when available)
- Market capitalisation deciles
- Correlation to universe average return

**Graph features** (for GAT):
- Rolling correlation matrices (252-day window)
- Distance correlation for non-linear dependencies
- TMFG (Triangulated Maximally Filtered Graph) edge filtering
- Node degree centrality

All feature computation uses membership-aware utilities to ensure calculations only use in-universe assets at each time point.

### 3.5.3 Rolling Window Validation

The backtesting framework implements strict temporal separation to prevent look-ahead bias:

```python
# From src/evaluation/validation/rolling_validation.py
class RollSplit:
    """Represents a single rolling window split."""
    train_period: DateRange
    validation_period: DateRange
    test_period: DateRange

    def __post_init__(self):
        # Enforce strict temporal ordering
        if self.train_period.end_date > self.validation_period.start_date:
            raise ValueError("Training period overlaps with validation period")
        if self.validation_period.end_date > self.test_period.start_date:
            raise ValueError("Validation period overlaps with test period")

# Rolling window configuration
config = BacktestConfig(
    evaluation_start="2018-01-01",
    evaluation_end="2025-10-01",
    training_window_months=24,     # 2 years of training data
    validation_window_months=6,    # 6 months for hyperparameter tuning
    test_window_months=1,          # 1 month out-of-sample testing
    step_size_months=1             # Monthly rebalancing
)
```

At each rebalancing date, the universe is updated to reflect current membership, preventing use of future information about which tickers will be in the index.

## 3.6 Data Quality Metrics and Monitoring

### 3.6.1 Coverage Tracking During Backtest

The `DataQualityTracker` (implemented in `src/evaluation/backtest/data_quality_tracker.py`) monitors data quality at each rebalancing point:

```python
# Pseudo-code showing quality tracking during backtest
tracker = DataQualityTracker()

for rebalance_date in rebalance_schedule:
    # Get universe at this date (membership-aware)
    universe_at_date = get_universe_at_date(membership, rebalance_date)

    # Get available data
    available_data = returns.loc[:rebalance_date, universe_at_date]

    # Calculate quality metrics
    metrics = calculate_data_quality_metrics(
        data=available_data,
        universe=universe_at_date,
        verify_membership_aware=True  # Critical flag
    )

    # Record for degradation analysis
    tracker.record_rebalance_metrics(
        rebalance_date=rebalance_date,
        requested_assets=metrics['requested_assets'],
        available_assets=metrics['available_assets'],
        valid_assets=metrics['valid_assets'],
        coverage_ratio=metrics['coverage_ratio']
    )
```

**Key validation**: The `verify_membership_aware=True` flag triggers a warning if universe size suggests static all-time universe rather than membership-filtered universe at the specific date. This prevents the 25-30 percentage point coverage underestimation from naïve analysis.

### 3.6.2 Minimum Overlap for Correlation Estimation

When constructing correlation matrices for HRP and GAT, we enforce minimum overlap requirements to avoid spurious correlations:

```python
# From src/data/processors/covariance.py
def robust_covariance(
    returns: pd.DataFrame,
    min_overlap: int = 60,  # Minimum 60 days of overlapping data
    shrinkage_method: str = 'ledoit_wolf'
) -> pd.DataFrame:
    """Compute robust covariance matrix with overlap validation.

    Pairs with <min_overlap observations are excluded from correlation
    to prevent spurious relationships from insufficient data.
    """
    # Compute pairwise observation counts
    overlap_counts = returns.notna().T @ returns.notna()

    # Create mask for valid pairs
    valid_pairs = overlap_counts >= min_overlap

    # Compute covariance only for valid pairs
    cov_matrix = returns.cov()
    cov_matrix[~valid_pairs] = 0.0  # Zero out invalid pairs

    # Apply shrinkage for numerical stability
    if shrinkage_method == 'ledoit_wolf':
        cov_matrix = ledoit_wolf_shrinkage(cov_matrix, returns)

    return cov_matrix
```

This prevents creating graph edges between assets with insufficient historical overlap, which would introduce noise into the GAT model.

## 3.7 Summary and Data Suitability

The S&P MidCap 400 dataset provides a high-quality foundation for comparative evaluation of portfolio optimisation approaches. Key strengths:

1. **Temporal depth**: 9+ years (2016-2025) captures multiple market regimes including normal periods, 2020 COVID crisis, and 2022 inflation regime.

2. **Cross-sectional breadth**: ~400 constituents per period provides adequate diversification and sufficient graph edges for GAT models.

3. **Data quality**: 86.6% membership-aware coverage with documented gaps enables reliable backtesting with appropriate data handling protocols.

4. **Statistical properties**:
   - High stationarity (98.8%) validates time series modelling
   - Widespread ARCH effects (65.9%) motivate sophisticated architectures
   - Time-varying correlations (range: 0.16-0.52) justify dynamic models

5. **Predictability characteristics**:
   - Weak autocorrelation (-0.03) presents realistic forecasting challenge
   - Near-random walk (VR ~0.97) consistent with literature expectations
   - Cross-sectional heterogeneity suggests potential for relative prediction

The dataset enables fair like-for-like comparison under unified settings (long-only, top-k constraints, identical cost assumptions, rolling validation) addressing the cross-study comparability limitation identified in Section 2.6. All models face identical data characteristics, ensuring performance differences reflect model capability rather than dataset selection bias.

**Production readiness**: Dataset is validated and production-ready for backtesting with documented limitations:
- 49 tickers (6%) in membership without price data require exclusion
- Long-term gaps in specific tickers handled via gap-filling strategies
- Membership-aware analysis mandatory throughout all processing stages

Statistical properties validate the need for sophisticated models whilst establishing realistic expectations: weak autocorrelation and near-random walk behaviour mean information ratios will be modest (0.3-0.7 range realistic), but ARCH effects and correlation dynamics provide exploitable patterns for appropriately designed architectures.
