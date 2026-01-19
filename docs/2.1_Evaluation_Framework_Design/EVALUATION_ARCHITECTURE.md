# Evaluation Framework Architecture

## Overview
Technical specification for the Phase 2 evaluation framework for the Bitcoin trading AI system. This document covers data schemas, metrics definitions, and implementation phases.

**Note**: This system trades exclusively BTC/USDT pairs on Coinbase and Kraken exchanges.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Evaluation Framework v2                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Backtest   │───▶│   Metrics    │───▶│   Report     │      │
│  │   Engine     │    │   Calculator │    │   Generator  │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Trade      │    │   Performance│    │   JSON/CSV   │      │
│  │   Records    │    │   Summary    │    │   Export     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Schemas

### TradeRecord Schema
```python
@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    trade_id: str              # Unique identifier
    timestamp: datetime        # Entry time
    symbol: str                # Trading pair (always 'BTC/USDT')
    direction: str             # 'LONG' or 'SHORT'
    entry_price: float         # Actual entry price
    exit_price: float          # Actual exit price
    position_size: float       # Size in base currency
    confidence: float          # AI confidence score (0-10)
    predicted_direction: str   # What AI predicted
    actual_direction: str      # What actually happened
    pnl: float                 # Profit/loss in quote currency
    pnl_percent: float         # Profit/loss as percentage
    holding_period: timedelta  # Time position was held
    regime: str                # Market regime at entry
    fees: float                # Trading fees paid
```

### PredictionRecord Schema
```python
@dataclass
class PredictionRecord:
    """Individual prediction for accuracy tracking"""
    prediction_id: str         # Unique identifier
    timestamp: datetime        # When prediction was made
    symbol: str                # Always 'BTC/USDT'
    predicted_direction: str   # 'UP' or 'DOWN'
    confidence: float          # Confidence score (0-10)
    price_at_prediction: float # Price when prediction made
    target_price: float        # Optional price target
    timeframe: str             # Prediction timeframe
    actual_direction: str      # Filled in after resolution
    actual_price_change: float # Actual % change
    is_correct: bool           # Whether prediction was correct
    regime: str                # Market regime
```

### PerformanceSummary Schema
```python
@dataclass
class PerformanceSummary:
    """Aggregated performance metrics"""
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Core Metrics
    directional_accuracy: float  # % correct predictions
    win_rate: float              # % profitable trades
    profit_factor: float         # gross_profit / gross_loss
    sharpe_ratio: float          # risk-adjusted return

    # Additional Metrics
    total_pnl: float
    average_pnl: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    avg_holding_period: timedelta

    # By Regime
    metrics_by_regime: Dict[str, RegimeMetrics]

    # By Exchange (BTC/USDT is the only asset)
    metrics_by_exchange: Dict[str, ExchangeMetrics]
```

---

## Metrics Definitions

### Core Success Metrics

#### 1. Directional Accuracy
**Target**: >55%
**Formula**:
```
accuracy = correct_predictions / total_predictions * 100
```
**Implementation**:
```python
def calculate_accuracy(predictions: List[PredictionRecord]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p.is_correct)
    return (correct / len(predictions)) * 100
```

#### 2. Win Rate
**Target**: >50%
**Formula**:
```
win_rate = profitable_trades / total_trades * 100
```
**Implementation**:
```python
def calculate_win_rate(trades: List[TradeRecord]) -> float:
    if not trades:
        return 0.0
    winners = sum(1 for t in trades if t.pnl > 0)
    return (winners / len(trades)) * 100
```

#### 3. Profit Factor
**Target**: >1.5
**Formula**:
```
profit_factor = gross_profits / abs(gross_losses)
```
**Implementation**:
```python
def calculate_profit_factor(trades: List[TradeRecord]) -> float:
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return gross_profit / gross_loss
```

#### 4. Sharpe Ratio
**Target**: >1.0
**Formula**:
```
sharpe = (mean_return - risk_free_rate) / std_deviation
```
**Implementation**:
```python
def calculate_sharpe_ratio(
    trades: List[TradeRecord],
    risk_free_rate: float = 0.05,  # Annual
    periods_per_year: int = 252
) -> float:
    if len(trades) < 2:
        return 0.0

    returns = [t.pnl_percent for t in trades]
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0:
        return 0.0

    # Annualize
    annualized_return = mean_return * periods_per_year
    annualized_std = std_return * np.sqrt(periods_per_year)

    return (annualized_return - risk_free_rate) / annualized_std
```

### Secondary Metrics

#### Maximum Drawdown
```python
def calculate_max_drawdown(equity_curve: List[float]) -> Tuple[float, int]:
    """Returns (max_drawdown_percent, duration_in_periods)"""
    peak = equity_curve[0]
    max_dd = 0
    max_dd_duration = 0
    current_dd_start = 0

    for i, equity in enumerate(equity_curve):
        if equity > peak:
            peak = equity
            current_dd_start = i

        drawdown = (peak - equity) / peak
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_duration = i - current_dd_start

    return max_dd * 100, max_dd_duration
```

#### Average Trade Duration
```python
def calculate_avg_duration(trades: List[TradeRecord]) -> timedelta:
    if not trades:
        return timedelta(0)
    total_duration = sum(
        (t.holding_period for t in trades),
        timedelta(0)
    )
    return total_duration / len(trades)
```

---

## Implementation Phases

### Phase 2.1.1 - Data Collection
- Implement TradeRecord and PredictionRecord dataclasses
- Create CSV/JSON export functionality
- Add logging to backtest engine

### Phase 2.1.2 - Metrics Calculator
- Implement MetricsCalculator class
- Add all core metrics calculations
- Create unit tests for each metric

### Phase 2.1.3 - Report Generator
- Create PerformanceSummary aggregation
- Implement breakdown by regime/asset
- Add visualization charts (matplotlib)

### Phase 2.1.4 - Contrarian Testing
- Implement ContrarianTester for inverse signal testing
- Compare AI vs inverse performance
- Statistical significance testing

---

## File Structure

```
scripts/
├── evaluation_framework_v2.py    # Main evaluation module
├── backtest.py                   # Backtest engine (existing)
├── trading_ai.py                 # AI engine (existing)
└── metrics/
    ├── __init__.py
    ├── calculator.py             # MetricsCalculator class
    ├── schemas.py                # Data classes
    └── reporters.py              # Report generation

docs/2.1_Evaluation_Framework_Design/
├── PHASE_2_ROADMAP.md
├── CONTINUATION_STATUS.md
├── EVALUATION_ARCHITECTURE.md    # This file
└── SESSION_NOTES.md
```

---

## API Reference

### MetricsCalculator Class
```python
class MetricsCalculator:
    def __init__(self, trades: List[TradeRecord], predictions: List[PredictionRecord]):
        pass

    def calculate_all(self) -> PerformanceSummary:
        """Calculate all metrics and return summary"""
        pass

    def calculate_accuracy(self) -> float:
        """Calculate directional accuracy"""
        pass

    def calculate_win_rate(self) -> float:
        """Calculate win rate"""
        pass

    def calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        pass

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio"""
        pass

    def breakdown_by_regime(self) -> Dict[str, PerformanceSummary]:
        """Get metrics broken down by market regime"""
        pass

    def export_csv(self, filepath: str) -> None:
        """Export trades to CSV"""
        pass
```

### ContrarianTester Class
```python
class ContrarianTester:
    def __init__(self, predictions: List[PredictionRecord]):
        pass

    def test_inverse_performance(self) -> PerformanceSummary:
        """Calculate performance if we did opposite of AI"""
        pass

    def compare_to_baseline(self) -> Dict[str, float]:
        """Compare AI vs inverse vs random"""
        pass

    def statistical_significance(self) -> float:
        """Calculate p-value for AI outperformance"""
        pass
```

---

## Testing Requirements

### Unit Tests
- Test each metric calculation with known inputs/outputs
- Test edge cases (no trades, all winners, all losers)
- Test floating point precision handling

### Integration Tests
- End-to-end backtest with metrics calculation
- Verify CSV export/import roundtrip
- Test regime breakdown accuracy

### Validation Tests
- Compare metrics against manual calculations
- Verify against external backtesting tools
- Cross-validate with different date ranges
