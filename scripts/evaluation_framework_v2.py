"""
Evaluation Framework v2
Phase 2.1 - Extended Validation

This module provides data classes and utilities for evaluating
the Bitcoin trading AI system's performance.

Note: This system trades exclusively BTC/USDT pairs on Coinbase and Kraken exchanges.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import csv
from pathlib import Path

# Optional imports for numerical calculations
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class TradeRecord:
    """Individual trade record for analysis"""
    trade_id: str
    timestamp: datetime
    symbol: str
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    exit_price: float
    position_size: float
    confidence: float
    predicted_direction: str
    actual_direction: str
    pnl: float
    pnl_percent: float
    holding_period: timedelta
    regime: str = "unknown"
    fees: float = 0.0

    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.pnl > 0

    def is_correct_prediction(self) -> bool:
        """Check if prediction direction was correct"""
        return self.predicted_direction == self.actual_direction

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'position_size': self.position_size,
            'confidence': self.confidence,
            'predicted_direction': self.predicted_direction,
            'actual_direction': self.actual_direction,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'holding_period_seconds': self.holding_period.total_seconds(),
            'regime': self.regime,
            'fees': self.fees
        }


@dataclass
class PredictionRecord:
    """Individual prediction for accuracy tracking"""
    prediction_id: str
    timestamp: datetime
    symbol: str
    predicted_direction: str  # 'UP' or 'DOWN'
    confidence: float
    price_at_prediction: float
    target_price: Optional[float] = None
    timeframe: str = "24h"
    actual_direction: Optional[str] = None
    actual_price_change: Optional[float] = None
    is_correct: Optional[bool] = None
    regime: str = "unknown"

    def resolve(self, actual_price: float) -> None:
        """Resolve prediction with actual outcome"""
        self.actual_price_change = (actual_price - self.price_at_prediction) / self.price_at_prediction * 100
        self.actual_direction = 'UP' if actual_price > self.price_at_prediction else 'DOWN'
        self.is_correct = self.predicted_direction == self.actual_direction

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'prediction_id': self.prediction_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'predicted_direction': self.predicted_direction,
            'confidence': self.confidence,
            'price_at_prediction': self.price_at_prediction,
            'target_price': self.target_price,
            'timeframe': self.timeframe,
            'actual_direction': self.actual_direction,
            'actual_price_change': self.actual_price_change,
            'is_correct': self.is_correct,
            'regime': self.regime
        }


@dataclass
class PerformanceSummary:
    """Aggregated performance metrics"""
    period_start: datetime
    period_end: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int

    # Core Metrics
    directional_accuracy: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float

    # Additional Metrics
    total_pnl: float
    average_pnl: float
    max_drawdown: float
    max_drawdown_duration: timedelta
    avg_holding_period: timedelta

    # Breakdowns
    metrics_by_regime: Dict[str, dict] = field(default_factory=dict)
    metrics_by_exchange: Dict[str, dict] = field(default_factory=dict)

    def passes_criteria(self) -> Dict[str, bool]:
        """Check if metrics pass success criteria"""
        return {
            'directional_accuracy': self.directional_accuracy > 55.0,
            'win_rate': self.win_rate > 50.0,
            'profit_factor': self.profit_factor > 1.5,
            'sharpe_ratio': self.sharpe_ratio > 1.0
        }

    def all_criteria_passed(self) -> bool:
        """Check if all success criteria are met"""
        return all(self.passes_criteria().values())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'period_start': self.period_start.isoformat(),
            'period_end': self.period_end.isoformat(),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'directional_accuracy': self.directional_accuracy,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'sharpe_ratio': self.sharpe_ratio,
            'total_pnl': self.total_pnl,
            'average_pnl': self.average_pnl,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_duration_seconds': self.max_drawdown_duration.total_seconds(),
            'avg_holding_period_seconds': self.avg_holding_period.total_seconds(),
            'criteria_passed': self.passes_criteria(),
            'all_passed': self.all_criteria_passed()
        }


# =============================================================================
# Metrics Calculator
# =============================================================================

class MetricsCalculator:
    """
    Calculate performance metrics from trade and prediction records.

    Success Criteria:
    - Directional Accuracy: >55%
    - Win Rate: >50%
    - Profit Factor: >1.5
    - Sharpe Ratio: >1.0
    """

    def __init__(
        self,
        trades: Optional[List[TradeRecord]] = None,
        predictions: Optional[List[PredictionRecord]] = None
    ):
        self.trades = trades or []
        self.predictions = predictions or []

    def calculate_accuracy(self) -> float:
        """
        Calculate directional accuracy.

        Returns:
            Percentage of correct predictions (0-100)
        """
        resolved = [p for p in self.predictions if p.is_correct is not None]
        if not resolved:
            return 0.0
        correct = sum(1 for p in resolved if p.is_correct)
        return (correct / len(resolved)) * 100

    def calculate_win_rate(self) -> float:
        """
        Calculate win rate from trades.

        Returns:
            Percentage of profitable trades (0-100)
        """
        if not self.trades:
            return 0.0
        winners = sum(1 for t in self.trades if t.is_winner())
        return (winners / len(self.trades)) * 100

    def calculate_profit_factor(self) -> float:
        """
        Calculate profit factor (gross profits / gross losses).

        Returns:
            Profit factor ratio (>1 is profitable)
        """
        if not self.trades:
            return 0.0

        gross_profit = sum(t.pnl for t in self.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.trades if t.pnl < 0))

        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def calculate_sharpe_ratio(
        self,
        risk_free_rate: float = 0.05,
        periods_per_year: int = 252
    ) -> float:
        """
        Calculate Sharpe ratio (risk-adjusted return).

        Args:
            risk_free_rate: Annual risk-free rate (default 5%)
            periods_per_year: Trading periods per year (default 252 for daily)

        Returns:
            Annualized Sharpe ratio
        """
        if len(self.trades) < 2:
            return 0.0

        returns = [t.pnl_percent for t in self.trades]

        if HAS_NUMPY:
            mean_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
        else:
            mean_return = sum(returns) / len(returns)
            variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
            std_return = variance ** 0.5

        if std_return == 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * (periods_per_year ** 0.5)

        return (annualized_return - risk_free_rate) / annualized_std

    def calculate_max_drawdown(self) -> Tuple[float, timedelta]:
        """
        Calculate maximum drawdown from equity curve.

        Returns:
            Tuple of (max_drawdown_percent, duration)
        """
        if not self.trades:
            return 0.0, timedelta(0)

        # Build equity curve
        equity = 0.0
        equity_curve = []
        timestamps = []

        for trade in sorted(self.trades, key=lambda t: t.timestamp):
            equity += trade.pnl
            equity_curve.append(equity)
            timestamps.append(trade.timestamp)

        if not equity_curve:
            return 0.0, timedelta(0)

        # Find max drawdown
        peak = equity_curve[0]
        max_dd = 0.0
        peak_idx = 0
        max_dd_duration = timedelta(0)

        for i, equity in enumerate(equity_curve):
            if equity > peak:
                peak = equity
                peak_idx = i

            if peak > 0:
                drawdown = (peak - equity) / peak
                if drawdown > max_dd:
                    max_dd = drawdown
                    if i > peak_idx:
                        max_dd_duration = timestamps[i] - timestamps[peak_idx]

        return max_dd * 100, max_dd_duration

    def calculate_avg_holding_period(self) -> timedelta:
        """Calculate average trade holding period."""
        if not self.trades:
            return timedelta(0)

        total_duration = sum(
            (t.holding_period for t in self.trades),
            timedelta(0)
        )
        return total_duration / len(self.trades)

    def calculate_all(self) -> PerformanceSummary:
        """
        Calculate all metrics and return a PerformanceSummary.

        Returns:
            PerformanceSummary with all calculated metrics
        """
        max_dd, max_dd_duration = self.calculate_max_drawdown()

        # Get date range
        if self.trades:
            sorted_trades = sorted(self.trades, key=lambda t: t.timestamp)
            period_start = sorted_trades[0].timestamp
            period_end = sorted_trades[-1].timestamp
        elif self.predictions:
            sorted_preds = sorted(self.predictions, key=lambda p: p.timestamp)
            period_start = sorted_preds[0].timestamp
            period_end = sorted_preds[-1].timestamp
        else:
            period_start = period_end = datetime.now()

        return PerformanceSummary(
            period_start=period_start,
            period_end=period_end,
            total_trades=len(self.trades),
            winning_trades=sum(1 for t in self.trades if t.is_winner()),
            losing_trades=sum(1 for t in self.trades if not t.is_winner()),
            directional_accuracy=self.calculate_accuracy(),
            win_rate=self.calculate_win_rate(),
            profit_factor=self.calculate_profit_factor(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            total_pnl=sum(t.pnl for t in self.trades),
            average_pnl=sum(t.pnl for t in self.trades) / len(self.trades) if self.trades else 0.0,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            avg_holding_period=self.calculate_avg_holding_period(),
            metrics_by_regime=self.breakdown_by_regime(),
            metrics_by_exchange=self.breakdown_by_exchange()
        )

    def breakdown_by_regime(self) -> Dict[str, dict]:
        """Get metrics broken down by market regime."""
        regimes: Dict[str, List[TradeRecord]] = {}

        for trade in self.trades:
            regime = trade.regime or 'unknown'
            if regime not in regimes:
                regimes[regime] = []
            regimes[regime].append(trade)

        results = {}
        for regime, trades in regimes.items():
            calc = MetricsCalculator(trades=trades)
            results[regime] = {
                'total_trades': len(trades),
                'win_rate': calc.calculate_win_rate(),
                'profit_factor': calc.calculate_profit_factor(),
                'total_pnl': sum(t.pnl for t in trades)
            }

        return results

    def breakdown_by_exchange(self) -> Dict[str, dict]:
        """Get metrics broken down by exchange (Coinbase, Kraken)."""
        exchanges: Dict[str, List[TradeRecord]] = {}

        for trade in self.trades:
            # Extract exchange from trade_id or use default
            # Trade IDs are formatted as "{exchange}_{id}" e.g., "coinbase_T001"
            exchange = trade.trade_id.split('_')[0] if '_' in trade.trade_id else 'unknown'
            if exchange not in exchanges:
                exchanges[exchange] = []
            exchanges[exchange].append(trade)

        results = {}
        for exchange, trades in exchanges.items():
            calc = MetricsCalculator(trades=trades)
            results[exchange] = {
                'total_trades': len(trades),
                'win_rate': calc.calculate_win_rate(),
                'profit_factor': calc.calculate_profit_factor(),
                'total_pnl': sum(t.pnl for t in trades)
            }

        return results

    def export_trades_csv(self, filepath: str) -> None:
        """Export trades to CSV file."""
        if not self.trades:
            return

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.trades[0].to_dict().keys())
            writer.writeheader()
            for trade in self.trades:
                writer.writerow(trade.to_dict())

    def export_summary_json(self, filepath: str) -> None:
        """Export performance summary to JSON file."""
        summary = self.calculate_all()
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)


# =============================================================================
# Contrarian Tester
# =============================================================================

class ContrarianTester:
    """
    Test performance of inverse signals (contrarian strategy).

    This helps validate that the AI is actually providing edge,
    not just random noise that happens to be profitable.
    """

    def __init__(self, predictions: List[PredictionRecord]):
        self.predictions = predictions

    def test_inverse_performance(self) -> dict:
        """
        Calculate what performance would be if we did opposite of AI.

        Returns:
            Dictionary with inverse performance metrics
        """
        resolved = [p for p in self.predictions if p.is_correct is not None]
        if not resolved:
            return {'accuracy': 0.0, 'sample_size': 0}

        # Inverse accuracy = incorrect predictions
        inverse_correct = sum(1 for p in resolved if not p.is_correct)
        inverse_accuracy = (inverse_correct / len(resolved)) * 100

        return {
            'inverse_accuracy': inverse_accuracy,
            'ai_accuracy': 100 - inverse_accuracy,
            'sample_size': len(resolved),
            'ai_outperforms': inverse_accuracy < 50  # AI beats inverse if inverse < 50%
        }

    def compare_to_random(self) -> dict:
        """
        Compare AI performance to random baseline (50%).

        Returns:
            Dictionary with comparison results
        """
        resolved = [p for p in self.predictions if p.is_correct is not None]
        if not resolved:
            return {'edge_over_random': 0.0}

        ai_accuracy = sum(1 for p in resolved if p.is_correct) / len(resolved) * 100
        random_baseline = 50.0

        return {
            'ai_accuracy': ai_accuracy,
            'random_baseline': random_baseline,
            'edge_over_random': ai_accuracy - random_baseline,
            'ai_beats_random': ai_accuracy > random_baseline,
            'sample_size': len(resolved)
        }

    def statistical_significance(self) -> dict:
        """
        Calculate statistical significance of AI outperformance.

        Uses binomial test to determine if accuracy is significantly
        better than random chance (50%).

        Returns:
            Dictionary with p-value and significance assessment
        """
        resolved = [p for p in self.predictions if p.is_correct is not None]
        n = len(resolved)

        if n < 10:
            return {
                'p_value': None,
                'is_significant': False,
                'reason': 'Insufficient sample size (need at least 10)'
            }

        correct = sum(1 for p in resolved if p.is_correct)

        # Simple binomial approximation using normal distribution
        # For H0: p = 0.5 (random chance)
        expected = n * 0.5
        std_dev = (n * 0.5 * 0.5) ** 0.5

        if std_dev == 0:
            return {'p_value': None, 'is_significant': False, 'reason': 'Zero variance'}

        z_score = (correct - expected) / std_dev

        # Approximate p-value (one-tailed, testing if better than random)
        # Using simple approximation for standard normal CDF
        if HAS_NUMPY:
            from scipy import stats
            p_value = 1 - stats.norm.cdf(z_score)
        else:
            # Simple approximation without scipy
            # P(Z > z) for z > 0
            if z_score > 0:
                p_value = 0.5 * (1 - min(1, z_score / 3))  # Very rough approximation
            else:
                p_value = 0.5

        return {
            'correct': correct,
            'total': n,
            'accuracy': correct / n * 100,
            'z_score': z_score,
            'p_value': p_value,
            'is_significant': p_value < 0.05,
            'significance_level': '95%' if p_value < 0.05 else 'Not significant'
        }


# =============================================================================
# Utility Functions
# =============================================================================

def load_trades_from_csv(filepath: str) -> List[TradeRecord]:
    """Load trade records from CSV file."""
    trades = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            trades.append(TradeRecord(
                trade_id=row['trade_id'],
                timestamp=datetime.fromisoformat(row['timestamp']),
                symbol=row['symbol'],
                direction=row['direction'],
                entry_price=float(row['entry_price']),
                exit_price=float(row['exit_price']),
                position_size=float(row['position_size']),
                confidence=float(row['confidence']),
                predicted_direction=row['predicted_direction'],
                actual_direction=row['actual_direction'],
                pnl=float(row['pnl']),
                pnl_percent=float(row['pnl_percent']),
                holding_period=timedelta(seconds=float(row['holding_period_seconds'])),
                regime=row.get('regime', 'unknown'),
                fees=float(row.get('fees', 0))
            ))
    return trades


def print_summary(summary: PerformanceSummary) -> None:
    """Print a formatted performance summary."""
    criteria = summary.passes_criteria()

    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"Period: {summary.period_start.date()} to {summary.period_end.date()}")
    print(f"Total Trades: {summary.total_trades}")
    print(f"Winners: {summary.winning_trades} | Losers: {summary.losing_trades}")
    print("-" * 60)
    print("SUCCESS CRITERIA:")
    print(f"  Directional Accuracy: {summary.directional_accuracy:.1f}% (target >55%) {'PASS' if criteria['directional_accuracy'] else 'FAIL'}")
    print(f"  Win Rate: {summary.win_rate:.1f}% (target >50%) {'PASS' if criteria['win_rate'] else 'FAIL'}")
    print(f"  Profit Factor: {summary.profit_factor:.2f} (target >1.5) {'PASS' if criteria['profit_factor'] else 'FAIL'}")
    print(f"  Sharpe Ratio: {summary.sharpe_ratio:.2f} (target >1.0) {'PASS' if criteria['sharpe_ratio'] else 'FAIL'}")
    print("-" * 60)
    print(f"Total PnL: ${summary.total_pnl:.2f}")
    print(f"Average PnL: ${summary.average_pnl:.2f}")
    print(f"Max Drawdown: {summary.max_drawdown:.1f}%")
    print("=" * 60)
    print(f"OVERALL: {'ALL CRITERIA PASSED' if summary.all_criteria_passed() else 'SOME CRITERIA FAILED'}")
    print("=" * 60 + "\n")


# =============================================================================
# Main (for testing)
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Evaluation Framework v2 - Phase 2.1")
    print("Run with trade data to calculate metrics.\n")

    # Create sample data for demonstration (BTC/USDT only)
    sample_trades = [
        TradeRecord(
            trade_id="coinbase_T001",
            timestamp=datetime(2025, 1, 13, 10, 0),
            symbol="BTC/USDT",
            direction="LONG",
            entry_price=42000.0,
            exit_price=42500.0,
            position_size=0.00238,
            confidence=7.2,
            predicted_direction="UP",
            actual_direction="UP",
            pnl=1.19,
            pnl_percent=1.19,
            holding_period=timedelta(hours=24),
            regime="trending_up"
        ),
        TradeRecord(
            trade_id="kraken_T002",
            timestamp=datetime(2025, 1, 14, 10, 0),
            symbol="BTC/USDT",
            direction="SHORT",
            entry_price=43200.0,
            exit_price=42900.0,
            position_size=0.00231,
            confidence=6.95,
            predicted_direction="DOWN",
            actual_direction="DOWN",
            pnl=0.69,
            pnl_percent=0.69,
            holding_period=timedelta(hours=24),
            regime="ranging"
        ),
    ]

    # Calculate metrics
    calc = MetricsCalculator(trades=sample_trades)
    summary = calc.calculate_all()
    print_summary(summary)
