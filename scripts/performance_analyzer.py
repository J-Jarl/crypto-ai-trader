"""
Performance Analyzer - Pandas-based analysis of backtest results

Analyzes evaluation results to identify:
- Pattern success rates
- Best/worst performing patterns
- Win rates by confidence level
- PnL analysis by pattern type
- Temporal patterns (which times/days perform best)
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys


class PerformanceAnalyzer:
    """Analyze trading system performance using pandas"""

    def __init__(self, evaluations_dir: str = "data/backtest_results/evaluations"):
        """
        Initialize performance analyzer

        Args:
            evaluations_dir: Directory containing evaluation JSON files
        """
        self.evaluations_dir = Path(evaluations_dir)
        self.df = None

    def load_evaluations(self) -> pd.DataFrame:
        """
        Load all evaluation files into a pandas DataFrame

        Returns:
            DataFrame with all evaluation results
        """
        print(f"Loading evaluations from {self.evaluations_dir}")

        eval_files = list(self.evaluations_dir.glob("eval_*.json"))

        if not eval_files:
            print(f"No evaluation files found in {self.evaluations_dir}")
            return pd.DataFrame()

        data = []
        for file in eval_files:
            try:
                with open(file, 'r') as f:
                    eval_data = json.load(f)
                    data.append(eval_data)
            except Exception as e:
                print(f"Error loading {file}: {e}")

        # Create DataFrame
        self.df = pd.DataFrame(data)

        # Convert timestamp columns to datetime
        if 'timestamp' in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        if 'prediction_time' in self.df.columns:
            self.df['prediction_time'] = pd.to_datetime(self.df['prediction_time'])

        # Add derived columns
        self.df['week'] = self.df['prediction_time'].dt.isocalendar().week
        self.df['day_of_week'] = self.df['prediction_time'].dt.day_name()
        self.df['hour'] = self.df['prediction_time'].dt.hour
        self.df['time_slot'] = self.df['hour'].apply(lambda h: 'AM' if h == 9 else 'PM')

        print(f"✅ Loaded {len(self.df)} evaluations")
        return self.df

    def get_overall_stats(self) -> Dict:
        """Get overall performance statistics"""
        if self.df is None or len(self.df) == 0:
            return {}

        stats = {
            'total_trades': len(self.df),
            'correct': self.df['prediction_correct'].sum(),
            'accuracy': (self.df['prediction_correct'].sum() / len(self.df)) * 100,
            'total_pnl': self.df['hypothetical_pnl'].sum(),
            'avg_pnl': self.df['hypothetical_pnl'].mean(),
            'win_rate': (self.df['hypothetical_pnl'] > 0).sum() / len(self.df) * 100,
            'avg_win': self.df[self.df['hypothetical_pnl'] > 0]['hypothetical_pnl'].mean(),
            'avg_loss': self.df[self.df['hypothetical_pnl'] < 0]['hypothetical_pnl'].mean(),
        }

        return stats

    def analyze_by_recommendation(self) -> pd.DataFrame:
        """Analyze performance by recommendation type (BUY/SELL/HOLD)"""
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        grouped = self.df.groupby('recommendation').agg({
            'prediction_correct': ['count', 'sum', 'mean'],
            'hypothetical_pnl': ['sum', 'mean', 'min', 'max']
        }).round(2)

        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.rename(columns={
            'prediction_correct_count': 'total_trades',
            'prediction_correct_sum': 'correct',
            'prediction_correct_mean': 'accuracy',
            'hypothetical_pnl_sum': 'total_pnl',
            'hypothetical_pnl_mean': 'avg_pnl',
            'hypothetical_pnl_min': 'worst_trade',
            'hypothetical_pnl_max': 'best_trade'
        })

        return grouped

    def analyze_by_confidence(self) -> pd.DataFrame:
        """Analyze performance by confidence level"""
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        grouped = self.df.groupby('confidence').agg({
            'prediction_correct': ['count', 'sum', 'mean'],
            'hypothetical_pnl': ['sum', 'mean']
        }).round(2)

        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.rename(columns={
            'prediction_correct_count': 'total_trades',
            'prediction_correct_sum': 'correct',
            'prediction_correct_mean': 'accuracy',
            'hypothetical_pnl_sum': 'total_pnl',
            'hypothetical_pnl_mean': 'avg_pnl'
        })

        return grouped

    def analyze_by_week(self) -> pd.DataFrame:
        """Analyze performance by week"""
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        grouped = self.df.groupby('week').agg({
            'prediction_correct': ['count', 'sum', 'mean'],
            'hypothetical_pnl': ['sum', 'mean']
        }).round(2)

        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.rename(columns={
            'prediction_correct_count': 'total_trades',
            'prediction_correct_sum': 'correct',
            'prediction_correct_mean': 'accuracy',
            'hypothetical_pnl_sum': 'total_pnl',
            'hypothetical_pnl_mean': 'avg_pnl'
        })

        return grouped

    def analyze_by_time_slot(self) -> pd.DataFrame:
        """Analyze performance by time slot (AM vs PM)"""
        if self.df is None or len(self.df) == 0:
            return pd.DataFrame()

        grouped = self.df.groupby('time_slot').agg({
            'prediction_correct': ['count', 'sum', 'mean'],
            'hypothetical_pnl': ['sum', 'mean']
        }).round(2)

        grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
        grouped = grouped.rename(columns={
            'prediction_correct_count': 'total_trades',
            'prediction_correct_sum': 'correct',
            'prediction_correct_mean': 'accuracy',
            'hypothetical_pnl_sum': 'total_pnl',
            'hypothetical_pnl_mean': 'avg_pnl'
        })

        return grouped

    def get_best_worst_trades(self, n: int = 5) -> Dict[str, pd.DataFrame]:
        """Get best and worst performing trades"""
        if self.df is None or len(self.df) == 0:
            return {}

        best = self.df.nlargest(n, 'hypothetical_pnl')[
            ['prediction_time', 'recommendation', 'percent_change', 'hypothetical_pnl', 'confidence']
        ]

        worst = self.df.nsmallest(n, 'hypothetical_pnl')[
            ['prediction_time', 'recommendation', 'percent_change', 'hypothetical_pnl', 'confidence']
        ]

        return {'best': best, 'worst': worst}

    def print_summary_report(self):
        """Print comprehensive summary report"""
        print("\n" + "="*80)
        print("TRADING SYSTEM PERFORMANCE ANALYSIS")
        print("="*80)

        # Overall stats
        stats = self.get_overall_stats()
        if stats:
            print(f"\n📊 OVERALL PERFORMANCE")
            print(f"{'─'*80}")
            print(f"Total Trades:     {stats['total_trades']}")
            print(f"Correct:          {stats['correct']} ({stats['accuracy']:.1f}%)")
            print(f"Total PnL:        ${stats['total_pnl']:+.2f}")
            print(f"Average PnL:      ${stats['avg_pnl']:+.2f}")
            print(f"Win Rate:         {stats['win_rate']:.1f}%")
            print(f"Average Win:      ${stats['avg_win']:+.2f}")
            print(f"Average Loss:     ${stats['avg_loss']:+.2f}")

        # By recommendation
        print(f"\n📈 PERFORMANCE BY RECOMMENDATION TYPE")
        print(f"{'─'*80}")
        by_rec = self.analyze_by_recommendation()
        if not by_rec.empty:
            print(by_rec.to_string())

        # By confidence
        print(f"\n💪 PERFORMANCE BY CONFIDENCE LEVEL")
        print(f"{'─'*80}")
        by_conf = self.analyze_by_confidence()
        if not by_conf.empty:
            print(by_conf.to_string())

        # By week
        print(f"\n📅 PERFORMANCE BY WEEK")
        print(f"{'─'*80}")
        by_week = self.analyze_by_week()
        if not by_week.empty:
            print(by_week.to_string())

        # By time slot
        print(f"\n⏰ PERFORMANCE BY TIME SLOT")
        print(f"{'─'*80}")
        by_time = self.analyze_by_time_slot()
        if not by_time.empty:
            print(by_time.to_string())

        # Best/Worst trades
        trades = self.get_best_worst_trades(5)
        if trades:
            print(f"\n🏆 TOP 5 BEST TRADES")
            print(f"{'─'*80}")
            if not trades['best'].empty:
                print(trades['best'].to_string(index=False))

            print(f"\n💀 TOP 5 WORST TRADES")
            print(f"{'─'*80}")
            if not trades['worst'].empty:
                print(trades['worst'].to_string(index=False))

        print("\n" + "="*80)


def main():
    """Run performance analysis"""
    analyzer = PerformanceAnalyzer()

    # Load data
    df = analyzer.load_evaluations()

    if df.empty:
        print("No evaluation data found. Run backtest first!")
        return

    # Print summary report
    analyzer.print_summary_report()

    # Optionally export to CSV
    export_path = "data/backtest_results/performance_analysis.csv"
    df.to_csv(export_path, index=False)
    print(f"\n💾 Full data exported to: {export_path}")


if __name__ == "__main__":
    main()
