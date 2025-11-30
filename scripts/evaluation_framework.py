#!/usr/bin/env python3
"""
Bitcoin Trading AI - Evaluation Framework
==========================================

Tests AI predictions against historical data to measure:
1. Prediction accuracy (BUY/SELL/HOLD correctness)
2. Sentiment analysis quality
3. Position sizing effectiveness
4. Risk/reward ratio outcomes
5. Contrarian vs following strategies

Author: J-Jarl
Created: November 14, 2025
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import ccxt
from pathlib import Path


class TradingEvaluator:
    """Evaluates trading AI predictions against actual market outcomes"""
    
    def __init__(self, results_dir: str = "data/analysis_results"):
        """
        Initialize the evaluator
        
        Args:
            results_dir: Path to directory containing AI trading recommendations
        """
        self.results_dir = Path(results_dir)
        self.exchange = ccxt.coinbase()
        self.evaluation_log = []
        
    def load_prediction(self, filepath: Path) -> Dict:
        """Load a single prediction JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def get_price_at_time(self, timestamp: datetime) -> float:
        """
        Fetch BTC/USD price at specific timestamp
        
        Args:
            timestamp: The datetime to fetch price for
            
        Returns:
            BTC price in USD, or None if unavailable
        """
        try:
            # Convert to milliseconds timestamp
            since = int(timestamp.timestamp() * 1000)
            
            # Fetch OHLCV data (1 hour candles)
            ohlcv = self.exchange.fetch_ohlcv(
                'BTC/USD',
                timeframe='1h',
                since=since,
                limit=1
            )
            
            if ohlcv and len(ohlcv) > 0:
                # Return close price [timestamp, open, high, low, close, volume]
                return ohlcv[0][4]
            return None
            
        except Exception as e:
            print(f"Error fetching price at {timestamp}: {e}")
            return None
    
    def get_price_change(self, start_time: datetime, hours: int = 24) -> Tuple[float, float]:
        """
        Calculate price change over specified period
        
        Args:
            start_time: Starting timestamp
            hours: Number of hours to measure (default 24)
            
        Returns:
            Tuple of (start_price, end_price, percent_change)
        """
        start_price = self.get_price_at_time(start_time)
        end_time = start_time + timedelta(hours=hours)
        end_price = self.get_price_at_time(end_time)
        
        if start_price and end_price:
            percent_change = ((end_price - start_price) / start_price) * 100
            return start_price, end_price, percent_change
        
        return None, None, None
    
    def evaluate_multiple_timeframes(self, prediction: Dict) -> Dict:
        """
        Evaluate a single prediction at multiple timeframes (4h, 12h, 24h)

        Args:
            prediction: The AI's trading recommendation

        Returns:
            Dictionary with evaluation results for each timeframe
        """
        timeframes = [4, 12, 24]
        results = {}

        # Parse prediction timestamp
        timestamp_str = prediction.get('timestamp', '')
        try:
            pred_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            print(f"Invalid timestamp format: {timestamp_str}")
            return None

        # Check how much time has passed since prediction
        time_since_prediction = datetime.now(pred_time.tzinfo) - pred_time
        hours_elapsed = time_since_prediction.total_seconds() / 3600

        # Evaluate each timeframe if enough time has passed
        for hours in timeframes:
            if hours_elapsed >= hours:
                # Sufficient time has passed for this evaluation
                evaluation = self.evaluate_prediction(prediction, hours_forward=hours)
                if evaluation:
                    results[f'{hours}h'] = evaluation
            else:
                # Not enough time has passed yet
                results[f'{hours}h'] = {
                    'status': 'pending',
                    'message': f'Need {hours - hours_elapsed:.1f} more hours',
                    'hours_needed': hours,
                    'hours_elapsed': round(hours_elapsed, 1)
                }

        return results

    def evaluate_prediction(self, prediction: Dict, hours_forward: int = 24) -> Dict:
        """
        Evaluate a single prediction against actual outcomes

        Args:
            prediction: The AI's trading recommendation
            hours_forward: Hours to look forward for outcome (default 24)

        Returns:
            Evaluation results dictionary
        """
        # Extract prediction data
        timestamp_str = prediction.get('timestamp', '')
        recommendation = prediction.get('recommendation', '')
        confidence = prediction.get('confidence_level', 'UNKNOWN')
        sentiment = prediction.get('sentiment_analysis', {})
        
        # Parse timestamp
        try:
            pred_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except:
            print(f"Invalid timestamp format: {timestamp_str}")
            return None
        
        # Get actual price movement
        start_price, end_price, percent_change = self.get_price_change(pred_time, hours_forward)
        
        if start_price is None or end_price is None:
            print(f"Could not fetch prices for evaluation at {timestamp_str}")
            return None
        
        # Determine if prediction was correct
        prediction_correct = self._evaluate_correctness(
            recommendation, 
            percent_change,
            confidence
        )
        
        # Evaluate sentiment accuracy
        sentiment_score = sentiment.get('overall_score', 0)
        sentiment_matches_price = self._evaluate_sentiment(sentiment_score, percent_change)
        
        # Calculate would-be profit/loss
        position_size = prediction.get('position_sizing', {}).get('recommended_size', 0)
        pnl = self._calculate_pnl(
            recommendation,
            start_price,
            end_price,
            position_size,
            prediction.get('risk_management', {})
        )
        
        # Compile evaluation results
        evaluation = {
            'timestamp': timestamp_str,
            'prediction_time': pred_time.isoformat(),
            'evaluation_period_hours': hours_forward,
            'recommendation': recommendation,
            'confidence': confidence,
            'sentiment_score': sentiment_score,
            'start_price': start_price,
            'end_price': end_price,
            'percent_change': round(percent_change, 2),
            'prediction_correct': prediction_correct,
            'sentiment_accurate': sentiment_matches_price,
            'hypothetical_pnl': pnl,
            'position_size': position_size
        }
        
        return evaluation
    
    def _evaluate_correctness(self, recommendation: str, percent_change: float, 
                             confidence: str) -> bool:
        """
        Determine if prediction was correct based on price movement
        
        Thresholds:
        - BUY: Price should increase (>0.5% for high confidence)
        - SELL: Price should decrease (<-0.5% for high confidence)
        - HOLD: Price should be relatively stable (-0.5% to +0.5%)
        """
        if confidence == "HIGH":
            buy_threshold = 0.5
            sell_threshold = -0.5
        else:
            buy_threshold = 0.0
            sell_threshold = 0.0
        
        if recommendation == "BUY":
            return percent_change > buy_threshold
        elif recommendation == "SELL":
            return percent_change < sell_threshold
        elif recommendation == "HOLD":
            return -0.5 <= percent_change <= 0.5
        
        return False
    
    def _evaluate_sentiment(self, sentiment_score: float, percent_change: float) -> bool:
        """
        Check if sentiment direction matches price movement
        
        Sentiment > 0 should correspond with price increase
        Sentiment < 0 should correspond with price decrease
        """
        if sentiment_score > 0 and percent_change > 0:
            return True
        elif sentiment_score < 0 and percent_change < 0:
            return True
        elif abs(sentiment_score) < 0.2 and abs(percent_change) < 0.5:
            # Both neutral
            return True
        
        return False
    
    def _calculate_pnl(self, recommendation: str, entry_price: float, 
                       exit_price: float, position_size: float, 
                       risk_mgmt: Dict) -> float:
        """
        Calculate hypothetical profit/loss for the trade
        
        Args:
            recommendation: BUY/SELL/HOLD
            entry_price: Entry price
            exit_price: Exit price after evaluation period
            position_size: Position size in BTC
            risk_mgmt: Risk management parameters (stop loss, take profit)
            
        Returns:
            PnL in USD
        """
        if recommendation == "HOLD" or position_size == 0:
            return 0.0
        
        stop_loss = risk_mgmt.get('stop_loss', 0)
        take_profit = risk_mgmt.get('take_profit', float('inf'))
        
        # Simulate exit
        if recommendation == "BUY":
            # Check if stop loss or take profit hit
            if exit_price <= stop_loss:
                exit_price = stop_loss
            elif exit_price >= take_profit:
                exit_price = take_profit
            
            pnl = (exit_price - entry_price) * position_size
            
        elif recommendation == "SELL":
            # Short position
            if exit_price >= stop_loss:
                exit_price = stop_loss
            elif exit_price <= take_profit:
                exit_price = take_profit
            
            pnl = (entry_price - exit_price) * position_size
        
        else:
            pnl = 0.0
        
        return round(pnl, 2)
    
    def evaluate_all_predictions(self, hours_forward: int = 24,
                                 limit: int = None) -> List[Dict]:
        """
        Evaluate all predictions in the results directory

        Args:
            hours_forward: Hours to look forward for outcomes
            limit: Maximum number of predictions to evaluate (None = all)

        Returns:
            List of evaluation results
        """
        # Get all prediction JSON files (exclude evaluation reports)
        all_files = sorted(self.results_dir.glob('*.json'), reverse=True)
        json_files = [f for f in all_files if not f.name.startswith('evaluation_')]

        if limit:
            json_files = json_files[:limit]

        print(f"\n{'='*60}")
        print(f"EVALUATING {len(json_files)} PREDICTIONS")
        print(f"{'='*60}\n")

        evaluations = []

        for i, filepath in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Evaluating: {filepath.name}")

            prediction = self.load_prediction(filepath)
            if not prediction:
                continue

            evaluation = self.evaluate_prediction(prediction, hours_forward)
            if evaluation:
                evaluations.append(evaluation)
                self._print_evaluation_summary(evaluation)

        return evaluations

    def evaluate_all_predictions_multi_timeframe(self, limit: int = None) -> Dict:
        """
        Evaluate all predictions at multiple timeframes (4h, 12h, 24h)

        Args:
            limit: Maximum number of predictions to evaluate (None = all)

        Returns:
            Dictionary with timeframe-separated evaluation results
        """
        # Get all prediction JSON files (exclude evaluation reports)
        all_files = sorted(self.results_dir.glob('*.json'), reverse=True)
        json_files = [f for f in all_files if not f.name.startswith('evaluation_')]

        if limit:
            json_files = json_files[:limit]

        print(f"\n{'='*60}")
        print(f"MULTI-TIMEFRAME EVALUATION: {len(json_files)} PREDICTIONS")
        print(f"Timeframes: 4h, 12h, 24h")
        print(f"{'='*60}\n")

        # Store results by timeframe
        timeframe_results = {
            '4h': [],
            '12h': [],
            '24h': []
        }

        for i, filepath in enumerate(json_files, 1):
            print(f"[{i}/{len(json_files)}] Evaluating: {filepath.name}")

            prediction = self.load_prediction(filepath)
            if not prediction:
                continue

            # Evaluate at all timeframes
            multi_eval = self.evaluate_multiple_timeframes(prediction)
            if not multi_eval:
                continue

            # Distribute results to appropriate timeframe lists
            for timeframe in ['4h', '12h', '24h']:
                tf_eval = multi_eval.get(timeframe)
                if tf_eval and tf_eval.get('status') != 'pending':
                    timeframe_results[timeframe].append(tf_eval)

            # Print summary for completed evaluations
            self._print_multi_timeframe_summary(multi_eval)

        return timeframe_results
    
    def _print_evaluation_summary(self, evaluation: Dict):
        """Print a brief summary of single evaluation"""
        correct_icon = "✓" if evaluation['prediction_correct'] else "✗"
        sentiment_icon = "✓" if evaluation['sentiment_accurate'] else "✗"

        print(f"  Recommendation: {evaluation['recommendation']}")
        print(f"  Price Change: {evaluation['percent_change']:+.2f}%")
        print(f"  Prediction Correct: {correct_icon}")
        print(f"  Sentiment Accurate: {sentiment_icon}")
        print(f"  Hypothetical PnL: ${evaluation['hypothetical_pnl']:+,.2f}")
        print()

    def _print_multi_timeframe_summary(self, multi_eval: Dict):
        """Print summary for multi-timeframe evaluation"""
        print(f"  Timeframe Results:")
        for timeframe in ['4h', '12h', '24h']:
            tf_result = multi_eval.get(timeframe, {})
            if tf_result.get('status') == 'pending':
                print(f"    {timeframe}: ⏳ {tf_result['message']}")
            elif 'prediction_correct' in tf_result:
                correct_icon = "✓" if tf_result['prediction_correct'] else "✗"
                print(f"    {timeframe}: {correct_icon} {tf_result['percent_change']:+.2f}% | PnL: ${tf_result['hypothetical_pnl']:+,.2f}")
        print()
    
    def generate_performance_report(self, evaluations: List[Dict]) -> Dict:
        """
        Generate comprehensive performance statistics
        
        Args:
            evaluations: List of evaluation results
            
        Returns:
            Performance metrics dictionary
        """
        if not evaluations:
            return {"error": "No evaluations to analyze"}
        
        total = len(evaluations)
        
        # Prediction accuracy
        correct_predictions = sum(1 for e in evaluations if e['prediction_correct'])
        accuracy = (correct_predictions / total) * 100
        
        # Sentiment accuracy
        correct_sentiment = sum(1 for e in evaluations if e['sentiment_accurate'])
        sentiment_accuracy = (correct_sentiment / total) * 100
        
        # By recommendation type
        buy_evals = [e for e in evaluations if e['recommendation'] == 'BUY']
        sell_evals = [e for e in evaluations if e['recommendation'] == 'SELL']
        hold_evals = [e for e in evaluations if e['recommendation'] == 'HOLD']
        
        buy_accuracy = self._calc_accuracy(buy_evals) if buy_evals else 0
        sell_accuracy = self._calc_accuracy(sell_evals) if sell_evals else 0
        hold_accuracy = self._calc_accuracy(hold_evals) if hold_evals else 0
        
        # PnL statistics
        total_pnl = sum(e['hypothetical_pnl'] for e in evaluations)
        winning_trades = [e for e in evaluations if e['hypothetical_pnl'] > 0]
        losing_trades = [e for e in evaluations if e['hypothetical_pnl'] < 0]
        
        win_rate = (len(winning_trades) / total) * 100 if total > 0 else 0
        
        avg_win = sum(e['hypothetical_pnl'] for e in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(e['hypothetical_pnl'] for e in losing_trades) / len(losing_trades) if losing_trades else 0
        
        # Contrarian analysis
        contrarian_results = self._analyze_contrarian_strategy(evaluations)
        
        report = {
            'evaluation_summary': {
                'total_predictions': total,
                'evaluation_period': f"{evaluations[0]['evaluation_period_hours']} hours",
                'date_range': f"{evaluations[-1]['prediction_time']} to {evaluations[0]['prediction_time']}"
            },
            'prediction_accuracy': {
                'overall_accuracy': round(accuracy, 2),
                'correct_predictions': correct_predictions,
                'incorrect_predictions': total - correct_predictions,
                'by_type': {
                    'BUY': {
                        'count': len(buy_evals),
                        'accuracy': round(buy_accuracy, 2)
                    },
                    'SELL': {
                        'count': len(sell_evals),
                        'accuracy': round(sell_accuracy, 2)
                    },
                    'HOLD': {
                        'count': len(hold_evals),
                        'accuracy': round(hold_accuracy, 2)
                    }
                }
            },
            'sentiment_analysis': {
                'sentiment_accuracy': round(sentiment_accuracy, 2),
                'correct_sentiment_predictions': correct_sentiment,
                'incorrect_sentiment_predictions': total - correct_sentiment
            },
            'trading_performance': {
                'total_hypothetical_pnl': round(total_pnl, 2),
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'win_rate': round(win_rate, 2),
                'average_win': round(avg_win, 2),
                'average_loss': round(avg_loss, 2),
                'profit_factor': round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else float('inf')
            },
            'contrarian_analysis': contrarian_results
        }
        
        return report
    
    def generate_multi_timeframe_report(self, timeframe_results: Dict) -> Dict:
        """
        Generate performance report comparing different timeframes

        Args:
            timeframe_results: Dictionary with results by timeframe (4h, 12h, 24h)

        Returns:
            Multi-timeframe comparison report
        """
        report = {
            'timeframe_comparison': {},
            'best_timeframe': None,
            'summary': {}
        }

        # Generate stats for each timeframe
        for timeframe in ['4h', '12h', '24h']:
            evaluations = timeframe_results.get(timeframe, [])

            if not evaluations:
                report['timeframe_comparison'][timeframe] = {
                    'status': 'no_data',
                    'message': 'Insufficient data for this timeframe'
                }
                continue

            # Calculate metrics for this timeframe
            total = len(evaluations)
            correct = sum(1 for e in evaluations if e['prediction_correct'])
            accuracy = (correct / total) * 100 if total > 0 else 0

            total_pnl = sum(e['hypothetical_pnl'] for e in evaluations)
            winning = [e for e in evaluations if e['hypothetical_pnl'] > 0]
            win_rate = (len(winning) / total) * 100 if total > 0 else 0

            avg_pnl = total_pnl / total if total > 0 else 0
            avg_win = sum(e['hypothetical_pnl'] for e in winning) / len(winning) if winning else 0

            report['timeframe_comparison'][timeframe] = {
                'total_predictions': total,
                'accuracy': round(accuracy, 2),
                'correct_predictions': correct,
                'win_rate': round(win_rate, 2),
                'total_pnl': round(total_pnl, 2),
                'average_pnl': round(avg_pnl, 2),
                'average_win': round(avg_win, 2),
                'winning_trades': len(winning)
            }

        # Determine best timeframe
        best_tf = None
        best_accuracy = 0
        for tf, stats in report['timeframe_comparison'].items():
            if stats.get('status') != 'no_data' and stats['accuracy'] > best_accuracy:
                best_accuracy = stats['accuracy']
                best_tf = tf

        report['best_timeframe'] = {
            'timeframe': best_tf,
            'accuracy': best_accuracy,
            'reason': 'Highest prediction accuracy'
        }

        # Summary statistics
        report['summary'] = {
            'timeframes_evaluated': [tf for tf in ['4h', '12h', '24h']
                                    if timeframe_results.get(tf)],
            'recommendation': self._get_timeframe_recommendation(report['timeframe_comparison'])
        }

        return report

    def _get_timeframe_recommendation(self, comparison: Dict) -> str:
        """Generate recommendation based on timeframe performance"""
        timeframes_with_data = [(tf, stats) for tf, stats in comparison.items()
                               if stats.get('status') != 'no_data']

        if not timeframes_with_data:
            return "Insufficient data to make recommendations"

        # Find best by accuracy
        best_acc = max(timeframes_with_data, key=lambda x: x[1]['accuracy'])

        # Find best by win rate
        best_wr = max(timeframes_with_data, key=lambda x: x[1]['win_rate'])

        if best_acc[0] == best_wr[0]:
            return f"Focus on {best_acc[0]} timeframe - best accuracy ({best_acc[1]['accuracy']:.1f}%) and win rate ({best_acc[1]['win_rate']:.1f}%)"
        else:
            return f"Mixed results: {best_acc[0]} has best accuracy ({best_acc[1]['accuracy']:.1f}%), {best_wr[0]} has best win rate ({best_wr[1]['win_rate']:.1f}%)"

    def _calc_accuracy(self, evaluations: List[Dict]) -> float:
        """Calculate accuracy percentage for subset of evaluations"""
        if not evaluations:
            return 0.0
        correct = sum(1 for e in evaluations if e['prediction_correct'])
        return (correct / len(evaluations)) * 100
    
    def _analyze_contrarian_strategy(self, evaluations: List[Dict]) -> Dict:
        """
        Test if doing the OPPOSITE of AI recommendations would perform better
        
        This tests the hypothesis that market sentiment inversely correlates
        with price movements due to market maker dynamics.
        """
        contrarian_correct = 0
        
        for eval in evaluations:
            # Flip the recommendation
            if eval['recommendation'] == 'BUY':
                contrarian_rec = 'SELL'
            elif eval['recommendation'] == 'SELL':
                contrarian_rec = 'BUY'
            else:
                contrarian_rec = 'HOLD'
            
            # Check if contrarian would be correct
            contrarian_would_be_correct = self._evaluate_correctness(
                contrarian_rec,
                eval['percent_change'],
                eval['confidence']
            )
            
            if contrarian_would_be_correct:
                contrarian_correct += 1
        
        total = len(evaluations)
        contrarian_accuracy = (contrarian_correct / total) * 100 if total > 0 else 0
        
        # Calculate hypothetical contrarian PnL
        contrarian_pnl = 0
        for eval in evaluations:
            if eval['recommendation'] == 'BUY':
                # Contrarian would SELL
                contrarian_pnl += -eval['hypothetical_pnl']
            elif eval['recommendation'] == 'SELL':
                # Contrarian would BUY
                contrarian_pnl += -eval['hypothetical_pnl']
            # HOLD remains HOLD
        
        return {
            'contrarian_accuracy': round(contrarian_accuracy, 2),
            'contrarian_correct_predictions': contrarian_correct,
            'contrarian_hypothetical_pnl': round(contrarian_pnl, 2),
            'performance_comparison': 'CONTRARIAN BETTER' if contrarian_accuracy > 50 else 'FOLLOWING BETTER'
        }
    
    def save_evaluation_results(self, evaluations: List[Dict], 
                               report: Dict, 
                               filename: str = None):
        """
        Save evaluation results and performance report to JSON
        
        Args:
            evaluations: List of individual evaluation results
            report: Performance report dictionary
            filename: Output filename (default: auto-generated with timestamp)
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"evaluation_report_{timestamp}.json"
        
        output_path = self.results_dir / filename
        
        output_data = {
            'generated_at': datetime.now().isoformat(),
            'performance_report': report,
            'individual_evaluations': evaluations
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"EVALUATION RESULTS SAVED")
        print(f"{'='*60}")
        print(f"Location: {output_path}")
        print()
    
    def print_performance_report(self, report: Dict):
        """Print formatted performance report to console"""
        print(f"\n{'='*60}")
        print(f"TRADING AI PERFORMANCE REPORT")
        print(f"{'='*60}\n")
        
        # Evaluation Summary
        summary = report['evaluation_summary']
        print(f"📊 EVALUATION SUMMARY")
        print(f"   Total Predictions: {summary['total_predictions']}")
        print(f"   Evaluation Period: {summary['evaluation_period']}")
        print(f"   Date Range: {summary['date_range']}")
        print()
        
        # Prediction Accuracy
        accuracy = report['prediction_accuracy']
        print(f"🎯 PREDICTION ACCURACY")
        print(f"   Overall: {accuracy['overall_accuracy']}%")
        print(f"   Correct: {accuracy['correct_predictions']}")
        print(f"   Incorrect: {accuracy['incorrect_predictions']}")
        print()
        print(f"   By Type:")
        for rec_type, stats in accuracy['by_type'].items():
            print(f"      {rec_type}: {stats['accuracy']}% ({stats['count']} predictions)")
        print()
        
        # Sentiment Analysis
        sentiment = report['sentiment_analysis']
        print(f"💭 SENTIMENT ANALYSIS")
        print(f"   Accuracy: {sentiment['sentiment_accuracy']}%")
        print(f"   Correct: {sentiment['correct_sentiment_predictions']}")
        print(f"   Incorrect: {sentiment['incorrect_sentiment_predictions']}")
        print()
        
        # Trading Performance
        trading = report['trading_performance']
        print(f"💰 TRADING PERFORMANCE")
        print(f"   Total Hypothetical PnL: ${trading['total_hypothetical_pnl']:+,.2f}")
        print(f"   Win Rate: {trading['win_rate']}%")
        print(f"   Winning Trades: {trading['winning_trades']}")
        print(f"   Losing Trades: {trading['losing_trades']}")
        print(f"   Average Win: ${trading['average_win']:,.2f}")
        print(f"   Average Loss: ${trading['average_loss']:,.2f}")
        print(f"   Profit Factor: {trading['profit_factor']}")
        print()
        
        # Contrarian Analysis
        contrarian = report['contrarian_analysis']
        print(f"🔄 CONTRARIAN STRATEGY ANALYSIS")
        print(f"   Contrarian Accuracy: {contrarian['contrarian_accuracy']}%")
        print(f"   Contrarian Correct: {contrarian['contrarian_correct_predictions']}")
        print(f"   Contrarian Hypothetical PnL: ${contrarian['contrarian_hypothetical_pnl']:+,.2f}")
        print(f"   Result: {contrarian['performance_comparison']}")
        print()
        
        print(f"{'='*60}\n")

    def print_multi_timeframe_report(self, report: Dict):
        """Print formatted multi-timeframe comparison report"""
        print(f"\n{'='*60}")
        print(f"MULTI-TIMEFRAME PERFORMANCE COMPARISON")
        print(f"{'='*60}\n")

        comparison = report['timeframe_comparison']

        print(f"⏱️  TIMEFRAME ANALYSIS (4h vs 12h vs 24h)\n")

        # Print table header
        print(f"{'Metric':<25} {'4h':>12} {'12h':>12} {'24h':>12}")
        print(f"{'-'*60}")

        # Prepare data for each metric
        metrics = {
            'Total Predictions': 'total_predictions',
            'Accuracy (%)': 'accuracy',
            'Win Rate (%)': 'win_rate',
            'Total PnL ($)': 'total_pnl',
            'Average PnL ($)': 'average_pnl',
            'Average Win ($)': 'average_win'
        }

        for metric_name, metric_key in metrics.items():
            values = []
            for tf in ['4h', '12h', '24h']:
                tf_data = comparison.get(tf, {})
                if tf_data.get('status') == 'no_data':
                    values.append('N/A')
                else:
                    val = tf_data.get(metric_key, 0)
                    if metric_key in ['total_pnl', 'average_pnl', 'average_win']:
                        values.append(f"${val:+,.2f}")
                    elif metric_key == 'total_predictions':
                        values.append(str(val))
                    else:
                        values.append(f"{val:.1f}%")

            print(f"{metric_name:<25} {values[0]:>12} {values[1]:>12} {values[2]:>12}")

        print()

        # Best timeframe
        if report['best_timeframe']['timeframe']:
            print(f"🏆 BEST TIMEFRAME")
            print(f"   Winner: {report['best_timeframe']['timeframe']}")
            print(f"   Accuracy: {report['best_timeframe']['accuracy']:.1f}%")
            print(f"   Reason: {report['best_timeframe']['reason']}")
            print()

        # Recommendation
        print(f"💡 RECOMMENDATION")
        print(f"   {report['summary']['recommendation']}")
        print()

        print(f"{'='*60}\n")


def main():
    """Main execution function"""
    import sys

    print("\n" + "="*60)
    print("BITCOIN TRADING AI - EVALUATION FRAMEWORK")
    print("="*60 + "\n")

    # Initialize evaluator
    evaluator = TradingEvaluator()

    # Check if results directory exists
    if not evaluator.results_dir.exists():
        print(f"❌ Results directory not found: {evaluator.results_dir}")
        print("   Please ensure trading_ai.py has generated some predictions first.")
        return

    # Check for multi-timeframe mode
    multi_timeframe = '--multi-timeframe' in sys.argv or '-m' in sys.argv

    print("Starting evaluation process...")
    print(f"Results directory: {evaluator.results_dir}")
    print(f"Mode: {'Multi-timeframe (4h, 12h, 24h)' if multi_timeframe else 'Single timeframe (24h)'}")
    print()

    if multi_timeframe:
        # Multi-timeframe evaluation
        timeframe_results = evaluator.evaluate_all_predictions_multi_timeframe(limit=10)

        # Check if we got any results
        has_data = any(len(results) > 0 for results in timeframe_results.values())

        if not has_data:
            print("❌ No predictions could be evaluated for any timeframe.")
            print("   This could be because:")
            print("   - No prediction files exist")
            print("   - Predictions are too recent (need at least 4h of price history)")
            print("   - Exchange API is unavailable")
            return

        # Generate and display multi-timeframe report
        multi_report = evaluator.generate_multi_timeframe_report(timeframe_results)
        evaluator.print_multi_timeframe_report(multi_report)

        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"evaluation_multi_timeframe_{timestamp}.json"
        output_path = evaluator.results_dir / filename

        output_data = {
            'generated_at': datetime.now().isoformat(),
            'mode': 'multi_timeframe',
            'timeframe_report': multi_report,
            'timeframe_results': {tf: results for tf, results in timeframe_results.items()}
        }

        with open(output_path, 'w') as f:
            import json
            json.dump(output_data, f, indent=2)

        print(f"\n{'='*60}")
        print(f"MULTI-TIMEFRAME RESULTS SAVED")
        print(f"{'='*60}")
        print(f"Location: {output_path}")
        print()

    else:
        # Standard 24-hour evaluation
        evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=10)

        if not evaluations:
            print("❌ No predictions could be evaluated.")
            print("   This could be because:")
            print("   - No prediction files exist")
            print("   - Predictions are too recent (need 24h of price history)")
            print("   - Exchange API is unavailable")
            return

        # Generate performance report
        report = evaluator.generate_performance_report(evaluations)

        # Display report
        evaluator.print_performance_report(report)

        # Save results
        evaluator.save_evaluation_results(evaluations, report)

    print("✅ Evaluation complete!")
    print()


if __name__ == "__main__":
    main()
