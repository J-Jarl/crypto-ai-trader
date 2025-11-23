#!/usr/bin/env python3
"""
Evaluation Framework Quick Test
================================

Verifies that the evaluation framework is set up correctly:
1. Checks for prediction files
2. Tests exchange API connection
3. Runs a minimal evaluation on 1-2 predictions
4. Validates output format

Use this before running full evaluations to catch setup issues early.

Author: J-Jarl
Created: November 14, 2025
"""

import sys
from pathlib import Path
from datetime import datetime
import ccxt

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def print_status(message, status='info'):
    """Print colored status message"""
    if status == 'success':
        print(f"{GREEN}✓{RESET} {message}")
    elif status == 'error':
        print(f"{RED}✗{RESET} {message}")
    elif status == 'warning':
        print(f"{YELLOW}⚠{RESET} {message}")
    else:
        print(f"{BLUE}ℹ{RESET} {message}")

def check_results_directory():
    """Check if results directory exists and has prediction files"""
    print("\n" + "="*60)
    print("1. CHECKING RESULTS DIRECTORY")
    print("="*60)
    
    results_dir = Path("data/analysis_results")
    
    if not results_dir.exists():
        print_status(f"Results directory not found: {results_dir}", 'error')
        print_status("Create it with: mkdir -p data/analysis_results", 'info')
        return False, None
    
    print_status(f"Results directory exists: {results_dir}", 'success')
    
    # Count JSON files
    json_files = list(results_dir.glob('*.json'))
    
    if not json_files:
        print_status("No prediction files found", 'warning')
        print_status("Run trading_ai.py first to generate predictions", 'info')
        return False, None
    
    print_status(f"Found {len(json_files)} prediction files", 'success')
    
    # Show most recent file
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
    print_status(f"Most recent: {latest.name} ({mod_time})", 'info')
    
    return True, json_files

def test_exchange_connection():
    """Test connection to exchange API"""
    print("\n" + "="*60)
    print("2. TESTING EXCHANGE API CONNECTION")
    print("="*60)
    
    try:
        exchange = ccxt.coinbase()
        print_status("Initialized Coinbase exchange", 'success')
        
        # Try to fetch current BTC price
        ticker = exchange.fetch_ticker('BTC/USD')
        price = ticker['last']
        
        print_status(f"Current BTC/USD price: ${price:,.2f}", 'success')
        print_status("Exchange API is working correctly", 'success')
        return True
        
    except Exception as e:
        print_status(f"Exchange API error: {e}", 'error')
        print_status("Evaluation framework requires exchange access", 'warning')
        return False

def test_single_evaluation(json_files):
    """Run a quick evaluation on the most recent prediction"""
    print("\n" + "="*60)
    print("3. TESTING EVALUATION ON RECENT PREDICTION")
    print("="*60)
    
    if not json_files:
        print_status("No files to evaluate", 'warning')
        return False
    
    # Import here to avoid issues if evaluation_framework.py has errors
    try:
        from evaluation_framework import TradingEvaluator
        print_status("Imported evaluation_framework.py successfully", 'success')
    except ImportError as e:
        print_status(f"Cannot import evaluation_framework: {e}", 'error')
        print_status("Ensure evaluation_framework.py is in scripts/ directory", 'info')
        return False
    
    # Initialize evaluator
    try:
        evaluator = TradingEvaluator()
        print_status("Initialized TradingEvaluator", 'success')
    except Exception as e:
        print_status(f"Error initializing evaluator: {e}", 'error')
        return False
    
    # Get most recent file
    latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
    print_status(f"Testing with: {latest_file.name}", 'info')
    
    # Load prediction
    try:
        prediction = evaluator.load_prediction(latest_file)
        if not prediction:
            print_status("Failed to load prediction", 'error')
            return False
        print_status("Loaded prediction successfully", 'success')
        
        # Show prediction details
        print_status(f"Recommendation: {prediction.get('recommendation', 'N/A')}", 'info')
        print_status(f"Timestamp: {prediction.get('timestamp', 'N/A')}", 'info')
        
    except Exception as e:
        print_status(f"Error loading prediction: {e}", 'error')
        return False
    
    # Try evaluation
    try:
        print_status("Running evaluation (this may take 10-20 seconds)...", 'info')
        evaluation = evaluator.evaluate_prediction(prediction, hours_forward=24)
        
        if not evaluation:
            print_status("Evaluation returned None (prediction may be too recent)", 'warning')
            print_status("Predictions need 24+ hours of history to evaluate", 'info')
            return False
        
        print_status("Evaluation completed successfully!", 'success')
        
        # Show evaluation results
        print("\n" + "-"*60)
        print("EVALUATION RESULTS")
        print("-"*60)
        print(f"Recommendation: {evaluation['recommendation']}")
        print(f"Price Change: {evaluation['percent_change']:+.2f}%")
        print(f"Prediction Correct: {'✓' if evaluation['prediction_correct'] else '✗'}")
        print(f"Sentiment Accurate: {'✓' if evaluation['sentiment_accurate'] else '✗'}")
        print(f"Hypothetical PnL: ${evaluation['hypothetical_pnl']:+,.2f}")
        print("-"*60)
        
        return True
        
    except Exception as e:
        print_status(f"Error during evaluation: {e}", 'error')
        import traceback
        print("\nDetailed error:")
        traceback.print_exc()
        return False

def check_dependencies():
    """Verify required Python packages are installed"""
    print("\n" + "="*60)
    print("4. CHECKING DEPENDENCIES")
    print("="*60)
    
    required = ['ccxt', 'requests', 'feedparser']
    all_good = True
    
    for package in required:
        try:
            __import__(package)
            print_status(f"{package}: installed", 'success')
        except ImportError:
            print_status(f"{package}: NOT INSTALLED", 'error')
            all_good = False
    
    if not all_good:
        print_status("Install missing packages with:", 'info')
        print_status("pip install -r requirements.txt", 'info')
    
    return all_good

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("EVALUATION FRAMEWORK - QUICK TEST")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # Test 1: Check results directory
    has_files, json_files = check_results_directory()
    results.append(('Results Directory', has_files))
    
    # Test 2: Exchange API
    exchange_ok = test_exchange_connection()
    results.append(('Exchange API', exchange_ok))
    
    # Test 3: Dependencies
    deps_ok = check_dependencies()
    results.append(('Dependencies', deps_ok))
    
    # Test 4: Single evaluation (only if previous tests passed)
    if has_files and exchange_ok:
        eval_ok = test_single_evaluation(json_files)
        results.append(('Evaluation Test', eval_ok))
    else:
        print_status("\nSkipping evaluation test (prerequisites not met)", 'warning')
        results.append(('Evaluation Test', False))
    
    # Final summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results:
        status = 'success' if passed else 'error'
        print_status(f"{test_name}: {'PASSED' if passed else 'FAILED'}", status)
    
    all_passed = all(passed for _, passed in results)
    
    print("\n" + "="*60)
    if all_passed:
        print_status("ALL TESTS PASSED! ✨", 'success')
        print_status("You can now run: python scripts/evaluation_framework.py", 'info')
    else:
        print_status("SOME TESTS FAILED", 'error')
        print_status("Fix issues above before running full evaluation", 'info')
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
