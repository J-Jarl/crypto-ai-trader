# Bitcoin Trading AI - Evaluation Framework Documentation

## Overview

The evaluation framework tests your trading AI's predictions against actual market outcomes to measure performance and identify areas for improvement. It systematically analyzes historical recommendations to determine accuracy, profitability, and whether contrarian strategies might perform better.

## Key Features

### 1. **Prediction Accuracy Testing**
- Compares AI recommendations (BUY/SELL/HOLD) against actual price movements
- Adjusts thresholds based on confidence levels
- Tracks accuracy by recommendation type

### 2. **Sentiment Analysis Validation**
- Tests if sentiment scores correlate with price direction
- Identifies when sentiment analysis is accurate vs inaccurate
- Helps refine sentiment interpretation

### 3. **Hypothetical Trading Performance**
- Calculates profit/loss as if trades were executed
- Respects stop loss and take profit levels
- Tracks win rate, average win/loss, and profit factor

### 4. **Contrarian Strategy Testing**
- Tests your hypothesis: does doing the OPPOSITE perform better?
- Compares standard vs contrarian accuracy
- Calculates hypothetical PnL for both strategies
- **This is a key feature for testing market maker dynamics theory**

### 5. **Comprehensive Reporting**
- Individual evaluation results for each prediction
- Aggregate performance statistics
- Saves results to JSON for further analysis

## How It Works

### Evaluation Process

```
1. Load Historical Prediction
   ├── Read JSON file from data/analysis_results/
   └── Extract: recommendation, confidence, sentiment, position size
   
2. Fetch Actual Price Movement
   ├── Get price at prediction time (entry)
   ├── Get price 24 hours later (exit)
   └── Calculate percent change
   
3. Determine Correctness
   ├── BUY: Did price increase?
   ├── SELL: Did price decrease?
   └── HOLD: Did price stay stable?
   
4. Calculate Hypothetical PnL
   ├── Simulate trade execution
   ├── Apply stop loss / take profit
   └── Calculate profit or loss in USD
   
5. Test Contrarian Strategy
   ├── Flip recommendation (BUY → SELL, SELL → BUY)
   ├── Check if opposite would be correct
   └── Calculate contrarian PnL
```

### Correctness Criteria

**HIGH Confidence Predictions:**
- BUY: Price must increase by >0.5%
- SELL: Price must decrease by <-0.5%
- HOLD: Price stays between -0.5% and +0.5%

**MEDIUM/LOW Confidence Predictions:**
- BUY: Any positive price movement
- SELL: Any negative price movement
- HOLD: Price stays between -0.5% and +0.5%

**Rationale:** Higher confidence should require stronger price movements to be considered correct.

### Sentiment Accuracy

Sentiment is accurate when:
- Positive sentiment (>0) → Price increases
- Negative sentiment (<0) → Price decreases
- Neutral sentiment (±0.2) → Price stable (±0.5%)

This helps you understand if your sentiment analysis is directionally correct, independent of the final recommendation.

## Usage

### Basic Usage

```bash
# From project root
cd ~/Documents/projects/crypto-ai-trader
python scripts/evaluation_framework.py
```

This will:
1. Find all predictions in `data/analysis_results/`
2. Evaluate the most recent 10 predictions (by default)
3. Generate performance report
4. Save results to JSON

### Output Example

```
============================================================
BITCOIN TRADING AI - EVALUATION FRAMEWORK
============================================================

EVALUATING 10 PREDICTIONS
============================================================

[1/10] Evaluating: bitcoin_analysis_20251113_143022.json
  Recommendation: BUY
  Price Change: +2.35%
  Prediction Correct: ✓
  Sentiment Accurate: ✓
  Hypothetical PnL: $+470.00

[2/10] Evaluating: bitcoin_analysis_20251113_120015.json
  Recommendation: SELL
  Price Change: +1.12%
  Prediction Correct: ✗
  Sentiment Accurate: ✗
  Hypothetical PnL: $-224.00

...

============================================================
TRADING AI PERFORMANCE REPORT
============================================================

📊 EVALUATION SUMMARY
   Total Predictions: 10
   Evaluation Period: 24 hours
   Date Range: 2025-11-03T12:00:00 to 2025-11-13T14:30:22

🎯 PREDICTION ACCURACY
   Overall: 65.00%
   Correct: 6
   Incorrect: 4
   
   By Type:
      BUY: 70.00% (5 predictions)
      SELL: 50.00% (2 predictions)
      HOLD: 66.67% (3 predictions)

💭 SENTIMENT ANALYSIS
   Accuracy: 60.00%
   Correct: 6
   Incorrect: 4

💰 TRADING PERFORMANCE
   Total Hypothetical PnL: $+1,234.56
   Win Rate: 60.00%
   Winning Trades: 6
   Losing Trades: 4
   Average Win: $458.33
   Average Loss: $-215.50
   Profit Factor: 2.13

🔄 CONTRARIAN STRATEGY ANALYSIS
   Contrarian Accuracy: 35.00%
   Contrarian Correct: 3
   Contrarian Hypothetical PnL: $-876.45
   Result: FOLLOWING BETTER

============================================================
```

### Advanced Usage

You can customize the evaluation by modifying the `main()` function:

```python
# Evaluate ALL predictions (no limit)
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=None)

# Use 48-hour evaluation window instead of 24
evaluations = evaluator.evaluate_all_predictions(hours_forward=48, limit=10)

# Evaluate only most recent 5 predictions
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=5)
```

## Understanding the Metrics

### Prediction Accuracy
**What it measures:** Percentage of recommendations that were correct based on actual price movement.

**How to interpret:**
- **>60%**: Good performance, AI is making profitable predictions
- **50-60%**: Slightly better than random, needs improvement
- **<50%**: Poor performance, may need significant strategy changes

### Win Rate
**What it measures:** Percentage of trades that would have been profitable.

**How to interpret:**
- **>55%**: Excellent, sustainable trading strategy
- **45-55%**: Acceptable if profit factor is good
- **<45%**: Needs improvement in trade selection

### Profit Factor
**What it measures:** Ratio of average win to average loss.

**Formula:** `Average Win / |Average Loss|`

**How to interpret:**
- **>2.0**: Excellent risk/reward ratio
- **1.5-2.0**: Good, sustainable
- **1.0-1.5**: Break-even to slightly profitable
- **<1.0**: Losing strategy

### Contrarian Analysis
**What it measures:** Performance if you did the OPPOSITE of AI recommendations.

**Why it matters:** Tests your hypothesis about market maker dynamics. If contrarian performs better, it suggests:
1. Market sentiment inversely correlates with price
2. "Buy the fear, sell the greed" may be effective
3. Your AI might benefit from contrarian logic

**How to interpret:**
- **Contrarian Better:** Consider inverting signals or adjusting sentiment interpretation
- **Following Better:** Current strategy is sound, focus on refinement

## Output Files

### Evaluation Report JSON

Saved to: `data/analysis_results/evaluation_report_YYYYMMDD_HHMMSS.json`

Structure:
```json
{
  "generated_at": "2025-11-14T10:30:45",
  "performance_report": {
    "evaluation_summary": { ... },
    "prediction_accuracy": { ... },
    "sentiment_analysis": { ... },
    "trading_performance": { ... },
    "contrarian_analysis": { ... }
  },
  "individual_evaluations": [
    {
      "timestamp": "...",
      "recommendation": "BUY",
      "percent_change": 2.35,
      "prediction_correct": true,
      "hypothetical_pnl": 470.00,
      ...
    },
    ...
  ]
}
```

This JSON can be:
- Loaded into Jupyter notebooks for visualization
- Tracked over time to monitor improvement
- Used to identify patterns in successful vs failed predictions

## Integration with Trading AI

### Workflow

```
Day 1-7: Generate Predictions
├── Run trading_ai.py daily
├── Accumulate predictions in data/analysis_results/
└── Wait for 24+ hours of price history

Day 8+: Evaluate Performance
├── Run evaluation_framework.py
├── Review accuracy and contrarian results
└── Identify patterns

Ongoing: Refine Strategy
├── If contrarian better → Consider inverting signals
├── If sentiment inaccurate → Adjust sentiment weights
├── If certain recommendation types fail → Investigate why
└── Document learnings for Phase 3 optimization
```

## Troubleshooting

### "No predictions could be evaluated"

**Causes:**
1. No JSON files in `data/analysis_results/`
2. Predictions are too recent (<24 hours old)
3. Exchange API is unavailable

**Solutions:**
1. Run `trading_ai.py` to generate predictions first
2. Wait at least 24 hours after generating predictions
3. Check internet connection and exchange availability

### "Error fetching price at [timestamp]"

**Causes:**
1. Exchange API rate limits
2. Timestamp outside available data range
3. Network connectivity issues

**Solutions:**
1. Add delays between API calls (system already includes this)
2. Focus on recent predictions (within last 30 days)
3. Verify internet connection

### Evaluation takes too long

**Solution:** Limit the number of predictions evaluated:

```python
# In main() function
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=5)
```

Start with 5-10 predictions for quick tests, then expand to all predictions for comprehensive analysis.

## Next Steps

### Immediate Actions
1. **Run Initial Evaluation**: Test with 5-10 most recent predictions
2. **Analyze Results**: Focus on prediction accuracy and contrarian comparison
3. **Document Findings**: Note which recommendation types perform best

### Short-term (This Week)
1. **Daily Evaluations**: Run framework weekly as new predictions accumulate
2. **Pattern Recognition**: Identify conditions where AI performs well vs poorly
3. **Hypothesis Testing**: Does Fear & Greed Index correlate with accuracy?

### Medium-term (This Month)
1. **Strategy Refinement**: Adjust based on evaluation findings
2. **Sentiment Tuning**: If sentiment accuracy is low, refine analysis
3. **Threshold Optimization**: Test different confidence thresholds

### Long-term (Phase 3)
1. **Backtesting**: Expand evaluation to cover longer time periods
2. **Feature Engineering**: Add new indicators based on learnings
3. **Fine-tuning Dataset**: Use successful predictions as training examples

## Technical Details

### Dependencies

```python
- json: Standard library
- os, pathlib: File system operations
- datetime, timedelta: Time calculations
- typing: Type hints
- ccxt: Exchange API (Coinbase)
```

### Data Flow

```
Input: data/analysis_results/*.json
  ↓
Load Predictions
  ↓
Fetch Historical Prices (CCXT)
  ↓
Calculate Outcomes
  ↓
Generate Metrics
  ↓
Output: evaluation_report_*.json
```

### Class Structure

**TradingEvaluator**
- `__init__()`: Initialize with results directory
- `load_prediction()`: Load single JSON file
- `get_price_at_time()`: Fetch BTC price at timestamp
- `get_price_change()`: Calculate price change over period
- `evaluate_prediction()`: Evaluate single prediction
- `evaluate_all_predictions()`: Batch evaluation
- `generate_performance_report()`: Aggregate statistics
- `save_evaluation_results()`: Save to JSON
- `print_performance_report()`: Console output

### Extending the Framework

Want to add new metrics? Here's how:

```python
# In generate_performance_report()
def generate_performance_report(self, evaluations: List[Dict]) -> Dict:
    # ... existing code ...
    
    # Add your custom metric
    custom_metric = self._calculate_custom_metric(evaluations)
    
    report['custom_analysis'] = custom_metric
    return report

# Add new method
def _calculate_custom_metric(self, evaluations: List[Dict]) -> Dict:
    # Your analysis here
    return {
        'metric_name': value,
        'interpretation': 'explanation'
    }
```

## Conclusion

This evaluation framework is your **Phase 2 foundation** for systematic testing. Use it to:

✅ Measure AI performance objectively  
✅ Test contrarian hypothesis  
✅ Identify improvement opportunities  
✅ Build confidence before paper trading  
✅ Create fine-tuning dataset from successful predictions

Remember: **Evaluation before optimization**. Gather data systematically before making strategy changes.

---

**Version:** 1.0  
**Last Updated:** November 14, 2025  
**Author:** J-Jarl  
**Phase:** 2 - Evaluation Framework
