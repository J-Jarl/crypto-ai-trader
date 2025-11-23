# Phase 2: Evaluation Framework - Complete Guide

## Overview

Phase 2 builds systematic testing capabilities to measure your trading AI's performance against historical data. This evaluation phase is **critical** before advancing to paper trading or model fine-tuning. You need to understand what works and what doesn't based on real data.

## What Phase 2 Achieves

✅ **Objective Performance Measurement** - Know exactly how accurate your predictions are  
✅ **Contrarian Strategy Testing** - Test if doing the opposite performs better  
✅ **Sentiment Validation** - Verify your sentiment analysis quality  
✅ **Pattern Recognition** - Identify when AI performs well vs poorly  
✅ **Data-Driven Decisions** - Build Phase 3 optimizations on evidence, not guesses

## Files Created

### Core Components

1. **`evaluation_framework.py`** (Main Script)
   - Loads historical predictions
   - Fetches actual price movements via CCXT
   - Calculates accuracy metrics
   - Tests contrarian strategy
   - Generates comprehensive reports

2. **`EVALUATION_FRAMEWORK_GUIDE.md`** (Documentation)
   - Detailed feature explanations
   - Metric interpretations
   - Usage examples
   - Troubleshooting guide
   - Integration workflows

3. **`test_evaluation.py`** (Quick Test)
   - Verifies setup is correct
   - Tests exchange connectivity
   - Runs minimal evaluation
   - Validates dependencies

## Quick Start

### Step 1: Setup Files

```bash
# Copy files to your project
cp evaluation_framework.py ~/Documents/projects/crypto-ai-trader/scripts/
cp test_evaluation.py ~/Documents/projects/crypto-ai-trader/scripts/
cp EVALUATION_FRAMEWORK_GUIDE.md ~/Documents/projects/crypto-ai-trader/

# Navigate to project
cd ~/Documents/projects/crypto-ai-trader
```

### Step 2: Verify Setup

```bash
# Run quick test
python scripts/test_evaluation.py
```

**Expected Output:**
```
✓ Results directory exists
✓ Found X prediction files
✓ Exchange API working
✓ All dependencies installed
✓ Evaluation test passed
```

**If tests fail**, see [Troubleshooting](#troubleshooting) section below.

### Step 3: Run First Evaluation

```bash
# Evaluate your predictions
python scripts/evaluation_framework.py
```

This will:
- Evaluate your 10 most recent predictions
- Calculate accuracy metrics
- Test contrarian strategy
- Save detailed report to JSON

### Step 4: Review Results

Check the output for:
- **Overall Accuracy**: Is it >60%? (Good performance)
- **Contrarian Analysis**: Does opposite perform better?
- **Win Rate**: Are profitable trades >55%?
- **Profit Factor**: Is it >2.0? (Excellent risk/reward)

## Understanding Your Results

### Key Metrics Explained

#### Prediction Accuracy
```
Overall: 65.00%
├── BUY: 70.00% (5 predictions)
├── SELL: 50.00% (2 predictions)
└── HOLD: 66.67% (3 predictions)
```

**What this tells you:**
- Your AI is slightly better at identifying buy opportunities
- Sell signals need improvement
- Overall performance is profitable

**Target:** >60% overall for sustainable trading

#### Contrarian Analysis
```
Contrarian Accuracy: 35.00%
Contrarian PnL: $-876.45
Result: FOLLOWING BETTER
```

**What this tells you:**
- Your current strategy is sound (following sentiment works)
- Market maker inverse correlation is NOT present in your data
- Continue with current approach, focus on refinement

**Alternative outcome:**
```
Contrarian Accuracy: 68.00%
Contrarian PnL: $+1,234.00
Result: CONTRARIAN BETTER
```

**What this would tell you:**
- Inverse signals may be more profitable
- Consider flipping buy/sell recommendations
- "Buy fear, sell greed" strategy worth exploring

#### Trading Performance
```
Total PnL: $+1,234.56
Win Rate: 60.00%
Average Win: $458.33
Average Loss: $-215.50
Profit Factor: 2.13
```

**What this tells you:**
- System is profitable
- Good win rate (>55%)
- Excellent profit factor (2.13 means wins are 2x larger than losses)
- Risk/reward ratio is well-balanced

## Workflow for Phase 2

### Week 1: Initial Testing

**Day 1-2: Setup & First Evaluation**
```bash
# Morning
python scripts/test_evaluation.py  # Verify setup

# If all tests pass
python scripts/evaluation_framework.py  # Run evaluation

# Review results
cat data/analysis_results/evaluation_report_*.json | python -m json.tool
```

**Day 3-4: Generate More Predictions**
```bash
# Run trading AI daily
python scripts/trading_ai.py

# Let predictions accumulate
# (Need 24+ hours of price history per prediction)
```

**Day 5-7: Comprehensive Analysis**
- Evaluate all accumulated predictions
- Look for patterns in correct vs incorrect predictions
- Document findings in a notebook or markdown file

### Week 2-4: Pattern Recognition

Create a simple analysis notebook:

```python
# analysis.ipynb
import json
from pathlib import Path

# Load evaluation results
results_dir = Path("data/analysis_results")
eval_files = list(results_dir.glob("evaluation_report_*.json"))

# Load latest report
with open(eval_files[-1]) as f:
    report = json.load(f)

# Analyze by conditions
for eval in report['individual_evaluations']:
    if eval['prediction_correct']:
        print(f"✓ Correct: {eval['recommendation']} - Sentiment: {eval['sentiment_score']}")
    else:
        print(f"✗ Wrong: {eval['recommendation']} - Sentiment: {eval['sentiment_score']}")
```

Look for patterns like:
- Do high sentiment scores (>0.7) correlate with better accuracy?
- Are predictions more accurate during certain times of day?
- Do HOLD recommendations during high volatility perform well?

## Customization

### Adjust Evaluation Period

By default, predictions are evaluated after 24 hours. To change this:

```python
# In evaluation_framework.py main() function
# Change this line:
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=10)

# To evaluate after 48 hours:
evaluations = evaluator.evaluate_all_predictions(hours_forward=48, limit=10)

# To evaluate all predictions (no limit):
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=None)
```

### Add Custom Metrics

Want to track additional metrics? Extend the framework:

```python
# In generate_performance_report() method
def generate_performance_report(self, evaluations: List[Dict]) -> Dict:
    # ... existing code ...
    
    # Add custom metric - example: accuracy during high volatility
    high_volatility = [e for e in evaluations if abs(e['percent_change']) > 3.0]
    high_vol_accuracy = self._calc_accuracy(high_volatility) if high_volatility else 0
    
    report['volatility_analysis'] = {
        'high_volatility_predictions': len(high_volatility),
        'high_volatility_accuracy': round(high_vol_accuracy, 2)
    }
    
    return report
```

## Integration with Trading AI

### Data Flow

```
Phase 1: Trading AI (Daily)
├── Fetch: News, Price, Fear & Greed
├── Analyze: Sentiment, Technical Indicators
├── Output: Trading recommendation JSON
└── Save to: data/analysis_results/

Phase 2: Evaluation (Weekly)
├── Load: Historical predictions
├── Fetch: Actual price movements (24h later)
├── Calculate: Accuracy, PnL, Contrarian
├── Output: Performance report JSON
└── Insights: What works, what doesn't

Phase 3: Optimization (Monthly)
├── Based on: Evaluation findings
├── Adjust: Sentiment weights, thresholds
├── Test: Modified strategy in evaluation
└── Iterate: Repeat until satisfactory
```

### Recommended Schedule

**Daily:**
- Run `trading_ai.py` once per day
- Accumulate predictions

**Weekly:**
- Run `evaluation_framework.py`
- Review performance trends
- Document insights

**Monthly:**
- Comprehensive analysis session
- Compare month-over-month improvement
- Plan Phase 3 optimizations

## Troubleshooting

### "No predictions could be evaluated"

**Problem:** Predictions are too recent (need 24h of history)

**Solution:**
```bash
# Check prediction timestamps
ls -lt data/analysis_results/*.json | head

# Verify they're 24+ hours old
# If not, wait and try again tomorrow
```

### "Exchange API error: 451 Client Error"

**Problem:** Geo-restricted API access

**Solution:** System automatically uses Coinbase/Kraken as fallback. If still failing:
```python
# In evaluation_framework.py, line 22
# Try alternative exchange:
self.exchange = ccxt.kraken()  # Instead of coinbase()
```

### "Import error: cannot import evaluation_framework"

**Problem:** File not in correct location

**Solution:**
```bash
# Verify file location
ls -l scripts/evaluation_framework.py

# If missing, copy it:
cp evaluation_framework.py ~/Documents/projects/crypto-ai-trader/scripts/
```

### Evaluation is slow

**Problem:** Too many API calls

**Solution:** Start with fewer predictions:
```python
# In main() function
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=5)
```

## Git Integration

### Commit Your Work

```bash
# Stage new files
git add scripts/evaluation_framework.py
git add scripts/test_evaluation.py
git add EVALUATION_FRAMEWORK_GUIDE.md

# Commit with descriptive message
git commit -m "Add Phase 2 evaluation framework for prediction testing"

# Push to GitHub
git push origin main
```

### What NOT to Commit

Evaluation reports contain trading data - add to `.gitignore`:

```bash
# In your .gitignore file, add:
data/analysis_results/evaluation_report_*.json
```

Keep prediction JSONs private (already ignored), but you can commit:
- Source code (`evaluation_framework.py`)
- Documentation (`EVALUATION_FRAMEWORK_GUIDE.md`)
- Tests (`test_evaluation.py`)

## Next Steps to Phase 3

Once you have 2-4 weeks of evaluation data:

### Analyze Patterns
- Which conditions lead to accurate predictions?
- What causes false signals?
- Are certain recommendation types more reliable?

### Optimize Strategy
- Adjust sentiment scoring weights
- Modify confidence thresholds
- Add new indicators based on findings

### Validate Changes
- Re-run evaluation with modified strategy
- Compare old vs new performance
- Document improvements

### Prepare for Phase 4
- If accuracy >65% and profit factor >2.0
- Consider paper trading validation
- Begin collecting fine-tuning dataset

## Success Criteria for Phase 2

Before advancing to Phase 3, you should have:

✅ **At least 20-30 evaluated predictions**  
✅ **Clear understanding of accuracy patterns**  
✅ **Contrarian hypothesis tested**  
✅ **Documentation of what works vs doesn't**  
✅ **Baseline metrics established for comparison**

Target Metrics for Success:
- Prediction Accuracy: >60%
- Win Rate: >55%
- Profit Factor: >1.5
- Sentiment Accuracy: >60%

## Resources

- **Main Guide**: `EVALUATION_FRAMEWORK_GUIDE.md`
- **Quick Test**: `python scripts/test_evaluation.py`
- **Project Structure**: `PROJECT_STRUCTURE.md`
- **Trading AI Code**: `scripts/trading_ai.py`

## Support & Community

If you encounter issues:

1. **Check Documentation**: Review `EVALUATION_FRAMEWORK_GUIDE.md` thoroughly
2. **Run Quick Test**: `python scripts/test_evaluation.py` to isolate issues
3. **Review Logs**: Check console output for specific error messages
4. **Inspect Data**: Manually review JSON files for inconsistencies

## Summary

Phase 2 transforms your trading AI from "black box" to "understood system." By systematically evaluating predictions, you gain confidence in what works and identify what needs improvement. This data-driven approach ensures Phase 3 optimizations are based on evidence, not guesswork.

**Remember:** Good evaluation is the foundation of good trading strategy. Take your time with Phase 2 - the insights you gain here will guide all future development.

---

**Phase 2 Status:** Complete ✅  
**Files Created:** 3 (framework, guide, test)  
**Next Phase:** Phase 3 - Strategy Optimization (based on evaluation findings)  
**Timeline:** 2-4 weeks of data collection recommended before Phase 3

**Last Updated:** November 14, 2025  
**Author:** J-Jarl
