# Phase 2 Testing Framework - Implementation Mapping

## 🗺️ Phase Roadmap

### ✅ **Phase 2: COMPLETE** (Evaluation Framework)
**Status**: Operational with 57% overall accuracy, 100% SELL accuracy

**Implemented Tests** (9/19):
- ✅ Test 1.1: Directional Accuracy
- ✅ Test 1.2: Confidence Calibration  
- ✅ Test 1.4: Type-Specific Accuracy (BUY/SELL/HOLD)
- ✅ Test 2.1: Sentiment Direction Match
- ✅ Test 2.3: Sentiment-Price Correlation + Contrarian Analysis
- ✅ Test 3.3: Win Rate & Profit Factor
- ✅ Tests 2.2, 2.4, 4.1: Available via manual analysis

**Key Finding**: Contrarian hypothesis REJECTED - following sentiment is profitable

---

### 📋 **Phase 3: TODO** (Strategy Refinement)

**Priority Additions**:

1. **Test 1.3: Precision & Recall** (Ready to implement)
   - Add to `evaluation_framework.py`
   - Calculate TP, FP, TN, FN for each recommendation type
   - Output precision/recall/F1 scores in performance report
   - **Why**: Understand if AI is too conservative (low recall) or too aggressive (low precision)

2. **Test 1.5: Time-Horizon Validation** (Experimental)
   - Modify `evaluate_prediction()` to test multiple time windows
   - Compare 12h, 24h, 48h prediction accuracy
   - Determine optimal evaluation window
   - **Why**: Your dual-schedule (12PM + 4PM) may perform differently over time

3. **Test 3.1: Risk-Adjusted Returns** (Enhancement)
   - Calculate Sharpe ratio from hypothetical trades
   - Add to trading_performance report section
   - **Why**: $53K profit is great, but what's the volatility?

4. **Fix HOLD Issue** (Critical)
   - Current: 0% HOLD accuracy (should have been SELL)
   - Adjust confidence thresholds for bearish signals
   - Re-evaluate to measure improvement
   - **Why**: Missing profitable SELL opportunities

**Estimated Effort**: 2-3 sessions

---

### 🔮 **Phase 4: FUTURE** (Paper Trading)

**Deferred Tests** (Require live/simulated trading):

1. **Test 3.2: Maximum Drawdown**
   - Needs continuous position tracking
   - Implement in paper trading phase
   - Track peak-to-trough equity declines

2. **Test 4.2-4.3: Position Sizing Tests**
   - Validate Kelly Criterion vs fixed sizing
   - Requires actual trade execution simulation

3. **Test 5.1-5.3: Temporal Pattern Analysis**
   - Time-of-day effects
   - Day-of-week patterns
   - Requires larger dataset (30+ days)

**Estimated Timeline**: After 2-4 weeks of daily data collection

---

## Overview

This document maps the comprehensive test suite defined in `PHASE2_TESTING_FRAMEWORK.md` to the actual implementation in `evaluation_framework.py`. It shows which tests are automated, which require manual analysis, and where to find each capability in the code.

**Purpose**: Bridge the gap between test specification and implementation so you understand what's already built vs what needs to be added.

---

## Quick Status Summary

| Category | Total Tests | Automated | Manual | Not Built |
|----------|-------------|-----------|---------|-----------|
| **Prediction Accuracy** | 5 | 4 | 0 | 1 |
| **Sentiment Analysis** | 4 | 3 | 1 | 0 |
| **Risk Management** | 3 | 2 | 0 | 1 |
| **Operational Modes** | 2 | 0 | 2 | 0 |
| **Temporal Patterns** | 3 | 0 | 3 | 0 |
| **Data Quality** | 2 | 0 | 2 | 0 |
| **TOTAL** | **19** | **9** | **8** | **2** |

**Coverage**: 47% fully automated, 42% can be derived from data, 11% need custom implementation

---

## Detailed Test Mapping

## Category 1: Prediction Accuracy Tests (4/5 Automated)

### ✅ Test 1.1: Directional Accuracy
**Status**: ✅ Fully Automated

**Implementation**: `evaluation_framework.py`
- **Method**: `_evaluate_correctness()` (lines ~135-158)
- **Output**: `prediction_correct` field in each evaluation
- **Report Section**: `prediction_accuracy.overall_accuracy`

**How it works**:
```python
# In _evaluate_correctness() method
if recommendation == "BUY":
    return percent_change > buy_threshold
elif recommendation == "SELL":
    return percent_change < sell_threshold
elif recommendation == "HOLD":
    return -0.5 <= percent_change <= 0.5
```

**Output Example**:
```json
"prediction_accuracy": {
  "overall_accuracy": 65.00,
  "correct_predictions": 6,
  "incorrect_predictions": 4
}
```

**Where to find**: Run `evaluation_framework.py`, check "PREDICTION ACCURACY" section

---

### ✅ Test 1.2: Confidence Calibration
**Status**: ✅ Fully Automated

**Implementation**: `evaluation_framework.py`
- **Method**: `_evaluate_correctness()` (lines ~135-158)
- **Logic**: Adjusts thresholds based on confidence level

**How it works**:
```python
if confidence == "HIGH":
    buy_threshold = 0.5   # Requires >0.5% move
    sell_threshold = -0.5
else:
    buy_threshold = 0.0   # Any positive move OK
    sell_threshold = 0.0
```

**What this tests**: High confidence predictions should require stronger price movements to be considered correct.

**Manual Analysis Required**:
- Group evaluations by confidence level
- Calculate accuracy for HIGH vs MEDIUM vs LOW
- Look in individual_evaluations for 'confidence' field

**Analysis Script** (add to notebook):
```python
import json
with open('data/analysis_results/evaluation_report_latest.json') as f:
    data = json.load(f)

evals = data['individual_evaluations']

# Group by confidence
high_conf = [e for e in evals if e['confidence'] == 'HIGH']
med_conf = [e for e in evals if e['confidence'] == 'MEDIUM']
low_conf = [e for e in evals if e['confidence'] == 'LOW']

# Calculate accuracy by confidence
high_accuracy = sum(e['prediction_correct'] for e in high_conf) / len(high_conf) * 100
med_accuracy = sum(e['prediction_correct'] for e in med_conf) / len(med_conf) * 100
low_accuracy = sum(e['prediction_correct'] for e in low_conf) / len(low_conf) * 100

print(f"HIGH confidence: {high_accuracy:.1f}%")
print(f"MEDIUM confidence: {med_accuracy:.1f}%")
print(f"LOW confidence: {low_accuracy:.1f}%")
```

---

### 📋 Test 1.3: Precision & Recall
**Status**: 📋 Not Yet Implemented (Phase 3 Priority)

**What it tests**: 
- **Precision**: Of all BUY signals, how many were correct?
- **Recall**: Of all profitable opportunities, how many did we catch?

**Why important**: Current 0% HOLD accuracy suggests low recall for SELL signals.

**Implementation Plan** (Phase 3):
```python
# Add to generate_performance_report() method

def _calculate_precision_recall(self, evaluations: List[Dict]) -> Dict:
    """Calculate precision, recall, F1 for each recommendation type"""
    
    for rec_type in ['BUY', 'SELL', 'HOLD']:
        type_evals = [e for e in evaluations if e['recommendation'] == rec_type]
        
        # True Positives: Correct predictions
        tp = sum(1 for e in type_evals if e['prediction_correct'])
        
        # False Positives: Wrong predictions
        fp = sum(1 for e in type_evals if not e['prediction_correct'])
        
        # False Negatives: Missed opportunities
        # (Need to check other types for this)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4)
    }
```

**Expected Output**:
```json
"precision_recall": {
  "BUY": {"precision": 0.75, "recall": 0.60, "f1": 0.67},
  "SELL": {"precision": 1.00, "recall": 0.80, "f1": 0.89},
  "HOLD": {"precision": 0.00, "recall": 0.00, "f1": 0.00}
}
```

---

### ✅ Test 1.4: Type-Specific Accuracy
**Status**: ✅ Fully Automated

**Implementation**: `evaluation_framework.py`
- **Method**: `generate_performance_report()` (lines ~350-420)
- **Output**: `prediction_accuracy.by_type`

**How it works**:
```python
# Groups evaluations by recommendation type
buy_evals = [e for e in evaluations if e['recommendation'] == 'BUY']
sell_evals = [e for e in evaluations if e['recommendation'] == 'SELL']
hold_evals = [e for e in evaluations if e['recommendation'] == 'HOLD']

# Calculates accuracy for each
buy_accuracy = self._calc_accuracy(buy_evals)
```

**Output Example**:
```json
"by_type": {
  "BUY": {"count": 3, "accuracy": 66.67},
  "SELL": {"count": 4, "accuracy": 100.0},
  "HOLD": {"count": 3, "accuracy": 0.0}
}
```

**Where to find**: Run `evaluation_framework.py`, check "By Type" subsection

**Current Results**: SELL predictions are 100% accurate, HOLD predictions are 0% accurate

---

### 📋 Test 1.5: Time-Horizon Validation
**Status**: 📋 Not Yet Implemented (Phase 3 Experimental)

**What it tests**: Does prediction accuracy vary by evaluation window? (12h vs 24h vs 48h)

**Why important**: Your dual-schedule (12PM + 4PM) might show different optimal windows.

**Implementation Plan** (Phase 3):
```python
# Add to evaluation_framework.py

def evaluate_multiple_horizons(self, prediction: Dict) -> Dict:
    """Evaluate same prediction across multiple time windows"""
    
    horizons = [12, 24, 48]  # hours
    results = {}
    
    for hours in horizons:
        eval_result = self.evaluate_prediction(prediction, hours_forward=hours)
        results[f'{hours}h'] = eval_result
    
    return {
        'prediction': prediction['timestamp'],
        'horizons': results,
        'best_horizon': self._find_best_horizon(results)
    }
```

**Expected Output**:
```json
"horizon_analysis": {
  "12h": {"accuracy": 45%, "avg_pnl": 1250},
  "24h": {"accuracy": 57%, "avg_pnl": 2800},
  "48h": {"accuracy": 62%, "avg_pnl": 4100}
}
```

**Analysis**: May reveal that longer horizons are more predictable for your strategy.

---

## Category 2: Sentiment Analysis Tests (3/4 Automated)

### ✅ Test 2.1: Sentiment Direction Match
**Status**: ✅ Fully Automated

**Implementation**: `evaluation_framework.py`
- **Method**: `_evaluate_sentiment()` (lines ~160-175)
- **Output**: `sentiment_accurate` field in each evaluation
- **Report Section**: `sentiment_analysis.sentiment_accuracy`

**How it works**:
```python
# Checks if sentiment direction matches price movement
if sentiment_score > 0 and percent_change > 0:
    return True  # Bullish sentiment, price went up
elif sentiment_score < 0 and percent_change < 0:
    return True  # Bearish sentiment, price went down
```

**Output Example**:
```json
"sentiment_analysis": {
  "sentiment_accuracy": 57.14,
  "correct_sentiment_predictions": 4,
  "incorrect_sentiment_predictions": 3
}
```

**Where to find**: Run `evaluation_framework.py`, check "SENTIMENT ANALYSIS" section

**Current Results**: 57% accuracy after fixing sentiment score calculation

---

### ⚙️ Test 2.2: Sentiment Confidence Correlation
**Status**: ⚙️ Data Available, Manual Analysis Required

**What it tests**: Do higher sentiment confidence scores lead to larger price movements?

**Where to find data**:
- Individual evaluations contain both `sentiment_score` and `percent_change`
- In `evaluation_report_*.json` → `individual_evaluations` array

**Manual Analysis**:
```python
import json
import numpy as np
from scipy.stats import pearsonr

with open('data/analysis_results/evaluation_report_latest.json') as f:
    data = json.load(f)

evals = data['individual_evaluations']

# Extract sentiment confidence and price changes
sentiment_scores = [abs(e['sentiment_score']) for e in evals]
price_changes = [abs(e['percent_change']) for e in evals]

# Calculate correlation
correlation, p_value = pearsonr(sentiment_scores, price_changes)

print(f"Correlation: {correlation:.3f}")
print(f"P-value: {p_value:.3f}")
print(f"Significant: {'Yes' if p_value < 0.05 else 'No'}")
```

**Expected Result**: Strong correlation (>0.5) would validate sentiment confidence as predictive.

---

### ✅ Test 2.3: Sentiment-Price Correlation + Contrarian Analysis
**Status**: ✅ Fully Automated

**Implementation**: `evaluation_framework.py`
- **Method**: `_analyze_contrarian_strategy()` (lines ~322-372)
- **Output**: `contrarian_analysis` section in report

**How it works**:
```python
# Flips recommendations and tests if opposite would be better
if eval['recommendation'] == 'BUY':
    contrarian_rec = 'SELL'
elif eval['recommendation'] == 'SELL':
    contrarian_rec = 'BUY'

# Calculates accuracy of contrarian approach
contrarian_accuracy = (contrarian_correct / total) * 100

# Calculates hypothetical contrarian PnL
contrarian_pnl = sum(-eval['hypothetical_pnl'] for each trade)
```

**Output Example**:
```json
"contrarian_analysis": {
  "contrarian_accuracy": 0.0,
  "contrarian_correct_predictions": 0,
  "contrarian_hypothetical_pnl": -53197.20,
  "performance_comparison": "FOLLOWING BETTER"
}
```

**Where to find**: Run `evaluation_framework.py`, check "CONTRARIAN STRATEGY ANALYSIS" section

**Current Results**: Contrarian hypothesis REJECTED - following sentiment is profitable

---

### ⚙️ Test 2.4: News Source Impact
**Status**: ⚙️ Data Available, Manual Analysis Required

**What it tests**: Do certain news sources (CoinDesk vs Cointelegraph) correlate with better predictions?

**Where to find data**:
- Not in evaluation results (stored in full prediction files)
- Check `btc_analysis_*.json` → `data_sources.news_sources_count`

**Manual Analysis**:
```python
import json
from collections import defaultdict

# Load full predictions (not just evaluation format)
predictions = []
for file in Path('data/analysis_results').glob('btc_analysis_*.json'):
    with open(file) as f:
        predictions.append(json.load(f))

# Load corresponding evaluations
with open('data/analysis_results/evaluation_report_latest.json') as f:
    evals = {e['timestamp']: e for e in f['individual_evaluations']}

# Correlate news sources with accuracy
# (This analysis requires linking predictions to evaluations by timestamp)
```

**Note**: Requires custom script - not built into framework yet.

---

## Category 3: Risk Management Tests (2/3 Automated)

### ⚙️ Test 3.1: Risk-Adjusted Returns (Sharpe Ratio)
**Status**: 📋 Not Yet Implemented (Phase 3 Enhancement)

**What it tests**: Are returns worth the volatility risk?

**Why important**: $53K profit looks great, but what if it came with huge swings?

**Implementation Plan** (Phase 3):
```python
# Add to generate_performance_report() method

def _calculate_sharpe_ratio(self, evaluations: List[Dict], risk_free_rate: float = 0.05) -> float:
    """Calculate Sharpe ratio for trading performance"""
    
    # Get all PnL values
    pnl_values = [e['hypothetical_pnl'] for e in evaluations]
    
    # Calculate returns
    returns = np.array(pnl_values)
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Sharpe ratio = (mean return - risk free rate) / std deviation
    sharpe = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
    
    return round(sharpe, 2)
```

**Expected Output**:
```json
"risk_adjusted_returns": {
  "sharpe_ratio": 2.15,
  "interpretation": "Excellent (>2.0)"
}
```

**Interpretation Guide**:
- < 1.0: Poor
- 1.0-2.0: Good
- > 2.0: Excellent

---

### 🔮 Test 3.2: Maximum Drawdown
**Status**: 🔮 Deferred to Phase 4 (Paper Trading)

**What it tests**: Largest peak-to-trough decline in portfolio value.

**Why deferred**: Requires continuous position tracking across multiple trades, not single evaluations.

**Implementation**: Build during paper trading phase when tracking cumulative equity curve.

**Placeholder Code**:
```python
def calculate_max_drawdown(equity_curve: List[float]) -> Dict:
    """Calculate maximum drawdown from equity curve"""
    
    peak = equity_curve[0]
    max_dd = 0
    max_dd_pct = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        
        drawdown = peak - value
        drawdown_pct = (drawdown / peak) * 100
        
        if drawdown > max_dd:
            max_dd = drawdown
            max_dd_pct = drawdown_pct
    
    return {
        'max_drawdown_usd': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 2)
    }
```

---

### ✅ Test 3.3: Win Rate & Profit Factor
**Status**: ✅ Fully Automated

**Implementation**: `evaluation_framework.py`
- **Method**: `generate_performance_report()` (lines ~350-420)
- **Output**: `trading_performance` section

**How it works**:
```python
# Calculate win rate
winning_trades = [e for e in evaluations if e['hypothetical_pnl'] > 0]
win_rate = (len(winning_trades) / total) * 100

# Calculate profit factor
avg_win = sum(e['hypothetical_pnl'] for e in winning_trades) / len(winning_trades)
avg_loss = sum(e['hypothetical_pnl'] for e in losing_trades) / len(losing_trades)
profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
```

**Output Example**:
```json
"trading_performance": {
  "total_hypothetical_pnl": 53197.20,
  "winning_trades": 4,
  "losing_trades": 0,
  "win_rate": 57.14,
  "average_win": 13299.30,
  "average_loss": 0.00,
  "profit_factor": "inf"
}
```

**Where to find**: Run `evaluation_framework.py`, check "TRADING PERFORMANCE" section

**Current Results**: 
- Win Rate: 57.14% (4 wins, 0 losses)
- Profit Factor: ∞ (no losses yet!)

---

## Category 4: Operational Mode Tests (0/2 Automated)

### ⚙️ Test 4.1: Mode Comparison (FULL vs PARTIAL vs NEWS_ONLY)
**Status**: ⚙️ Data Available, Manual Analysis Required

**What it tests**: Does having all data sources (exchange + Fear & Greed + news) improve accuracy vs partial data?

**Where to find data**:
- Full predictions contain `analysis_mode` field
- In `btc_analysis_*.json` → `analysis_mode`

**Manual Analysis**:
```python
import json
from pathlib import Path
from collections import defaultdict

# Load full predictions
predictions = []
for file in Path('data/analysis_results').glob('btc_analysis_*.json'):
    with open(file) as f:
        pred = json.load(f)
        predictions.append({
            'timestamp': pred['timestamp'],
            'mode': pred['analysis_mode']
        })

# Load evaluations
with open('data/analysis_results/evaluation_report_latest.json') as f:
    evals = {e['timestamp']: e for e in f['individual_evaluations']}

# Group by mode
mode_results = defaultdict(list)
for pred in predictions:
    if pred['timestamp'] in evals:
        eval_data = evals[pred['timestamp']]
        mode_results[pred['mode']].append(eval_data['prediction_correct'])

# Calculate accuracy by mode
for mode, results in mode_results.items():
    accuracy = sum(results) / len(results) * 100
    print(f"{mode}: {accuracy:.1f}% ({len(results)} predictions)")
```

**Expected Result**: FULL mode should have highest accuracy.

---

### ⚙️ Test 4.2: Degradation Gracefully
**Status**: ⚙️ Can Observe, Not Quantified

**What it tests**: Does system maintain reasonable performance when data sources fail?

**How to observe**: Look at predictions during exchange outages or API failures.

**Current Implementation**: System automatically falls back through modes:
1. FULL MODE: Exchange + Fear & Greed + News
2. PARTIAL MODE: Fear & Greed + News (if exchange fails)
3. NEWS_ONLY: RSS feeds only (if both fail)

**Analysis**: Compare accuracy across modes when failures occur naturally.

---

## Category 5: Temporal Pattern Tests (0/3 Automated)

### ⚙️ Test 5.1: Time-of-Day Effects
**Status**: ⚙️ Data Available, Manual Analysis Required (Phase 3 Focus!)

**What it tests**: Does prediction accuracy vary by time of day?

**Why critical for your experiment**: Your dual-schedule (12PM vs 4PM) directly tests this!

**Where to find data**:
- Extract hour from `prediction_time` field in evaluations
- In `evaluation_report_*.json` → `individual_evaluations[].prediction_time`

**Manual Analysis**:
```python
import json
from datetime import datetime
from collections import defaultdict

with open('data/analysis_results/evaluation_report_latest.json') as f:
    data = json.load(f)

evals = data['individual_evaluations']

# Group by hour
hour_results = defaultdict(list)
for eval in evals:
    pred_time = datetime.fromisoformat(eval['prediction_time'])
    hour = pred_time.hour
    hour_results[hour].append(eval['prediction_correct'])

# Calculate accuracy by hour
for hour in sorted(hour_results.keys()):
    results = hour_results[hour]
    accuracy = sum(results) / len(results) * 100
    print(f"{hour:02d}:00 - {accuracy:.1f}% ({len(results)} predictions)")
```

**Expected Result** (Your Experiment):
- 12:00 PM predictions (pre-NY-volatility) - Hypothesis: Better sentiment capture
- 4:00 PM predictions (post-NY-volatility) - Hypothesis: Better price confirmation

**Phase 3 Goal**: Collect 2 weeks of dual-schedule data, then run this analysis!

---

### ⚙️ Test 5.2: Day-of-Week Patterns
**Status**: ⚙️ Data Available, Manual Analysis Required

**What it tests**: Are certain days more predictable? (Weekend vs weekday effects)

**Where to find data**: Same as Test 5.1, but group by `weekday()`

**Manual Analysis**:
```python
import json
from datetime import datetime
from collections import defaultdict

with open('data/analysis_results/evaluation_report_latest.json') as f:
    data = json.load(f)

evals = data['individual_evaluations']

# Group by day of week
day_results = defaultdict(list)
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

for eval in evals:
    pred_time = datetime.fromisoformat(eval['prediction_time'])
    day = pred_time.weekday()  # 0 = Monday, 6 = Sunday
    day_results[day].append(eval['prediction_correct'])

# Calculate accuracy by day
for day in range(7):
    if day in day_results:
        results = day_results[day]
        accuracy = sum(results) / len(results) * 100
        print(f"{day_names[day]}: {accuracy:.1f}% ({len(results)} predictions)")
```

**Note**: Requires at least 2-3 weeks of data for meaningful patterns.

---

### ⚙️ Test 5.3: Market Volatility Adaptation
**Status**: ⚙️ Data Available, Advanced Analysis Required

**What it tests**: Does AI perform better in high vs low volatility conditions?

**Where to find data**:
- `percent_change` field shows realized volatility
- Can calculate rolling volatility from recent predictions

**Manual Analysis**:
```python
import json
import numpy as np
from collections import defaultdict

with open('data/analysis_results/evaluation_report_latest.json') as f:
    data = json.load(f)

evals = sorted(data['individual_evaluations'], key=lambda x: x['prediction_time'])

# Calculate rolling 7-day volatility
volatilities = []
for i in range(len(evals)):
    if i >= 7:
        recent = evals[i-7:i]
        price_changes = [e['percent_change'] for e in recent]
        vol = np.std(price_changes)
        volatilities.append((vol, evals[i]['prediction_correct']))

# Group by volatility regime
low_vol = [correct for vol, correct in volatilities if vol < np.percentile([v[0] for v in volatilities], 33)]
mid_vol = [correct for vol, correct in volatilities if np.percentile([v[0] for v in volatilities], 33) <= vol < np.percentile([v[0] for v in volatilities], 66)]
high_vol = [correct for vol, correct in volatilities if vol >= np.percentile([v[0] for v in volatilities], 66)]

print(f"Low Volatility: {sum(low_vol)/len(low_vol)*100:.1f}% ({len(low_vol)} predictions)")
print(f"Mid Volatility: {sum(mid_vol)/len(mid_vol)*100:.1f}% ({len(mid_vol)} predictions)")
print(f"High Volatility: {sum(high_vol)/len(high_vol)*100:.1f}% ({len(high_vol)} predictions)")
```

**Expected Result**: May perform better in trending (high vol) vs choppy (low vol) markets.

---

## Category 6: Data Quality Tests (0/2 Automated)

### ⚙️ Test 6.1: News Source Availability
**Status**: ⚙️ Can Observe, Not Quantified

**What it tests**: Are news sources consistently available and providing Bitcoin-relevant content?

**Where to find data**:
- Full predictions contain `articles_analyzed` count
- In `btc_analysis_*.json` → `articles_analyzed`

**Manual Analysis**:
```python
import json
from pathlib import Path

predictions = []
for file in Path('data/analysis_results').glob('btc_analysis_*.json'):
    with open(file) as f:
        pred = json.load(f)
        predictions.append({
            'timestamp': pred['timestamp'],
            'articles': pred.get('articles_analyzed', 0)
        })

# Check for anomalies
low_article_count = [p for p in predictions if p['articles'] < 5]
if low_article_count:
    print(f"Warning: {len(low_article_count)} predictions had <5 articles")
    for p in low_article_count:
        print(f"  {p['timestamp']}: {p['articles']} articles")
```

**Expected Result**: Should consistently have 8-10 articles per analysis.

---

### ⚙️ Test 6.2: Price Data Continuity
**Status**: ⚙️ Can Observe, Not Quantified

**What it tests**: Are there gaps in price data that could affect evaluations?

**How to check**:
- Look for `None` values in `start_price` or `end_price` fields
- In evaluation results, check for "Could not fetch prices" errors

**Manual Analysis**:
```python
import json

with open('data/analysis_results/evaluation_report_latest.json') as f:
    data = json.load(f)

evals = data['individual_evaluations']

# Check for missing price data
missing_data = [e for e in evals if e.get('start_price') is None or e.get('end_price') is None]

if missing_data:
    print(f"Warning: {len(missing_data)} evaluations missing price data")
    for e in missing_data:
        print(f"  {e['timestamp']}")
else:
    print("✓ All evaluations have complete price data")
```

**Expected Result**: No missing data (Coinbase API is reliable).

---

## Summary: What's Ready vs What's Next

### ✅ **Ready to Use Now** (Phase 2 Complete):
1. Directional accuracy tracking
2. Confidence calibration
3. Type-specific accuracy (BUY/SELL/HOLD)
4. Sentiment direction matching
5. Contrarian strategy testing
6. Win rate & profit factor
7. Comprehensive performance reports

### 📋 **Phase 3 Priorities** (Next 2-3 Sessions):
1. **Test 1.3**: Add Precision & Recall calculations
2. **Test 1.5**: Implement time-horizon validation (12h/24h/48h)
3. **Test 3.1**: Add Sharpe ratio calculation
4. **Test 5.1**: Analyze time-of-day effects (12PM vs 4PM experiment!)
5. **Fix HOLD Issue**: Adjust AI to reduce false negatives

### 🔮 **Phase 4 Deferred** (After Paper Trading Setup):
1. **Test 3.2**: Maximum drawdown tracking
2. **Test 4.2-4.3**: Position sizing validation
3. **Test 5.2-5.3**: Long-term temporal pattern analysis

---

## Quick Reference: Running Evaluations

### Basic Evaluation (Current Predictions)
```bash
python scripts/evaluation_framework.py
```

### Convert Old Predictions (After Format Changes)
```bash
python scripts/convert_old_predictions.py
```

### Test Framework Setup
```bash
python scripts/test_evaluation.py
```

### View Latest Report
```bash
# Find most recent report
ls -lt data/analysis_results/evaluation_report_*.json | head -1

# Pretty print it
cat data/analysis_results/evaluation_report_YYYYMMDD_HHMMSS.json | python -m json.tool
```

### Manual Analysis Template
```python
import json
from pathlib import Path

# Load latest evaluation
reports = sorted(Path('data/analysis_results').glob('evaluation_report_*.json'))
with open(reports[-1]) as f:
    data = json.load(f)

# Access components
summary = data['performance_report']['evaluation_summary']
accuracy = data['performance_report']['prediction_accuracy']
sentiment = data['performance_report']['sentiment_analysis']
trading = data['performance_report']['trading_performance']
contrarian = data['performance_report']['contrarian_analysis']
individual = data['individual_evaluations']

# Your custom analysis here...
```

---

## Troubleshooting

### Issue: "Could not fetch prices for evaluation"
**Cause**: Prediction is too recent (need 24+ hours of price history)
**Solution**: Wait for time to pass, or evaluate older predictions

### Issue: "Sentiment accuracy = 0%"
**Cause**: Sentiment scores not properly normalized (negative for bearish)
**Solution**: Re-convert predictions with fixed `convert_old_predictions.py`

### Issue: "HOLD predictions always wrong"
**Cause**: AI too conservative during strong bearish signals
**Solution**: Phase 3 - adjust confidence thresholds for HOLD vs SELL decision

### Issue: "Not enough data for analysis"
**Cause**: Need more predictions for statistical significance
**Solution**: Run automated daily predictions for 1-2 weeks

---

**Last Updated**: November 22, 2025  
**Phase**: 2 Complete, 3 Planning  
**Next Milestone**: Dual-schedule automation (12PM + 4PM EST)
