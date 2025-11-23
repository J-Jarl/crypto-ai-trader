# Phase 2 Testing Framework
## Comprehensive Test Suite for AI Trading System Evaluation

**Project**: Crypto AI Trading System  
**Phase**: 2 - Evaluation & Measurement  
**Objective**: Systematically measure AI prediction accuracy before optimization  
**Last Updated**: November 22, 2024

---

## Testing Philosophy

> **"Measure before optimizing. Make data-driven decisions, not guesses."**

Phase 2 focuses on building evaluation infrastructure to objectively assess the AI's performance against historical market data. This document outlines all tests needed to determine:
1. Is the AI profitable?
2. Does the contrarian hypothesis work?
3. What conditions favor successful predictions?
4. Where should we optimize in Phase 3?

---

## Test Categories Overview

| Category | Tests | Priority | Purpose |
|----------|-------|----------|---------|
| **1. Prediction Accuracy** | 5 tests | 🔴 Critical | Core profitability validation |
| **2. Sentiment Analysis** | 4 tests | 🔴 Critical | Hypothesis testing |
| **3. Risk Management** | 3 tests | 🟡 High | Capital preservation |
| **4. Operational Modes** | 2 tests | 🟢 Medium | System reliability |
| **5. Temporal Patterns** | 3 tests | 🟢 Medium | Optimization opportunities |
| **6. Data Quality** | 2 tests | 🟡 High | Foundation validation |

**Total Tests**: 19

---

## Category 1: Prediction Accuracy Tests

### Test 1.1: Directional Accuracy
**Purpose**: Validate if AI predicts price direction correctly

**Methodology**:
```python
# For each historical recommendation:
# - AI said BUY → Did price increase in next N hours?
# - AI said SELL → Did price decrease in next N hours?
# - AI said HOLD → Did price stay relatively flat (±2%)?
```

**Metrics to Track**:
- Overall accuracy % (correct predictions / total predictions)
- Accuracy by signal type (BUY/SELL/HOLD separately)
- Confusion matrix showing prediction vs actual outcomes
- Time horizons: 1-hour, 4-hour, 24-hour

**Success Criteria**:
- ✅ >55% accuracy = Better than random chance
- ✅ >60% accuracy = Potentially profitable after fees
- ✅ >65% accuracy = Strong predictive power

**Output Location**: `data/analysis_results/test1.1_directional_accuracy.json`

---

### Test 1.2: Entry/Exit Timing
**Purpose**: Assess if AI catches optimal entry/exit windows

**Methodology**:
```python
# When AI was directionally correct:
# - How long until price reached optimal entry?
# - What % of the move did we capture?
# - Did we enter too early (drawdown) or too late (missed move)?
```

**Metrics to Track**:
- Average time to optimal entry (hours)
- Percentage of move captured (entry to peak/trough)
- Entry efficiency score (0-100)
- Missed opportunity cost ($ left on table)

**Success Criteria**:
- ✅ Catches 70%+ of moves within 4-hour window
- ✅ Entry within ±5% of optimal price
- ✅ Average opportunity cost <10% of total move

**Output Location**: `data/analysis_results/test1.2_entry_exit_timing.json`

---

### Test 1.3: Position Sizing Accuracy
**Purpose**: Determine if recommended position sizes match actual risk/reward

**Methodology**:
```python
# Analyze recommended position sizes:
# - Too aggressive for actual volatility?
# - Too conservative (missed upside)?
# - Compare to Kelly Criterion optimal sizing
```

**Metrics to Track**:
- Average position size vs Kelly optimal
- Risk-adjusted returns (Sharpe ratio)
- Position size vs actual volatility correlation
- Overexposure events (position too large for move)

**Success Criteria**:
- ✅ Position sizes within 10% of mathematically optimal
- ✅ Sharpe ratio >1.0
- ✅ Zero catastrophic overexposure events

**Output Location**: `data/analysis_results/test1.3_position_sizing.json`

---

### Test 1.4: Stop Loss Effectiveness
**Purpose**: Validate stop losses protect capital without premature exits

**Methodology**:
```python
# For each trade with stop loss:
# - Did stop loss trigger?
# - If yes, did price recover within 24h? (false stop-out)
# - Average loss when stop loss hit
# - Compare: stopped out vs letting it run
```

**Metrics to Track**:
- Stop loss hit rate (% of trades)
- False stop-out rate (price recovered shortly after)
- Average loss when SL triggers
- Capital saved by stop losses vs not using them

**Success Criteria**:
- ✅ <20% false stop-out rate
- ✅ Stop losses save capital on 80%+ of hits
- ✅ Average loss when hit = within expected range

**Output Location**: `data/analysis_results/test1.4_stop_loss.json`

---

### Test 1.5: Take Profit Realism
**Purpose**: Assess if take profit targets are achievable and optimal

**Methodology**:
```python
# For each trade with take profit:
# - Did price reach TP level?
# - If yes, did it continue higher? (missed upside)
# - If no, how close did it get? (TP too aggressive?)
```

**Metrics to Track**:
- Take profit hit rate (% of trades reaching TP)
- Missed upside (price continued 10%+ past TP)
- Average distance from TP when trade ends
- TP level vs actual peak analysis

**Success Criteria**:
- ✅ 40-60% TP hit rate (balance realistic/ambitious)
- ✅ Missed upside <15% of winning trades
- ✅ Failed TPs were within 20% of target

**Output Location**: `data/analysis_results/test1.5_take_profit.json`

---

## Category 2: Sentiment Analysis Quality Tests

### Test 2.1: Sentiment Extraction Accuracy
**Purpose**: Verify AI correctly interprets bullish/bearish sentiment from text

**Methodology**:
```python
# Create labeled test set:
# 1. Collect 100 news headlines (33 bullish, 33 bearish, 34 neutral)
# 2. Manually label each
# 3. Run AI sentiment analysis
# 4. Compare AI labels vs human labels
```

**Metrics to Track**:
- Precision/Recall for each category (bullish/bearish/neutral)
- F1 score overall
- Inter-rater reliability (Cohen's Kappa)
- Common misclassification patterns

**Success Criteria**:
- ✅ >80% agreement with human labeling
- ✅ F1 score >0.75
- ✅ No systematic bias (consistently missing one category)

**Output Location**: `data/analysis_results/test2.1_sentiment_extraction.json`

---

### Test 2.2: Multi-Source Sentiment Correlation
**Purpose**: Determine if different data sources (RSS, Fear & Greed) agree

**Methodology**:
```python
# For 30 days of historical data:
# - Calculate RSS sentiment score (aggregate across feeds)
# - Record Fear & Greed Index values
# - Compute correlation coefficient
# - Identify divergence events and outcomes
```

**Metrics to Track**:
- Pearson correlation between RSS and F&G
- Divergence frequency (sources disagree significantly)
- Lead/lag analysis (which source predicts the other?)
- Outcome when sources disagree

**Success Criteria**:
- ✅ >0.6 correlation = sources generally agree
- ✅ When divergent, identify which source to trust
- ✅ Understand lead/lag dynamics (±4 hours)

**Output Location**: `data/analysis_results/test2.2_source_correlation.json`

---

### Test 2.3: Sentiment-Price Correlation (CRITICAL - CONTRARIAN HYPOTHESIS)
**Purpose**: Test if sentiment inversely correlates with price (contrarian strategy validation)

**Methodology**:
```python
# THE MAKE-OR-BREAK TEST FOR YOUR STRATEGY
# 
# For each data point:
# - Record Fear & Greed Index value
# - Record RSS sentiment aggregate
# - Measure price change in next 1h, 4h, 24h
# - Calculate correlation coefficient
#
# Positive correlation = following strategy works
# Negative correlation = contrarian strategy works
# Near zero = sentiment doesn't predict price
```

**Metrics to Track**:
- Correlation coefficient: sentiment vs price change
- Significance level (p-value)
- Optimal time lag for signal (1h, 4h, 24h?)
- Extreme values analysis:
  - When F&G = "Extreme Fear" (0-25), what happens?
  - When F&G = "Extreme Greed" (75-100), what happens?

**Success Criteria**:
- ✅ Negative correlation with p<0.05 = Contrarian strategy validated
- ✅ Identify optimal time lag for signal
- ✅ Extreme values show strongest inverse correlation

**Output Location**: `data/analysis_results/test2.3_CONTRARIAN_HYPOTHESIS.json`

**⚠️ CRITICAL DECISION POINT**: This test determines Phase 3 strategy direction.

---

### Test 2.4: News Recency Impact
**Purpose**: Determine if fresher news produces better predictions

**Methodology**:
```python
# Compare predictions made with:
# - 0-2 hour old news (fresh)
# - 2-6 hour old news (recent)
# - 6-12 hour old news (stale)
# - 12-24 hour old news (very stale)
```

**Metrics to Track**:
- Prediction accuracy by news age bucket
- Half-life of news relevance (when does accuracy degrade 50%?)
- Optimal news window (best accuracy range)

**Success Criteria**:
- ✅ Identify statistically significant accuracy drop-off point
- ✅ Define minimum acceptable news freshness
- ✅ Determine if news older than X hours should be excluded

**Output Location**: `data/analysis_results/test2.4_news_recency.json`

---

## Category 3: Risk Management Tests

### Test 3.1: Risk/Reward Ratio Accuracy
**Purpose**: Verify stated R/R ratios are actually achieved in practice

**Methodology**:
```python
# For each completed trade:
# - AI predicted R/R ratio (e.g., 3:1)
# - Actual R/R achieved
# - Calculate prediction error
# - Identify systematic bias (over/under estimation)
```

**Metrics to Track**:
- Predicted vs actual R/R scatter plot
- Mean absolute error (MAE)
- Systematic bias (consistently optimistic/pessimistic?)
- Accuracy by trade type (BUY vs SELL)

**Success Criteria**:
- ✅ Predicted R/R within 20% of actual
- ✅ No systematic bias >10%
- ✅ Calibration curve shows good alignment

**Output Location**: `data/analysis_results/test3.1_risk_reward_accuracy.json`

---

### Test 3.2: Maximum Drawdown
**Purpose**: Determine worst-case losing streak and recovery characteristics

**Methodology**:
```python
# Simulate following AI signals for 30+ days:
# - Track cumulative P&L
# - Identify maximum drawdown (peak to trough)
# - Measure time to recover to new highs
# - Analyze drawdown frequency and depth
```

**Metrics to Track**:
- Maximum drawdown % (worst peak-to-trough)
- Average drawdown depth
- Drawdown duration (how long underwater?)
- Recovery time to new equity highs
- Drawdown frequency (how often?)

**Success Criteria**:
- ✅ Max drawdown <15% (with 5% position sizing)
- ✅ Average recovery time <7 days
- ✅ No drawdowns lasting >14 days

**Output Location**: `data/analysis_results/test3.2_maximum_drawdown.json`

---

### Test 3.3: Win Rate vs Profit Factor
**Purpose**: Validate that wins are large enough to offset losses

**Methodology**:
```python
# Calculate comprehensive profitability metrics:
# - Win rate = (winning trades / total trades) * 100
# - Average win $ vs average loss $
# - Profit factor = gross wins / gross losses
# - Expected value per trade
```

**Metrics to Track**:
- Win rate %
- Average win size ($)
- Average loss size ($)
- Win/loss ratio
- Profit factor
- Expected value per trade
- Largest win vs largest loss

**Success Criteria**:
- ✅ Profit factor >1.5 (minimum for profitability)
- ✅ Win/loss ratio >1.5:1 OR win rate >55%
- ✅ Positive expected value per trade

**Output Location**: `data/analysis_results/test3.3_win_rate_profit_factor.json`

**⚠️ CRITICAL DECISION POINT**: If profit factor <1.5, strategy is not profitable.

---

## Category 4: Operational Mode Tests

### Test 4.1: Degraded Mode Performance
**Purpose**: Assess how well AI performs with missing data sources

**Methodology**:
```python
# Compare accuracy across operating modes:
# - 🟢 FULL MODE: Exchange + F&G + News (all sources)
# - 🟡 PARTIAL MODE: F&G + News (no exchange data)
# - 🟠 NEWS ONLY MODE: RSS feeds only (minimal data)
#
# Test on same historical periods with artificially removed data
```

**Metrics to Track**:
- Accuracy % by mode
- Confidence level adjustments by mode
- Prediction quality degradation curve
- Minimum viable mode threshold

**Success Criteria**:
- ✅ Define minimum acceptable mode for predictions
- ✅ Document accuracy drop per missing source
- ✅ Create decision rule: "Skip prediction if mode < X"

**Output Location**: `data/analysis_results/test4.1_degraded_mode.json`

---

### Test 4.2: Exchange API Reliability
**Purpose**: Measure reliability of different exchange APIs

**Methodology**:
```python
# Track over 7-14 days of live operation:
# - API call success rate (Coinbase, Kraken)
# - Response times
# - Error types and frequency
# - Fallback trigger events
```

**Metrics to Track**:
- Uptime % by exchange (Coinbase vs Kraken)
- Average response time (ms)
- Error 451 frequency (geo-blocking)
- Time-of-day reliability patterns
- Weekend vs weekday differences

**Success Criteria**:
- ✅ Identify most reliable exchange (>99% uptime)
- ✅ Document optimal fallback sequence
- ✅ Set appropriate timeout thresholds

**Output Location**: `data/analysis_results/test4.2_api_reliability.json`

---

## Category 5: Temporal Pattern Tests

### Test 5.1: Time-of-Day Performance
**Purpose**: Identify if prediction quality varies by time/trading session

**Methodology**:
```python
# Segment historical predictions by time (UTC):
# - 00:00-08:00 UTC (Asia session)
# - 08:00-13:00 UTC (Europe session)
# - 13:00-21:00 UTC (US session)
# - 21:00-00:00 UTC (After hours)
#
# Also test:
# - Weekday vs weekend
# - First hour of each session (high volatility)
```

**Metrics to Track**:
- Accuracy % by time bucket
- Accuracy by day of week
- Volume correlation with accuracy
- Volatility correlation with accuracy

**Success Criteria**:
- ✅ Identify optimal trading windows (>60% accuracy)
- ✅ Identify avoid times (<50% accuracy)
- ✅ Create time-based filters for strategy

**Output Location**: `data/analysis_results/test5.1_time_of_day.json`

---

### Test 5.2: Volatility Regime Performance
**Purpose**: Determine if AI needs different strategies for different volatility levels

**Methodology**:
```python
# Classify days by volatility:
# - Low volatility: Daily range <2%
# - Medium volatility: Daily range 2-5%
# - High volatility: Daily range >5%
#
# Test AI performance in each regime
```

**Metrics to Track**:
- Accuracy by volatility regime
- Optimal position sizing by regime
- Stop loss effectiveness by regime
- False signal rate in low volatility (chop)

**Success Criteria**:
- ✅ Separate strategies for high/low volatility
- ✅ Identify when to stay out (low volatility traps)
- ✅ Adjust position sizing rules by regime

**Output Location**: `data/analysis_results/test5.2_volatility_regime.json`

---

### Test 5.3: Trending vs Range-Bound Performance
**Purpose**: Assess if AI performs better in trends or ranges

**Methodology**:
```python
# Classify market conditions using ADX or similar:
# - Strong uptrend (ADX >25, +DI > -DI)
# - Strong downtrend (ADX >25, -DI > +DI)
# - Range-bound/choppy (ADX <20)
#
# Measure AI accuracy in each regime
```

**Metrics to Track**:
- Accuracy by market regime
- False signal rate in choppy markets
- Best performing signal type per regime
- Regime transition detection success

**Success Criteria**:
- ✅ Identify when to trade vs stay out
- ✅ Regime-specific strategy parameters
- ✅ Filter to avoid choppy markets

**Output Location**: `data/analysis_results/test5.3_market_regime.json`

---

## Category 6: Data Quality Tests

### Test 6.1: RSS Feed Quality
**Purpose**: Ensure RSS feeds return relevant, Bitcoin-focused content

**Methodology**:
```python
# For 7 days of RSS data:
# - Count total articles per feed
# - Manually label 50 articles: Bitcoin-relevant vs noise
# - Calculate relevance score per feed
# - Identify altcoin noise level
# - Check update frequency reliability
```

**Metrics to Track**:
- Bitcoin relevance % by feed
- Articles per day per feed
- Duplicate article rate
- Update reliability (expected vs actual)
- Feed-specific noise patterns

**Success Criteria**:
- ✅ >90% Bitcoin relevance (exclude feeds below this)
- ✅ All feeds update at least 5x per day
- ✅ <5% duplicate articles

**Output Location**: `data/analysis_results/test6.1_rss_quality.json`

---

### Test 6.2: Price Data Integrity
**Purpose**: Validate exchange data is clean and consistent

**Methodology**:
```python
# Check historical price data:
# - Gaps in data (missing timestamps)
# - Outliers (flash crashes, erroneous ticks)
# - Cross-exchange consistency (Coinbase vs Kraken)
# - Tick-by-tick vs OHLCV consistency
```

**Metrics to Track**:
- Data completeness % (no gaps)
- Outlier frequency (>3 std dev)
- Cross-exchange price spread (should be <0.5%)
- Timestamp accuracy

**Success Criteria**:
- ✅ <1% data gaps
- ✅ Outliers properly identified and handled
- ✅ Exchange prices within 0.5% of each other

**Output Location**: `data/analysis_results/test6.2_price_integrity.json`

---

## Test Execution Roadmap

### Week 1: Foundation & Hypothesis (Priority 1)
**Goal**: Validate data quality and test core contrarian hypothesis

- [ ] **Test 6.1**: RSS Feed Quality
- [ ] **Test 6.2**: Price Data Integrity  
- [ ] **Test 2.3**: Sentiment-Price Correlation (CONTRARIAN HYPOTHESIS) ⚠️
- [ ] **Test 1.1**: Directional Accuracy

**Deliverable**: Know if contrarian strategy has merit

---

### Week 2: Timing & Sizing (Priority 2)
**Goal**: Refine entry/exit mechanics and position sizing

- [ ] **Test 1.2**: Entry/Exit Timing
- [ ] **Test 1.3**: Position Sizing Accuracy
- [ ] **Test 3.1**: Risk/Reward Ratio Accuracy

**Deliverable**: Optimized "how much" and "when" parameters

---

### Week 3: Risk Management (Priority 3)
**Goal**: Ensure capital preservation and profitability

- [ ] **Test 1.4**: Stop Loss Effectiveness
- [ ] **Test 1.5**: Take Profit Realism
- [ ] **Test 3.2**: Maximum Drawdown
- [ ] **Test 3.3**: Win Rate vs Profit Factor ⚠️

**Deliverable**: Complete risk profile and profitability proof

---

### Week 4: Optimization Context (Priority 4)
**Goal**: Identify when/where to trade for best results

- [ ] **Test 5.1**: Time-of-Day Performance
- [ ] **Test 5.2**: Volatility Regime Performance
- [ ] **Test 5.3**: Trending vs Range-Bound Performance
- [ ] **Test 2.1**: Sentiment Extraction Accuracy
- [ ] **Test 2.2**: Multi-Source Sentiment Correlation
- [ ] **Test 2.4**: News Recency Impact

**Deliverable**: Context-aware trading rules

---

### Ongoing: Operational Tests
**Goal**: Monitor system reliability throughout Phase 2

- [ ] **Test 4.1**: Degraded Mode Performance
- [ ] **Test 4.2**: Exchange API Reliability

**Deliverable**: Operational guidelines and fallback procedures

---

## Critical Decision Points

### Decision Point 1: After Week 1
**Question**: Does the contrarian hypothesis work?

**If Test 2.3 shows negative correlation (p<0.05)**:
- ✅ **CONTINUE** with contrarian strategy
- Focus Phase 3 on optimizing inverse signals
- Build confidence in extreme sentiment readings

**If Test 2.3 shows positive correlation**:
- ⚠️ **PIVOT** to following strategy (trade with sentiment)
- Redesign Phase 3 approach
- Update signal logic

**If Test 2.3 shows no correlation**:
- 🔄 **RE-EVALUATE** data sources
- Test different time lags
- Consider alternative sentiment measures
- May need to pivot strategy entirely

---

### Decision Point 2: After Week 3
**Question**: Is the AI profitable on historical data?

**If Test 3.3 shows Profit Factor >1.5**:
- ✅ **PROCEED** to Phase 3 (optimization)
- High confidence in strategy foundation
- Focus on refinement, not redesign

**If Test 3.3 shows Profit Factor 1.0-1.5**:
- ⚠️ **CAUTIOUS** - marginally profitable
- Deep dive into Week 4 tests (context optimization)
- May salvage with better filtering/timing

**If Test 3.3 shows Profit Factor <1.0**:
- 🛑 **HALT** - strategy is unprofitable
- Do NOT proceed to Phase 3
- Options:
  1. Redesign strategy from scratch
  2. Pivot to different approach
  3. Abort project

---

## Test Output Format Standards

All test results should be saved as JSON with this structure:

```json
{
  "test_metadata": {
    "test_id": "2.3",
    "test_name": "Sentiment-Price Correlation",
    "execution_date": "2024-11-22T14:30:00Z",
    "data_period": {
      "start": "2024-10-01",
      "end": "2024-11-22"
    },
    "sample_size": 672
  },
  "results": {
    "primary_metric": -0.42,
    "secondary_metrics": {
      "p_value": 0.003,
      "optimal_lag_hours": 4,
      "extreme_fear_accuracy": 0.68,
      "extreme_greed_accuracy": 0.71
    }
  },
  "interpretation": {
    "success_criteria_met": true,
    "key_findings": [
      "Negative correlation confirmed (r=-0.42, p<0.05)",
      "Strongest at 4-hour lag",
      "Extreme values show best results"
    ],
    "recommendations": [
      "Use contrarian strategy",
      "Focus on extreme F&G readings",
      "Apply 4-hour signal delay"
    ]
  },
  "next_steps": [
    "Proceed with contrarian strategy in Phase 3",
    "Build confidence weighting for extreme values"
  ]
}
```

---

## Tools & Scripts

### Test Execution Scripts (To Be Created)

```bash
# Run individual test
python scripts/tests/run_test_2.3.py

# Run test category
python scripts/tests/run_category_2.py

# Run week's test suite
python scripts/tests/run_week_1_tests.py

# Run all Phase 2 tests
python scripts/tests/run_all_phase2_tests.py

# Generate test report
python scripts/tests/generate_report.py --week 1
```

### Test Automation

```bash
# Schedule daily test execution (Windows Task Scheduler)
# Run: tests/daily_test_runner.bat at 9 AM daily
```

---

## Success Metrics - Phase 2 Completion

Phase 2 is complete when you can confidently answer:

1. ✅ **Is my AI profitable on historical data?**
   - Evidence: Test 3.3 results
   - Threshold: Profit factor >1.5

2. ✅ **Does my contrarian hypothesis work?**
   - Evidence: Test 2.3 results
   - Threshold: Negative correlation, p<0.05

3. ✅ **What's my expected win rate and risk/reward?**
   - Evidence: Tests 3.1, 3.3 results
   - Threshold: 50%+ win rate OR 2:1+ R/R

4. ✅ **When should I trade vs stay out?**
   - Evidence: Tests 5.1, 5.2, 5.3 results
   - Threshold: Defined filter rules

5. ✅ **What's my maximum risk exposure?**
   - Evidence: Test 3.2 results
   - Threshold: Max drawdown <15%

---

## Documentation Updates Required

After Phase 2 completion, update:

1. **README.md**
   - Add Phase 2 completion date
   - Summarize key findings
   - Link to test results

2. **PROJECT_STRUCTURE.md**
   - Move Phase 2 to "Completed"
   - Update Phase 3 objectives based on findings

3. **Create PHASE2_RESULTS.md**
   - Consolidated test findings
   - Decision rationale for Phase 3
   - Lessons learned

---

## Notes & Reminders

- **Measure before optimizing** - Don't skip straight to tweaking
- **Data-driven decisions** - Let test results guide Phase 3, not intuition
- **Document everything** - Future you will thank present you
- **Version control discipline** - Commit after each test completion
- **Focus on priorities** - Week 1 tests are make-or-break

---

**Next Action**: Build Test 2.3 (Sentiment-Price Correlation) - the critical hypothesis test.
