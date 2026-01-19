# Phase 2: Evaluation Framework Roadmap

## Overview
Phase 2 focuses on rigorous evaluation and validation of the Bitcoin trading AI system before moving to live trading. The goal is to establish statistical confidence in system performance through comprehensive backtesting and metrics analysis.

**Note**: This system trades exclusively BTC/USDT pairs on Coinbase and Kraken exchanges.

---

## Success Criteria

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Directional Accuracy | >55% | 62.5% | :white_check_mark: Exceeds |
| Win Rate | >50% | TBD | Pending 2-week test |
| Profit Factor | >1.5 | TBD | Pending 2-week test |
| Sharpe Ratio | >1.0 | TBD | Pending 2-week test |

### Metric Definitions
- **Directional Accuracy**: % of predictions where predicted direction matches actual price movement
- **Win Rate**: % of trades that are profitable (exit price > entry price for longs)
- **Profit Factor**: Gross profits / Gross losses (measures risk-adjusted return)
- **Sharpe Ratio**: (Mean return - Risk free rate) / Std deviation of returns

---

## Completed Work

### Phase 2.0 - Bug Fixes & Initial Validation
**Status**: :white_check_mark: COMPLETE

#### Critical Bug Fixes
1. **Floating Point Precision Bug** (CRITICAL)
   - **Issue**: Threshold comparison `if confidence >= 6.95` was failing due to floating point representation
   - **Root Cause**: Python floating point arithmetic - `6.95` stored as `6.949999999999999...`
   - **Fix**: Changed to `if confidence >= 6.94` or use `round(confidence, 2)`
   - **Impact**: Predictions were being skipped silently

2. **Import Statement Fix**
   - **Issue**: Missing import for regime detection module
   - **Fix**: Added proper import path resolution

3. **Debug Print Syntax Issue**
   - **Issue**: `if/elif` blocks with only debug print statements caused syntax errors when prints removed
   - **Fix**: Proper conditional structure with pass statements or meaningful operations

#### Initial Backtest Results (Jan 13-17, 2025)
- **Period**: 5 trading days
- **Predictions Made**: 8
- **Directional Accuracy**: 62.5% (5/8 correct)
- **Total PnL**: +$10.88
- **Result**: PASSING initial validation

---

## Current Phase: 2.1 - Extended Validation

### Objective
Run 2-week backtest to establish statistical significance and validate performance across different market conditions.

### Tasks
- [ ] Run 2-week historical backtest (14+ trading days)
- [ ] Calculate all success metrics
- [ ] Analyze performance by market regime
- [ ] Document edge cases and failure modes
- [ ] Create performance visualization charts

### Test Parameters
- **Period**: 2 weeks minimum (10+ trading days)
- **Asset**: BTC/USDT only
- **Exchanges**: Coinbase, Kraken
- **Confidence Threshold**: 6.94 (adjusted for floating point)
- **Position Sizing**: Fixed $100 per trade for normalization

---

## Phase 2.2 - Stress Testing (Planned)

### Objectives
- Test system behavior during high volatility periods
- Validate drawdown limits and risk management
- Test recovery from losing streaks

### Test Scenarios
- Flash crash conditions
- Extended sideways markets
- Strong trend reversals
- Low liquidity periods

---

## Phase 2.3 - Paper Trading (Planned)

### Objectives
- Real-time validation without capital risk
- Test execution timing and slippage assumptions
- Validate API integration and order management

### Duration
- Minimum 2 weeks of paper trading
- Must maintain success criteria throughout

---

## Key Discoveries & Lessons Learned

### Technical Discoveries
1. **Floating Point Comparisons**: Always use slightly lower thresholds or explicit rounding for financial comparisons
2. **Silent Failures**: Add explicit logging when predictions are skipped to catch threshold issues
3. **Debug Code Hygiene**: Structure conditionals to work with or without debug prints

### Market Insights
- Initial 5-day test showed strong directional accuracy (62.5%)
- System appears to handle both long and short predictions
- Need longer test period to validate consistency

---

## Next Steps (Priority Order)

1. **Immediate**: Run 2-week backtest with current parameters
2. **Short-term**: Implement automated metrics calculation
3. **Medium-term**: Build performance dashboard
4. **Long-term**: Prepare for Phase 2.3 paper trading

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2025-01-17 | Initial bug fixes, 5-day validation complete |
| 2.1 | 2025-01-19 | Extended validation framework design |
