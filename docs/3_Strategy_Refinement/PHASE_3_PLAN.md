# Phase 3: Strategy Refinement

**Status:** Starting
**Date:** January 31, 2026
**Baseline:** Phase 2 results (83.33% accuracy, 3.69 PF, +$12.58 PnL)

---

## Problem Statement

The current system is **too conservative** for production:
- HOLD rate: 67% (sitting out 2/3 of opportunities)
- BUY signals: 0 in 2 weeks
- Active trades: Only 8/24 (33%)
- 2-week return: +0.13% (underperforms buy-and-hold)

BTC rallied 6.6% (Jan 1-14) and the system generated zero BUY signals.

---

## Root Causes

### Filters Blocking Valid Trades:

| Date | Signal | Why Blocked |
|------|--------|-------------|
| Jan 7 | BUY → HOLD | R/R 1.2:1 < 2:1 minimum |
| Jan 8 | BUY → SELL | Volume still elevated |
| Jan 14 | TRENDING_UP | "Exhaustion" → HOLD |
| Jan 15 | BUY → SELL | R/R 0.99:1, distribution override |

### Potentially Too Strict:
1. R/R minimum 2:1 for regular trades
2. Volume blocking during accumulation
3. TRENDING + sweep → HOLD (could be dip-buy)
4. Distribution override flipping BUY → SELL

---

## Phase 3 Strategy: Core + Expansion Module

### Core System (LOCKED - Do Not Modify)
- Current decision matrix logic
- Two-layer architecture
- Conservative HOLD behavior
- Baseline: 3.69 PF, 83.33% accuracy

### Expansion Module (EXPERIMENTAL)

**Candidates to loosen (carefully):**
1. R/R minimum: 2:1 → 1.5:1 for high-confidence setups
2. TRENDING_UP + pullback: Consider BUY instead of HOLD
3. Accumulation volume threshold adjustment

**Keep strict:**
1. Late entry filter
2. Distribution dominance override
3. Regime detection thresholds

---

## Hard Constraints (Revert if Breached)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Profit Factor | < 2.0 | Disable expansion |
| Max Drawdown | > 10-12% | Revert to core |
| Win Rate | < 40% | Review filters |
| Worst Month | > -8% | Pause and analyze |

---

## Tracking Requirements

Run backtests with TWO configurations:
1. **Core only** - Current system unchanged
2. **Core + Expansion** - With experimental loosening

Compare metrics side-by-side before promoting any change.

---

## Success Criteria for Phase 3

| Metric | Target |
|--------|--------|
| HOLD rate | < 50% (more engagement) |
| BUY signals | > 0 (capture uptrends) |
| Profit Factor | > 2.0 (maintain edge) |
| Max Drawdown | < 10% |
| Annualized Return | > Buy-and-hold |

---

## Next Steps

1. [ ] Run 1-month backtest as expanded baseline
2. [ ] Identify specific filter thresholds to adjust
3. [ ] Create expansion module toggle in code
4. [ ] Test R/R 1.5:1 for high-confidence setups
5. [ ] Test TRENDING_UP dip-buy logic
6. [ ] Compare Core vs Core+Expansion metrics
