# Phase 2 Quick Reference Card

## Essential Commands

### Initial Setup
```bash
# Navigate to project
cd ~/Documents/projects/crypto-ai-trader

# Copy evaluation files to project
cp evaluation_framework.py scripts/
cp test_evaluation.py scripts/
cp EVALUATION_FRAMEWORK_GUIDE.md .
cp PHASE_2_README.md .

# Verify setup
python scripts/test_evaluation.py
```

### Daily Workflow
```bash
# Generate prediction (do this daily)
python scripts/trading_ai.py

# Wait 24+ hours, then evaluate
python scripts/evaluation_framework.py
```

### Git Workflow
```bash
# Check what changed
git status

# Stage all changes
git add .

# Commit with message
git commit -m "Add Phase 2 evaluation framework"

# Push to GitHub
git push origin main

# Pull latest
git pull origin main
```

## Quick Diagnostics

### Check Prediction Files
```bash
# List all predictions
ls -lt data/analysis_results/*.json | head

# Count predictions
ls data/analysis_results/*.json | wc -l

# View latest prediction
cat data/analysis_results/bitcoin_analysis_*.json | python -m json.tool | head -30
```

### Check Evaluation Reports
```bash
# List evaluation reports
ls -lt data/analysis_results/evaluation_report_*.json

# View latest report summary
cat data/analysis_results/evaluation_report_*.json | python -m json.tool | grep -A 10 "prediction_accuracy"
```

### Test Components
```bash
# Test evaluation framework
python scripts/test_evaluation.py

# Test single prediction file
python -c "import json; print(json.load(open('data/analysis_results/bitcoin_analysis_20251113_143022.json')))"

# Test exchange API
python -c "import ccxt; print(ccxt.coinbase().fetch_ticker('BTC/USD')['last'])"
```

## Common Issues & Quick Fixes

### Issue: "No predictions to evaluate"
```bash
# Generate a prediction first
python scripts/trading_ai.py

# Wait 24 hours, then evaluate
```

### Issue: Exchange API error
```python
# In evaluation_framework.py, line 22, try:
self.exchange = ccxt.kraken()  # Instead of coinbase()
```

### Issue: Import error
```bash
# Verify file location
ls -l scripts/evaluation_framework.py

# Install dependencies
pip install -r requirements.txt
```

### Issue: Evaluation too slow
```python
# In evaluation_framework.py main(), change:
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=5)
```

## Metric Interpretations

### Good Performance Targets
- **Prediction Accuracy:** >60%
- **Win Rate:** >55%
- **Profit Factor:** >2.0
- **Sentiment Accuracy:** >60%

### Metric Meanings
| Metric | Formula | Good Value |
|--------|---------|------------|
| Accuracy | Correct / Total | >60% |
| Win Rate | Winning Trades / Total | >55% |
| Profit Factor | Avg Win / Avg Loss | >2.0 |
| Contrarian | Opposite Correct / Total | <50% (want current strategy better) |

## File Locations

```
crypto-ai-trader/
├── scripts/
│   ├── trading_ai.py              # Phase 1: Generate predictions
│   ├── evaluation_framework.py    # Phase 2: Evaluate predictions
│   └── test_evaluation.py         # Test setup
├── data/
│   └── analysis_results/
│       ├── bitcoin_analysis_*.json           # Predictions (Phase 1)
│       └── evaluation_report_*.json          # Evaluations (Phase 2)
├── EVALUATION_FRAMEWORK_GUIDE.md  # Detailed documentation
├── PHASE_2_README.md              # Phase 2 overview
└── PROJECT_STRUCTURE.md           # Overall project structure
```

## Customization Quick Edits

### Change Evaluation Window
```python
# In evaluation_framework.py main()
# Line ~585: hours_forward=24 → Change to 48, 72, etc.
evaluations = evaluator.evaluate_all_predictions(hours_forward=48, limit=10)
```

### Evaluate More/Fewer Predictions
```python
# In evaluation_framework.py main()
# Line ~585: limit=10 → Change to desired number or None for all
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=20)
```

### Change Correctness Thresholds
```python
# In evaluation_framework.py _evaluate_correctness()
# Lines ~145-148: Adjust buy_threshold and sell_threshold
buy_threshold = 1.0    # Require 1% increase for BUY
sell_threshold = -1.0  # Require 1% decrease for SELL
```

## Phase 2 Checklist

### Week 1
- [ ] Copy all Phase 2 files to project
- [ ] Run `test_evaluation.py` - all tests pass
- [ ] Run first evaluation on 5-10 predictions
- [ ] Review results and understand metrics

### Week 2-3
- [ ] Generate daily predictions with `trading_ai.py`
- [ ] Run weekly evaluations
- [ ] Document patterns in notebook
- [ ] Test contrarian hypothesis

### Week 4
- [ ] Comprehensive analysis of all data
- [ ] Identify optimization opportunities
- [ ] Document findings for Phase 3
- [ ] Commit work to Git

### Ready for Phase 3?
- [ ] 20+ evaluated predictions
- [ ] Accuracy >60%
- [ ] Understand what works/doesn't
- [ ] Contrarian tested
- [ ] Patterns documented

## Integration Points

### From Phase 1 (Trading AI)
```
trading_ai.py outputs → data/analysis_results/*.json
```

### To Phase 3 (Optimization)
```
evaluation_report_*.json → Identify patterns → Adjust strategy parameters
```

### To Phase 4 (Paper Trading)
```
If accuracy >65% + profit factor >2.0 → Ready for paper trading validation
```

## Key Python Objects

### TradingEvaluator Class
```python
from evaluation_framework import TradingEvaluator

evaluator = TradingEvaluator()

# Load prediction
prediction = evaluator.load_prediction(filepath)

# Evaluate single prediction
evaluation = evaluator.evaluate_prediction(prediction, hours_forward=24)

# Evaluate all
evaluations = evaluator.evaluate_all_predictions(hours_forward=24, limit=10)

# Generate report
report = evaluator.generate_performance_report(evaluations)

# Save results
evaluator.save_evaluation_results(evaluations, report)
```

## Helpful One-Liners

```bash
# Count total predictions
find data/analysis_results -name "bitcoin_analysis_*.json" | wc -l

# Find oldest prediction
ls -t data/analysis_results/bitcoin_analysis_*.json | tail -1

# Find newest prediction
ls -t data/analysis_results/bitcoin_analysis_*.json | head -1

# Check prediction timestamps
grep -h "timestamp" data/analysis_results/bitcoin_analysis_*.json | head -5

# View latest evaluation accuracy
grep -A 3 "prediction_accuracy" data/analysis_results/evaluation_report_*.json | tail -5

# Check contrarian results
grep -A 5 "contrarian_analysis" data/analysis_results/evaluation_report_*.json | tail -6
```

## Documentation Hierarchy

1. **PHASE_2_README.md** ← Start here (Overview & quick start)
2. **This file** ← Quick reference for commands
3. **EVALUATION_FRAMEWORK_GUIDE.md** ← Detailed technical docs
4. **PROJECT_STRUCTURE.md** ← Overall project organization

## Support Flow

1. **Quick issue?** → Check this reference card
2. **Need details?** → Check EVALUATION_FRAMEWORK_GUIDE.md
3. **Setup problem?** → Run `test_evaluation.py`
4. **Still stuck?** → Review error messages and check file locations

---

**Keep this file handy!** Bookmark it for quick access to common commands.

**Last Updated:** November 14, 2025
