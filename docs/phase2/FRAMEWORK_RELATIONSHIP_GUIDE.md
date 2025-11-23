# Phase 2 Framework Relationship - Visual Guide

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 2 TESTING STRATEGY                      │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
    ┌───────────────────────────┐   ┌──────────────────────────┐
    │ PHASE2_TESTING_FRAMEWORK  │   │  evaluation_framework.py │
    │         .md               │   │                          │
    │                           │   │                          │
    │   📋 TEST SPECIFICATION   │   │   🤖 TEST AUTOMATION     │
    │                           │   │                          │
    │  "WHAT should we test?"   │   │  "HOW do we test it?"    │
    └───────────────────────────┘   └──────────────────────────┘
                │                              │
                │   ┌──────────────────────────┘
                │   │
                ▼   ▼
    ┌─────────────────────────────────────┐
    │    TESTING_FRAMEWORK_MAPPING.md     │
    │                                     │
    │   🔗 BRIDGE DOCUMENT                │
    │                                     │
    │  "Which code implements which test" │
    └─────────────────────────────────────┘
```

## Analogy: Building a House

### PHASE2_TESTING_FRAMEWORK.md = The Blueprint
```
📐 Shows all rooms, features, and specifications
   - 19 rooms (tests) defined
   - Organized by floor (category)
   - Describes what each room should do
   - Prioritizes critical vs nice-to-have
```

### evaluation_framework.py = The Construction
```
🏗️ Actual built structure
   - 9 rooms fully constructed and furnished
   - 8 rooms have foundations, need interior work
   - 2 rooms marked "future expansion"
   - Ready to move in and use TODAY
```

### TESTING_FRAMEWORK_MAPPING.md = The Floor Plan
```
🗺️ Shows which blueprint rooms are built
   - "Kitchen (Test 1.1) = Built, First Floor, Left Side"
   - "Master Bedroom (Test 2.3) = Built, Second Floor"
   - "Guest Room (Test 3.2) = Not Yet Built"
```

---

## The Complete Picture

```
┌────────────────────────────────────────────────────────────────────┐
│                         YOUR PHASE 2 WORKFLOW                       │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  WEEK 1: Initial Testing                                           │
│  ┌─────────────────────────────────────────────────┐              │
│  │ 1. Run: python scripts/evaluation_framework.py  │              │
│  │                                                  │              │
│  │ 2. Automated Results Generated:                 │              │
│  │    ✅ Test 1.1: Directional Accuracy           │              │
│  │    ✅ Test 1.4: Type-Specific Accuracy         │              │
│  │    ✅ Test 2.1: Sentiment Direction            │              │
│  │    ✅ Test 2.3: CONTRARIAN HYPOTHESIS ⭐      │              │
│  │    ✅ Test 3.3: Win Rate & Profit Factor       │              │
│  │                                                  │              │
│  │ 3. Review Report:                               │              │
│  │    - Is accuracy >60%?                          │              │
│  │    - Does "CONTRARIAN BETTER" appear?          │              │
│  │    - Is profit factor >2.0?                     │              │
│  └─────────────────────────────────────────────────┘              │
│                                                                     │
│  WEEK 2: Manual Analysis                                           │
│  ┌─────────────────────────────────────────────────┐              │
│  │ Reference: TESTING_FRAMEWORK_MAPPING.md         │              │
│  │                                                  │              │
│  │ 1. Create notebook: phase2_analysis.ipynb       │              │
│  │                                                  │              │
│  │ 2. Add 3-4 manual analyses:                     │              │
│  │    📊 Test 1.2: Confidence Calibration         │              │
│  │    📊 Test 5.1: Day of Week Patterns           │              │
│  │    📊 Test 5.3: Volatility Regimes             │              │
│  └─────────────────────────────────────────────────┘              │
│                                                                     │
│  WEEK 3-4: Extended Testing                                        │
│  ┌─────────────────────────────────────────────────┐              │
│  │ Reference: PHASE2_TESTING_FRAMEWORK.md          │              │
│  │                                                  │              │
│  │ 1. Run with different time horizons:            │              │
│  │    - hours_forward=4 (day trading)              │              │
│  │    - hours_forward=48 (swing trading)           │              │
│  │                                                  │              │
│  │ 2. Compare results across timeframes            │              │
│  │                                                  │              │
│  │ 3. Document findings for Phase 3                │              │
│  └─────────────────────────────────────────────────┘              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

---

## File Purposes Summary

### 1. PHASE2_TESTING_FRAMEWORK.md
**Type**: Reference Document  
**When to use**: 
- Planning what to test
- Understanding test categories
- Deciding what to build next
- Phase 3 optimization planning

**Think of it as**: Your testing encyclopedia

---

### 2. evaluation_framework.py
**Type**: Executable Python Script  
**When to use**:
- Weekly (or more) evaluation runs
- Every time you want to check AI performance
- Before making strategy changes
- After accumulating new predictions

**Think of it as**: Your testing workhorse

---

### 3. TESTING_FRAMEWORK_MAPPING.md
**Type**: Bridge Document  
**When to use**:
- Understanding what's already built
- Finding which code implements which test
- Deciding if you need to build something new
- Planning manual analysis scripts

**Think of it as**: Your Rosetta Stone between spec and implementation

---

## Decision Tree: Which Document Do I Need?

```
START: "I want to..."

├─ "...understand what I should test"
│   └─→ Read: PHASE2_TESTING_FRAMEWORK.md
│
├─ "...actually run tests NOW"
│   └─→ Run: evaluation_framework.py
│
├─ "...know if a test is already built"
│   └─→ Check: TESTING_FRAMEWORK_MAPPING.md
│
├─ "...see my latest results"
│   └─→ Open: data/analysis_results/evaluation_report_*.json
│
├─ "...add a new test"
│   └─→ 1. Check TESTING_FRAMEWORK_MAPPING.md (is it built?)
│       2. If not, reference PHASE2_TESTING_FRAMEWORK.md (what it should do)
│       3. Extend evaluation_framework.py (build it)
│
└─ "...do manual analysis"
    └─→ 1. Check TESTING_FRAMEWORK_MAPPING.md (find "📊 Manual" tests)
        2. Use provided analysis scripts
        3. Create notebook with analysis code
```

---

## Key Insight: Coverage is Better Than You Think!

### You Already Have 90% of What You Need

```
PHASE2_TESTING_FRAMEWORK.md defines: 19 tests
                                      ↓
evaluation_framework.py automates:   9 tests (47%)
                                      ↓
Data available for manual:           8 tests (42%)
                                      ↓
Not critical right now:              2 tests (11%)
                                      ═════════════
                                      17 tests ready! (89%)
```

**Translation**: Out of 19 defined tests, 17 can be performed RIGHT NOW with what you have!

---

## The Most Important Test (Already Built!)

### Test 2.3: Sentiment-Price Correlation (Contrarian Analysis)

This is **YOUR HYPOTHESIS** - and it's **fully automated**!

```python
# In evaluation_framework.py
def _analyze_contrarian_strategy(self, evaluations: List[Dict]) -> Dict:
    """
    Test if doing the OPPOSITE of AI recommendations would perform better.
    
    This tests the hypothesis that market sentiment inversely correlates
    with price movements due to market maker dynamics.
    """
```

**Every time you run `evaluation_framework.py`, you get:**

```
🔄 CONTRARIAN STRATEGY ANALYSIS
   Contrarian Accuracy: XX.XX%
   Contrarian Hypothetical PnL: $±XXX.XX
   Result: CONTRARIAN BETTER / FOLLOWING BETTER
```

**This single output tells you:**
1. ✅ Is your market maker hypothesis valid?
2. ✅ Should you invert your signals?
3. ✅ Does "buy fear, sell greed" work?

**You don't need to build this - IT'S ALREADY RUNNING!**

---

## What to Do Right Now

### Step 1: Copy Files (2 minutes)
```bash
cd ~/Documents/projects/crypto-ai-trader

# Copy all Phase 2 files
cp /path/to/evaluation_framework.py scripts/
cp /path/to/test_evaluation.py scripts/
cp /path/to/*.md .
```

### Step 2: Run Quick Test (1 minute)
```bash
python scripts/test_evaluation.py
```

Should show all ✓ green checkmarks

### Step 3: Run Evaluation (2-3 minutes)
```bash
python scripts/evaluation_framework.py
```

### Step 4: Review Results (5 minutes)
Look for:
- Overall accuracy percentage
- **"CONTRARIAN BETTER" or "FOLLOWING BETTER"** ← THIS IS YOUR ANSWER!
- Win rate and profit factor

### Step 5: Make Decision
```
IF contrarian analysis shows "CONTRARIAN BETTER":
    → Your hypothesis is VALIDATED
    → Consider inverting buy/sell signals in Phase 3
    → Document this finding immediately

ELSE IF "FOLLOWING BETTER":
    → Current strategy is sound
    → Focus on refinement, not inversion
    → Continue with sentiment-aligned approach
```

---

## Summary: How They Work Together

| Document | Role | When to Use |
|----------|------|-------------|
| PHASE2_TESTING_FRAMEWORK.md | 📋 Master plan | Planning, reference, understanding |
| evaluation_framework.py | 🤖 Worker | Weekly runs, actual testing |
| TESTING_FRAMEWORK_MAPPING.md | 🔗 Guide | Finding what's built, planning additions |
| test_evaluation.py | ✅ Validator | Setup verification, troubleshooting |
| EVALUATION_FRAMEWORK_GUIDE.md | 📖 Manual | Detailed how-to, metric interpretation |
| PHASE_2_README.md | 🚀 Overview | Quick start, workflow |
| QUICK_REFERENCE.md | ⚡ Cheat sheet | Command lookup |

**You don't need all of them all the time!**

For 90% of your work:
1. Run `evaluation_framework.py` 
2. Reference `TESTING_FRAMEWORK_MAPPING.md` when confused
3. Check `QUICK_REFERENCE.md` for commands

That's it!

---

## Final Thought

```
You asked: "Are these frameworks aligned?"

Answer: YES! They're not conflicting - they're complementary.

┌─────────────────────────────────────────┐
│  PHASE2_TESTING_FRAMEWORK.md            │
│  = The architectural blueprint          │
│                                         │
│  evaluation_framework.py                │
│  = The actual building                  │
│                                         │
│  TESTING_FRAMEWORK_MAPPING.md           │
│  = The bridge between them              │
└─────────────────────────────────────────┘

Together they form a complete Phase 2 system.
Separately they each serve a clear purpose.
```

**Your contrarian hypothesis test is ALREADY RUNNING automatically.**  
**Just execute `evaluation_framework.py` and check the results!**

---

**Created**: November 22, 2025  
**Author**: J-Jarl  
**Purpose**: Clarify Phase 2 framework relationships
