# Crypto AI Trading System - Project Structure & Documentation

## Project Overview
An AI-driven Bitcoin trading system using sentiment analysis, technical indicators, and machine learning. Built with Python, Ollama (Qwen3-coder), and various data sources for comprehensive market analysis.

## Current Project Structure

```
crypto-ai-trader/
├── .git/                          # Git repository metadata
├── .gitignore                     # Git ignore rules
├── .vscode/                       # VS Code settings
├── LICENSE                        # Project license
├── README.md                      # Project description
├── exercises.md                   # Learning exercises
├── requirements.txt               # Python dependencies
│
├── data/                          # Data storage
│   ├── raw/                       # Downloaded external data (price history, news)
│   ├── processed/                 # Cleaned/transformed data
│   └── analysis_results/          # AI trading recommendations (PRIVATE - not committed)
│
├── scripts/                       # Main code files
│   ├── trading_ai.py             # Main Bitcoin trading AI
│   ├── evaluation_framework.py   # Testing and accuracy measurement
│   └── test_rss_feeds.py         # RSS feed testing utility
│
├── notebooks/                     # Jupyter notebooks for exploration
│   └── .ipynb_checkpoints/       # Jupyter autosave (ignored)
│
└── tests/                         # Unit tests (future)
```

## Folder Purposes

### `/data/` - Data Organization
- **`raw/`**: Original, unprocessed data
  - Downloaded news articles
  - Historical price data from exchanges
  - Social media data (future)
  - Never modify files here - treat as read-only

- **`processed/`**: Cleaned and transformed data
  - Normalized price data
  - Aggregated sentiment scores
  - Calculated technical indicators
  - Ready for analysis

- **`analysis_results/`**: AI trading outputs (PRIVATE)
  - Daily trading recommendations
  - Sentiment analysis results
  - Position sizing decisions
  - **NOT committed to Git** - contains your trading strategy

### `/scripts/` - Core Application Code
- **`trading_ai.py`**: Main trading system
  - Fetches BTC price, news, Fear & Greed Index
  - Analyzes sentiment using Qwen3-coder via Ollama
  - Generates trading recommendations (BUY/SELL/HOLD)
  - Calculates position sizing and risk/reward ratios

- **`evaluation_framework.py`**: Testing system
  - Evaluates AI prediction accuracy
  - Tests sentiment analysis quality
  - Measures trading decision correctness
  - Tracks performance over time

- **`test_rss_feeds.py`**: Utility for testing news sources

### `/notebooks/` - Jupyter Notebooks
- Exploratory data analysis
- Strategy backtesting
- Visualization and research

### `/tests/` - Unit Tests
- Automated testing (future implementation)

## Git Workflow

### Standard Commands

```bash
# Check status
git status

# Stage changes
git add .                          # Stage all changes
git add scripts/trading_ai.py      # Stage specific file
git add data/processed/            # Stage directory

# Commit
git commit -m "Add CCXT integration for exchange data"

# Push to GitHub
git push origin main

# Pull updates
git pull origin main

# View history
git log --oneline
```

### Commit Message Guidelines

**Format**: `[Action] Brief description of what changed`

**Good Examples**:
- `Add Fear & Greed Index integration to trading AI`
- `Fix Bitcoin-only filtering in RSS feeds`
- `Update evaluation framework with historical testing`
- `Refactor position sizing calculation for clarity`

**Bad Examples**:
- `stuff` (too vague)
- `fixed things` (not descriptive)
- `updates` (what kind of updates?)

**Principle**: Each commit = one logical change. If using "and" multiple times, split into separate commits.

## What to Commit vs Ignore

### ✅ Always Commit
- Source code (`.py`)
- Jupyter notebooks (`.ipynb`)
- Documentation (`.md`, `.txt`)
- Configuration files
- `requirements.txt`
- `.gitignore` itself

### ❌ Never Commit (.gitignore rules)

```gitignore
# Jupyter
.ipynb_checkpoints/
*/.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.so
*.egg-info/

# Virtual Environments
venv/
env/
.venv/
anaconda_projects/

# Analysis Results (PRIVATE TRADING DATA)
data/analysis_results/*

# Large Data Files
*.csv
*.xlsx
*.zip
*.h5
data/raw/*
data/processed/*

# OS-Specific
.DS_Store        # macOS
Thumbs.db        # Windows
desktop.ini

# API Keys and Secrets
.env
*.key
secrets/

# IDE
.vscode/settings.json
.idea/
```

**Why**: These files are auto-generated, environment-specific, too large, private, or can be recreated from source code.

## Development Environment

### Core Technologies
- **Python 3.13** (Anaconda distribution)
- **Ollama** with Qwen3-coder:30b model
- **Git** for version control
- **VS Code** with Claude Code extension
- **GitHub** for remote repository
                                                                                                                                                                                                      
### Python Environment
- **Conda Environment**: `crypto-ai-trader`
- **Python Version**: 3.13.9
- **Environment Location**: `C:\Users\JarlJ\anaconda3\envs\crypto-ai-trader`

To activate the environment:
```bash
conda activate crypto-ai-trader
```

### Key Python Libraries
```txt
requests>=2.31.0        # HTTP requests for APIs
feedparser>=6.0.10      # RSS feed parsing
ccxt                    # Cryptocurrency exchange data
```

### Setup Instructions

1. **Clone Repository**
   ```bash
   cd /Apps/Obsidian/Jarl
   git clone https://github.com/J-Jarl/crypto-ai-trader.git
   cd crypto-ai-trader
   ```

2. **Install Dependencies**
   ```bash                                                                                                                                                                                          
   conda activate crypto-ai-trader
   pip install -r requirements.txt
   ```

3. **Verify Ollama**
   ```bash
   ollama list  # Should show qwen3-coder:30b
   ```

4. **Run Trading AI**
   ```bash
   python scripts/trading_ai.py
   ```

## Important Conventions

### File Naming
- **Python files**: `lowercase_with_underscores.py`
- **Notebooks**: `week1_topic_description.ipynb`
- **Data files**: `YYYY-MM-DD_description.csv`

### Code Organization
- Keep functions focused (single responsibility)
- Use descriptive variable names
- Add docstrings for complex functions
- Comment WHY, not WHAT (code shows what)

### Data Management
- **Raw data**: Never modify originals
- **Processed data**: Document transformations
- **Analysis results**: Include timestamps in filenames

### Git Best Practices
1. **Commit frequently** with focused changes
2. **Pull before push** to avoid conflicts
3. **Review changes** with `git status` before committing
4. **Never commit secrets** (API keys, passwords)
5. **Use branches** for experimental features (future)

## Repository Information

- **URL**: `https://github.com/J-Jarl/crypto-ai-trader`
- **Primary Branch**: `main`
- **Location**: `C:\Apps\Obsidian\Jarl\crypto-ai-trader`
- **Backup**: `Automatically synced to NAS via Synology Drive`

## Setup & Migration History

### November 16, 2025 - Project Migration & Conda Environment Setup
**Major Changes:**
- Migrated entire project from `C:\Users\JarlJ\Documents\Projects\crypto-ai-trader` to `C:\Apps\Obsidian\Jarl\crypto-ai-trader`
- Project now automatically syncs to NAS via Synology Drive for backup
- Created dedicated conda environment `crypto-ai-trader` with Python 3.13.9
- Installed project dependencies: requests (2.32.5), feedparser (6.0.12), ccxt (4.5.19)
- Verified Git repository integrity at new location - all history preserved
- Tested trading_ai.py successfully - system fully functional
- Updated all documentation paths to reflect new location
- Updated conda from version 25.5.1 to 25.9.1

**Environment Details:**
- Conda Environment: `crypto-ai-trader`
- Location: `C:\Users\JarlJ\anaconda3\envs\crypto-ai-trader`
- Python Version: 3.13.9
- Activation: `conda activate crypto-ai-trader`

### Data Flow
```
1. Data Sources
   ├── RSS Feeds (CoinDesk, Cointelegraph, Bitcoin Magazine)
   ├── CCXT (Coinbase/Kraken exchange data)
   └── Alternative.me (Fear & Greed Index)
   
2. Analysis (Qwen3-coder via Ollama)
   ├── Sentiment Analysis
   ├── Technical Indicator Analysis
   └── Market Context Integration
   
3. Output
   ├── Trading Recommendation (BUY/SELL/HOLD)
   ├── Position Sizing
   ├── Stop Loss & Take Profit levels
   └── Risk/Reward Ratio
   
4. Storage
   └── JSON files in data/analysis_results/
```

### Operating Modes
- **🟢 FULL MODE**: Exchange + Fear & Greed + News (optimal)
- **🟡 PARTIAL MODE**: Fear & Greed + News (exchange unavailable)
- **🟠 NEWS ONLY MODE**: RSS feeds only (fallback)

## Current Development Phase

### ✅ Completed (Phase 1)
- Bitcoin trading AI built and functional
- Multi-source data integration
- Sentiment analysis with Qwen3-coder
- Technical indicator calculations
- Organized project structure

### 🔄 In Progress (Phase 2)
- Building evaluation framework
- Creating test cases from historical data
- Measuring prediction accuracy
- Testing contrarian vs following strategies

### 📋 Planned (Phase 3-5)
- **Phase 3**: Strategy optimization based on evaluation findings
- **Phase 4**: Collect fine-tuning dataset (500-1000 examples)
- **Phase 5**: Fine-tune model using Meta Llama API

## Troubleshooting

### Common Issues

**Issue**: CRLF/LF warnings on Windows
- **Solution**: Normal behavior, Git handles conversion automatically

**Issue**: Ollama model not found
- **Solution**: `ollama pull qwen3-coder:30b`

**Issue**: Exchange API blocked (Error 451)
- **Solution**: System automatically falls back to Kraken/Coinbase

**Issue**: JSON files in wrong location
- **Solution**: Update complete - now saves to `data/analysis_results/`

## Next Steps

### Immediate (This Week)
1. Build evaluation framework test cases
2. Analyze past trading recommendations for accuracy
3. Test contrarian hypothesis (inverse sentiment signals)

### Short-term (This Month)
1. Run daily analysis and collect results
2. Identify patterns in successful vs failed predictions
3. Refine strategy based on data

### Long-term (3-6 Months)
1. Accumulate 500+ examples of successful trades
2. Prepare fine-tuning dataset
3. Fine-tune model on proven strategies
4. Consider paper trading validation

---

**Last Updated**: November 13, 2025  
**Maintained By**: J-Jarl  
**Project Status**: Active Development - Phase 2 (Evaluation)