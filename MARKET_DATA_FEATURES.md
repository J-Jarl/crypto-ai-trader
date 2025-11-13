# Market Data Features Added to Bitcoin Trading AI

## Overview
Enhanced the Bitcoin trading sentiment analyzer with real-time market data, technical indicators, and sentiment indices to provide more comprehensive trading recommendations.

## New Data Sources

### 1. CCXT Library Integration
- **Exchange**: Coinbase (primary) with Kraken fallback (BTC/USD or BTC/USDT pair)
- **Data Fetched**:
  - Real-time BTC price
  - 24-hour trading volume
  - 24-hour price change (absolute and percentage)
  - OHLCV data for technical analysis

### 2. Fear & Greed Index
- **Source**: Alternative.me API (`https://api.alternative.me/fng/`)
- **Data**: Current crypto market sentiment score (0-100)
- **Classification**: Extreme Fear, Fear, Neutral, Greed, Extreme Greed

### 3. Technical Indicators
All calculated from real exchange data:
- **RSI (14)**: Relative Strength Index
  - >70 = Overbought
  - <30 = Oversold
  - 30-70 = Neutral
- **SMA (20)**: 20-day Simple Moving Average
- **SMA (50)**: 50-day Simple Moving Average
- **Price Position**: Whether current price is above/below each SMA

## Integration Points

### Sentiment Analysis
The AI now considers market data when analyzing news:
- Fear & Greed Index influences sentiment confidence
- RSI helps identify overbought/oversold conditions
- Moving averages show trend direction
- All included in the analysis context sent to Ollama

### Trading Recommendations
The AI advisor uses technical indicators to:
- Validate news-based sentiment
- Adjust position sizing based on market conditions
- Set stop-loss and take-profit levels using support/resistance
- Consider market volatility in risk management

### Output Display
New "MARKET DATA & TECHNICAL INDICATORS" section shows:
```
MARKET DATA & TECHNICAL INDICATORS
--------------------------------------------------------------------------------
Price: $XX,XXX.XX
24h Change: +X.XX%
24h Volume: $X,XXX,XXX,XXX
Fear & Greed Index: XX - Classification
RSI(14): XX.XX (Signal)
SMA(20): $XX,XXX.XX (Price is above/below)
SMA(50): $XX,XXX.XX (Price is above/below)
```

## Files Modified

1. **trading_ai.py**:
   - Added `MarketData` dataclass
   - Added `MarketDataFetcher` class
   - Updated `BitcoinSentimentAnalyzer.analyze_articles()` to accept market data
   - Updated `BitcoinTradingAdvisor.generate_recommendation()` to use market data
   - Enhanced `BitcoinTradingBot` to fetch and integrate market data
   - Added market metrics to output display

2. **requirements.txt**:
   - Added `ccxt>=4.0.0`

## Usage

No configuration changes needed! The system automatically:
1. Fetches market data from Binance via CCXT
2. Gets Fear & Greed Index from Alternative.me
3. Calculates technical indicators
4. Integrates everything into AI analysis

Simply run:
```bash
pip install -r requirements.txt
python scripts/trading_ai.py
```

## Technical Details

### RSI Calculation
- Uses 14-period RSI
- Calculated from daily closing prices
- Smoothed using Wilder's exponential moving average

### Moving Averages
- SMA(20) and SMA(50) from daily closing prices
- Minimum 50 days of historical data required
- Fetched from Coinbase/Kraken with 1d timeframe

### Error Handling
- **Exchange Selection**: Tries Coinbase first, falls back to Kraken if unavailable
- **Graceful Degradation**: Analysis continues even if exchange data fails
- **Null Safety**: All market data fields are checked before formatting
- **Continues Operation**: System works with news sentiment alone if technical data unavailable
- **Fear & Greed**: Still fetches even if exchange is down
- **All Indicators Optional**: Missing technical indicators don't break the analysis

## Benefits

1. **More Informed Decisions**: Combines news sentiment with market reality
2. **Risk Management**: Technical indicators help validate trading signals
3. **Market Context**: Fear & Greed Index adds behavioral finance perspective
4. **Free Data**: All sources are free to use (no API keys required)
5. **Real-time**: Data is fetched live for each analysis run
