# Mode Notifications Feature

## Overview
Added clear mode notifications to help users quickly identify which data sources are available and diagnose any failures.

## Display Modes

### 🟢 FULL MODE
All data sources working:
```
[GREEN] Running in FULL MODE (Exchange + Fear & Greed + News)
================================================================================
OK Exchange: Coinbase connected
OK Fear & Greed Index: Available
OK Technical Indicators: RSI, SMA available
OK News Sources: RSS feeds active
================================================================================
```

### 🟡 PARTIAL MODE
Exchange failed, but Fear & Greed and news available:
```
[YELLOW] Running in PARTIAL MODE (Fear & Greed + News only)
================================================================================
OK Fear & Greed Index: Available
OK News Sources: RSS feeds active

WARNINGS:
  X Exchange: Coinbase: HTTP 451 - Geo-restricted (try VPN or different exchange)

  Suggestions:
    - Use a VPN to bypass geo-restrictions
    - Try a different exchange (modify MarketDataFetcher in code)
    - Use CoinGecko API as alternative (free, no restrictions)
================================================================================
```

### 🟠 NEWS ONLY MODE
Only news sources available:
```
[ORANGE] Running in NEWS ONLY MODE
================================================================================
OK News Sources: RSS feeds active

WARNINGS:
  X Exchange: Coinbase: HTTP 451 - Geo-restricted (try VPN or different exchange)

  Suggestions:
    - Use a VPN to bypass geo-restrictions
    - Try a different exchange (modify MarketDataFetcher in code)
    - Use CoinGecko API as alternative (free, no restrictions)

  X Fear & Greed: Timeout - API took too long to respond

  Suggestions:
    - API is slow, try again later
================================================================================
```

## Error Diagnostics

### Exchange Errors

**HTTP 451 / Geo-restricted**
- Cause: API blocked in your location
- Suggestions:
  - Use a VPN
  - Try different exchange
  - Use CoinGecko as alternative

**Timeout**
- Cause: API response too slow
- Suggestions:
  - Check internet connection
  - Try again later
  - Increase timeout in code

**Connection Error**
- Cause: Cannot connect to API
- Suggestions:
  - Verify internet connection
  - Check firewall settings

### Fear & Greed Errors

**Timeout**
- Cause: Alternative.me API slow
- Suggestions:
  - Try again later

**Connection Error**
- Cause: Cannot reach API
- Suggestions:
  - Check internet connection

## JSON Output

Mode information is saved in every analysis JSON file:

```json
{
  "timestamp": "2025-01-13T10:30:00",
  "analysis_mode": "PARTIAL",
  "data_sources": {
    "exchange_available": false,
    "exchange_name": "Coinbase",
    "exchange_error": "Coinbase: HTTP 451 - Geo-restricted (try VPN or different exchange)",
    "fear_greed_available": true,
    "fear_greed_error": null,
    "news_sources_count": 3
  },
  "market_data": { ... },
  "sentiment": { ... },
  "recommendation": { ... }
}
```

## Benefits

1. **Quick Status Check**: Immediately see what data is available
2. **Diagnostic Info**: Specific error messages with root causes
3. **Actionable Suggestions**: Clear steps to fix each issue
4. **Historical Tracking**: JSON files show mode over time
5. **Debugging**: Easy to identify patterns in failures

## Mode History Tracking

You can track your mode history by examining saved JSON files:

```bash
# See all analysis modes from saved files
grep -h "analysis_mode" btc_analysis_*.json

# Count how often each mode occurred
grep -h "analysis_mode" btc_analysis_*.json | sort | uniq -c

# Find files where exchange failed
grep -l "\"exchange_available\": false" btc_analysis_*.json
```

## Implementation Details

### Mode Detection Logic
- `FULL`: exchange_available AND fear_greed_available
- `PARTIAL`: fear_greed_available (but NOT exchange_available)
- `NEWS_ONLY`: neither exchange nor fear_greed available

### Data Source Diagnostics
New fields in `MarketData` dataclass:
- `exchange_available`: bool
- `exchange_name`: str
- `exchange_error`: Optional[str]
- `fear_greed_available`: bool
- `fear_greed_error`: Optional[str]

### Error Classification
Errors are categorized to provide specific suggestions:
- HTTP 451 → Geo-restriction
- Timeout → Network/API slowness
- Connection Error → Internet/firewall
- Other → General exchange/API issues
