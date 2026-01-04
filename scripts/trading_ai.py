"""
Bitcoin Trading Sentiment Analyzer
Uses Ollama with qwen3-coder:30b model to analyze news and make trading recommendations
"""

import json
import os
import requests
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
from dataclasses import dataclass
import feedparser
import ccxt


@dataclass
class NewsArticle:
    """Represents a news article about Bitcoin"""
    title: str
    description: str
    url: str
    published_at: str
    source: str


@dataclass
class SentimentAnalysis:
    """Represents sentiment analysis results"""
    sentiment: str  # 'bullish', 'bearish', or 'neutral'
    confidence: float  # 0-100
    key_points: List[str]
    reasoning: str


@dataclass
class TradingRecommendation:
    """Represents a trading recommendation"""
    action: str  # 'buy', 'sell', or 'hold'
    confidence: float  # 0-100
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    position_size_percentage: float  # percentage of portfolio
    reasoning: str


@dataclass
class MarketData:
    """Represents current market data and indicators"""
    current_price: float
    volume_24h: float
    price_change_24h: float
    price_change_percentage_24h: float
    fear_greed_index: Optional[int]  # 0-100
    fear_greed_classification: Optional[str]
    rsi_14: Optional[float]  # 0-100
    sma_5: Optional[float]
    sma_10: Optional[float]
    sma_20: Optional[float]
    sma_50: Optional[float]
    ema_9: Optional[float]
    price_vs_sma20: Optional[str]  # "above" or "below"
    price_vs_sma50: Optional[str]  # "above" or "below"
    # Liquidity zone analysis
    nearest_support: Optional[float] = None
    nearest_resistance: Optional[float] = None
    distance_to_support_pct: Optional[float] = None
    distance_to_resistance_pct: Optional[float] = None
    volume_ratio: Optional[float] = None
    volume_spike: bool = False
    potential_liquidity_zone: bool = False
    # Wyckoff pattern detection (multi-timeframe)
    wyckoff_patterns: Optional[Dict] = None  # {"short_term": {...}, "medium_term": {...}, "long_term": {...}}
    # Diagnostic info
    exchange_available: bool = False
    exchange_name: str = "None"
    exchange_error: Optional[str] = None
    fear_greed_available: bool = False
    fear_greed_error: Optional[str] = None


def format_relative_time(timestamp_struct) -> str:
    """Format a time.struct_time as relative time (e.g., '2 hours ago')"""
    if not timestamp_struct:
        return "Unknown time"

    try:
        published_time = time.mktime(timestamp_struct)
        now = time.time()
        diff_seconds = now - published_time

        if diff_seconds < 0:
            return "Just now"
        elif diff_seconds < 60:
            return "Just now"
        elif diff_seconds < 3600:
            minutes = int(diff_seconds / 60)
            return f"{minutes} {'minute' if minutes == 1 else 'minutes'} ago"
        elif diff_seconds < 86400:
            hours = int(diff_seconds / 3600)
            return f"{hours} {'hour' if hours == 1 else 'hours'} ago"
        elif diff_seconds < 604800:
            days = int(diff_seconds / 86400)
            return f"{days} {'day' if days == 1 else 'days'} ago"
        else:
            # Format as date if more than a week old
            dt = datetime.fromtimestamp(published_time)
            return dt.strftime("%b %d, %I:%M %p")
    except Exception:
        return "Unknown time"


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(self, model: str = "qwen2.5-coder:32b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.api_url = f"{base_url}/api/generate"

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7) -> str:
        """Generate text using Ollama model"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error communicating with Ollama: {e}")

    def check_model_availability(self) -> bool:
        """Check if the specified model is available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return any(self.model in model.get("name", "") for model in models)
        except Exception as e:
            print(f"Error checking model availability: {e}")
            return False


class MarketDataFetcher:
    """Fetches real-time market data and calculates technical indicators"""

    def __init__(self):
        # Try exchanges in order: Coinbase -> Kraken -> Binance.US -> CoinGecko (API only)
        self.exchange = None
        self.exchange_name = "None"
        self.exchanges_tried = []

        # Try Coinbase
        try:
            self.exchange = ccxt.coinbase()
            self.exchange_name = "Coinbase"
            self.exchanges_tried.append("Coinbase: Success")
        except Exception as e:
            self.exchanges_tried.append(f"Coinbase: {str(e)[:50]}")

            # Try Kraken
            try:
                self.exchange = ccxt.kraken()
                self.exchange_name = "Kraken"
                self.exchanges_tried.append("Kraken: Success")
            except Exception as e2:
                self.exchanges_tried.append(f"Kraken: {str(e2)[:50]}")

                # Try Binance.US
                try:
                    self.exchange = ccxt.binanceus()
                    self.exchange_name = "Binance.US"
                    self.exchanges_tried.append("Binance.US: Success")
                except Exception as e3:
                    self.exchanges_tried.append(f"Binance.US: {str(e3)[:50]}")
                    # CoinGecko will be tried as final fallback in fetch methods
                    self.exchange = None
                    self.exchange_name = "None"

    def get_coingecko_data(self) -> Optional[Dict]:
        """Fetch BTC price data from CoinGecko API (final fallback, no auth needed)"""
        try:
            url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_vol=true&include_24hr_change=true"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("bitcoin"):
                btc_data = data["bitcoin"]
                return {
                    "price": btc_data.get("usd", 0),
                    "volume_24h": btc_data.get("usd_24h_vol", 0),
                    "price_change_percentage_24h": btc_data.get("usd_24h_change", 0)
                }
            return None
        except Exception as e:
            print(f"CoinGecko API failed: {str(e)}")
            return None

    def get_fear_greed_index(self) -> tuple[Optional[int], Optional[str], Optional[str]]:
        """Fetch Fear & Greed Index from Alternative.me API"""
        try:
            response = requests.get("https://api.alternative.me/fng/", timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get("data") and len(data["data"]) > 0:
                latest = data["data"][0]
                value = int(latest.get("value", 50))
                classification = latest.get("value_classification", "Unknown")
                return value, classification, None

            return None, None, "No data returned from API"
        except requests.exceptions.Timeout:
            return None, None, "Timeout - API took too long to respond"
        except requests.exceptions.ConnectionError:
            return None, None, "Connection error - Check internet connection"
        except Exception as e:
            return None, None, str(e)

    def _try_fetch_ticker_with_failover(self) -> Optional[Dict]:
        """Try fetching ticker from all exchanges in order, return data with exchange name"""
        # Try current exchange if available
        if self.exchange:
            try:
                # Try BTC/USD first (Coinbase), then BTC/USDT (others)
                try:
                    ticker = self.exchange.fetch_ticker('BTC/USD')
                    return {"ticker": ticker, "exchange": self.exchange_name, "symbol": "BTC/USD"}
                except Exception:
                    ticker = self.exchange.fetch_ticker('BTC/USDT')
                    return {"ticker": ticker, "exchange": self.exchange_name, "symbol": "BTC/USDT"}
            except Exception as e:
                print(f"Error fetching ticker from {self.exchange_name}: {str(e)[:50]}")

        # Try other CCXT exchanges in failover order
        exchanges_to_try = [
            ('coinbase', 'Coinbase', 'BTC/USD'),
            ('kraken', 'Kraken', 'BTC/USDT'),
            ('binanceus', 'Binance.US', 'BTC/USDT')
        ]

        for exchange_id, exchange_name, symbol in exchanges_to_try:
            if exchange_name == self.exchange_name:
                continue  # Already tried above

            try:
                temp_exchange = getattr(ccxt, exchange_id)()
                ticker = temp_exchange.fetch_ticker(symbol)
                print(f"Successfully fetched ticker from fallback: {exchange_name}")
                return {"ticker": ticker, "exchange": exchange_name, "symbol": symbol}
            except Exception as e:
                print(f"Failover {exchange_name} ticker failed: {str(e)[:50]}")
                continue

        # Final fallback: CoinGecko API
        print("All CCXT exchanges failed, trying CoinGecko API...")
        coingecko_data = self.get_coingecko_data()
        if coingecko_data:
            # Convert CoinGecko format to ticker-like format
            print("Successfully fetched data from CoinGecko API")
            return {
                "ticker": {
                    "last": coingecko_data["price"],
                    "close": coingecko_data["price"],
                    "change": None,
                    "percentage": coingecko_data["price_change_percentage_24h"]
                },
                "exchange": "CoinGecko",
                "symbol": "BTC/USD",
                "volume_24h": coingecko_data["volume_24h"]
            }

        # All sources failed
        return None

    def get_btc_ohlcv(self, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Fetch OHLCV data from exchange with multi-exchange failover"""
        # Try current exchange if available
        if self.exchange:
            try:
                # Try BTC/USD first (Coinbase), then BTC/USDT (others)
                try:
                    ohlcv = self.exchange.fetch_ohlcv('BTC/USD', timeframe=timeframe, limit=limit)
                    return ohlcv
                except Exception:
                    ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
                    return ohlcv
            except Exception as e:
                print(f"Error fetching OHLCV from {self.exchange_name}: {str(e)[:50]}")

        # Try other exchanges in failover order
        exchanges_to_try = [
            ('coinbase', 'Coinbase', 'BTC/USD'),
            ('kraken', 'Kraken', 'BTC/USDT'),
            ('binanceus', 'Binance.US', 'BTC/USDT')
        ]

        for exchange_id, exchange_name, symbol in exchanges_to_try:
            if exchange_name == self.exchange_name:
                continue  # Already tried above

            try:
                temp_exchange = getattr(ccxt, exchange_id)()
                ohlcv = temp_exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                print(f"Successfully fetched OHLCV from fallback: {exchange_name}")
                return ohlcv
            except Exception as e:
                print(f"Failover {exchange_name} OHLCV failed: {str(e)[:50]}")
                continue

        # All exchanges failed
        print("All exchanges failed for OHLCV data - no technical indicators available")
        return []

    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return round(rsi, 2)

    def calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return round(sum(prices[-period:]) / period, 2)

    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None

        # Start with SMA for the first value
        sma = sum(prices[:period]) / period
        ema = sma

        # Calculate EMA for remaining prices
        multiplier = 2 / (period + 1)
        for price in prices[period:]:
            ema = (price - ema) * multiplier + ema

        return round(ema, 2)

    def calculate_atr(self, ohlcv_data: List[List], period: int = 14, timeframe: str = "1d") -> Optional[float]:
        """
        Calculate Average True Range (ATR) for volatility measurement.

        Args:
            ohlcv_data: OHLCV candle data [timestamp, open, high, low, close, volume]
            period: ATR period (default 14)
            timeframe: Candle timeframe for reference (default "1d")

        Returns:
            ATR value or None if insufficient data
        """
        if not ohlcv_data or len(ohlcv_data) < period + 1:
            return None

        true_ranges = []

        for i in range(1, len(ohlcv_data)):
            high = ohlcv_data[i][2]
            low = ohlcv_data[i][3]
            prev_close = ohlcv_data[i-1][4]

            # True Range is the greatest of:
            # 1. Current High - Current Low
            # 2. abs(Current High - Previous Close)
            # 3. abs(Current Low - Previous Close)
            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            true_ranges.append(tr)

        if len(true_ranges) < period:
            return None

        # Calculate ATR as simple moving average of True Ranges
        atr = sum(true_ranges[-period:]) / period
        return round(atr, 2)

    def calculate_hybrid_stop_loss(
        self,
        ohlcv_data: List[List],
        entry_price: float,
        target_price: Optional[float],
        action: str,  # "buy" or "sell"
        portfolio_value: float,
        timeframe: str = "1h",  # Timeframe for ATR calculation
        is_bounce_pattern: bool = False,  # Relaxed R/R for bounce patterns
        market_data: Optional['MarketData'] = None  # For capitulation detection
    ) -> Dict:
        """
        Calculate hybrid stop-loss with ATR, R/R validation, and leverage optimization.

        Args:
            ohlcv_data: OHLCV candle data
            entry_price: Entry price for the trade
            target_price: Take profit target (optional)
            action: Trade direction ("buy" or "sell")
            portfolio_value: Total portfolio value in USD
            timeframe: Timeframe for ATR calculation (default "1h" for 12h evaluation window)
            is_bounce_pattern: Whether this is a bounce pattern trade (relaxed R/R: 1:1 vs 1:2)

        Returns:
            Dictionary with comprehensive stop-loss and leverage analysis
        """
        # Calculate ATR with specified timeframe
        atr = self.calculate_atr(ohlcv_data, period=14, timeframe=timeframe)

        if not atr or not entry_price or entry_price <= 0:
            return {
                "atr_value": None,
                "atr_stop_loss": None,
                "risk_reward_ratio": None,
                "meets_rr_minimum": False,
                "recommended_leverage": 1,
                "leverage_analysis": {},
                "max_safe_position_pct": 5.0,
                "stop_type": "ATR-based hybrid",
                "error": "Insufficient data or invalid entry price"
            }

        # Calculate ATR-based stop loss (2.5x ATR from entry)
        if action.lower() == "buy":
            atr_stop_loss = entry_price - (2.5 * atr)
        else:  # sell
            atr_stop_loss = entry_price + (2.5 * atr)

        # Calculate risk/reward ratio
        risk_reward_ratio = None
        meets_rr_minimum = False

        # Check for capitulation (Wyckoff bottom conditions)
        is_capitulation = False
        if is_bounce_pattern and market_data and action.lower() == 'buy':
            rsi = market_data.rsi_14 if market_data.rsi_14 else 50
            fear_greed = market_data.fear_greed_index if market_data.fear_greed_index else 50

            # Check for volume decline (confirms accumulation maturing)
            volume_declining = False
            if hasattr(market_data, 'wyckoff_patterns'):
                short_term = market_data.wyckoff_patterns.get('short_term', {})
                if short_term and short_term.get('pattern') == 'accumulation':
                    volume_declining = short_term.get('volume_declining', False)

            # VOLUME CHECK FIRST - Block ALL accumulation BUYs with high volume
            if not volume_declining:
                # Volume still elevated - could be:
                # 1. Early accumulation (not mature yet)
                # 2. Market maker pump/trap (especially if RSI > 70)
                # 3. Distribution disguised as accumulation
                print(f"  ⚠️  Accumulation detected but volume still high - BLOCKING TRADE")
                if rsi > 70:
                    print(f"  → RSI {rsi:.1f} OVERBOUGHT + high volume = potential trap!")
                else:
                    print(f"  → Volume not declining yet - too early to enter")
                # Explicitly block this trade - volume must decline before entry
                is_capitulation = False
                block_trade_high_volume = True
            elif rsi < 50 and fear_greed < 30:
                # Volume declining + oversold + fear = True capitulation
                is_capitulation = True
                print(f"  🎯 CAPITULATION DETECTED: RSI {rsi:.1f} + F&G {fear_greed} + Volume Declining")
                print(f"  → Mature Wyckoff bottom - lowering R/R requirement to 0.75:1")

        # Check if this is a bounce pattern trade - these get relaxed R/R requirements
        if is_bounce_pattern:
            min_rr_ratio = 0.75 if is_capitulation else 1.0  # Capitulation: 0.75:1, Regular bounce: 1:1
        else:
            min_rr_ratio = 2.0  # Regular trades: 1:2 minimum

        # Check volume blocking flag
        block_trade_high_volume = locals().get('block_trade_high_volume', False)

        if target_price and target_price > 0:
            if action.lower() == "buy":
                potential_profit = target_price - entry_price
                potential_loss = entry_price - atr_stop_loss
            else:  # sell
                potential_profit = entry_price - target_price
                potential_loss = atr_stop_loss - entry_price

            if potential_loss > 0:
                risk_reward_ratio = round(potential_profit / potential_loss, 2)
                # Block if volume high OR R/R insufficient
                if block_trade_high_volume:
                    meets_rr_minimum = False  # Force fail
                    print(f"  ⚠️  Volume blocking active - trade will be rejected")
                else:
                    meets_rr_minimum = risk_reward_ratio >= min_rr_ratio

        # Leverage optimization analysis
        leverage_levels = [1, 2, 3, 5, 10]
        leverage_analysis = {}
        recommended_leverage = 1
        max_safe_position_pct = 5.0

        for leverage in leverage_levels:
            analysis = self._analyze_leverage_level(
                leverage=leverage,
                entry_price=entry_price,
                atr_stop_loss=atr_stop_loss,
                atr=atr,
                action=action,
                portfolio_value=portfolio_value
            )
            leverage_analysis[f"{leverage}x"] = analysis

            # Update recommended leverage if this level is safe
            if analysis["safe"] and leverage > recommended_leverage:
                recommended_leverage = leverage
                max_safe_position_pct = analysis["max_position"]

        return {
            "atr_value": round(atr, 2),
            "atr_timeframe": timeframe,
            "atr_stop_loss": round(atr_stop_loss, 2),
            "risk_reward_ratio": risk_reward_ratio,
            "meets_rr_minimum": meets_rr_minimum,
            "recommended_leverage": recommended_leverage,
            "leverage_analysis": leverage_analysis,
            "max_safe_position_pct": round(max_safe_position_pct, 2),
            "stop_type": "ATR-based hybrid"
        }

    def _analyze_leverage_level(
        self,
        leverage: int,
        entry_price: float,
        atr_stop_loss: float,
        atr: float,
        action: str,
        portfolio_value: float
    ) -> Dict:
        """
        Analyze a specific leverage level for safety and position sizing.

        Returns:
            Dict with liquidation price, safety status, and max position size
        """
        # Calculate liquidation price
        # For long: liq_price = entry * (1 - 1/leverage * 0.9)  # 90% to account for fees
        # For short: liq_price = entry * (1 + 1/leverage * 0.9)

        if leverage == 1:
            # No leverage, no liquidation
            liq_price = None
            safe = True
            max_position = 5.0  # 5% of portfolio
        else:
            if action.lower() == "buy":
                # Long position - liquidation below entry
                liq_price = round(entry_price * (1 - (1 / leverage) * 0.9), 2)

                # Check if liquidation is beyond stop + 1.5 ATR buffer
                required_liq_price = atr_stop_loss - (1.5 * atr)
                buffer_pct = ((atr_stop_loss - liq_price) / atr_stop_loss) * 100 if atr_stop_loss > 0 else 0

            else:  # sell/short
                # Short position - liquidation above entry
                liq_price = round(entry_price * (1 + (1 / leverage) * 0.9), 2)

                # Check if liquidation is beyond stop + 1.5 ATR buffer
                required_liq_price = atr_stop_loss + (1.5 * atr)
                buffer_pct = ((liq_price - atr_stop_loss) / atr_stop_loss) * 100 if atr_stop_loss > 0 else 0

            # Safety criteria:
            # 1. Liquidation must be beyond stop + 1.5 ATR
            # 2. Buffer must be >= 20% beyond stop loss
            if action.lower() == "buy":
                safe = (liq_price <= required_liq_price) and (buffer_pct >= 20)
            else:
                safe = (liq_price >= required_liq_price) and (buffer_pct >= 20)

            # Calculate max position size based on leverage and safety
            # Higher leverage = smaller position to maintain safety
            base_position = 5.0  # 5% base
            max_position = min(base_position, base_position / (leverage / 2))

        return {
            "liq_price": liq_price,
            "safe": safe,
            "max_position": round(max_position, 1)
        }

    def detect_wyckoff_patterns(self, ohlcv_data: List[List]) -> Dict:
        """
        Detect Wyckoff accumulation/distribution patterns and failed breakouts across multiple timeframes.

        Args:
            ohlcv_data: OHLCV candle data [timestamp, open, high, low, close, volume]

        Returns:
            Dictionary with nested timeframe structure:
            {
                "short_term": {...pattern details...} or None,
                "medium_term": {...pattern details...} or None,
                "long_term": {...pattern details...} or None
            }
        """
        if not ohlcv_data or len(ohlcv_data) < 20:
            return {
                "short_term": None,
                "medium_term": None,
                "long_term": None
            }

        # Multi-timeframe analysis for 12h trading
        # Short-term: Last 20 candles (most relevant for 12h predictions)
        # Medium-term: Last 50 candles
        # Long-term: Last 100 candles (full dataset)

        result = {}

        # Short-term analysis (last 20 candles - highest priority for 12h)
        if len(ohlcv_data) >= 20:
            short_data = ohlcv_data[-20:]
            result["short_term"] = self._detect_wyckoff_single_timeframe(short_data)
        else:
            result["short_term"] = None

        # Medium-term analysis (last 50 candles)
        if len(ohlcv_data) >= 50:
            medium_data = ohlcv_data[-50:]
            result["medium_term"] = self._detect_wyckoff_single_timeframe(medium_data)
        else:
            result["medium_term"] = None

        # Long-term analysis (full dataset, up to 100 candles)
        long_data = ohlcv_data[-100:] if len(ohlcv_data) > 100 else ohlcv_data
        result["long_term"] = self._detect_wyckoff_single_timeframe(long_data)

        return result

    def _detect_wyckoff_single_timeframe(self, ohlcv_data: List[List]) -> Optional[Dict]:
        """
        Detect Wyckoff pattern for a single timeframe.

        Args:
            ohlcv_data: OHLCV candle data for specific timeframe

        Returns:
            Dictionary with pattern type and details, or None if no pattern
        """
        if not ohlcv_data or len(ohlcv_data) < 10:
            return None

        # Extract highs, lows, closes, and volumes
        highs = [candle[2] for candle in ohlcv_data]
        lows = [candle[3] for candle in ohlcv_data]
        closes = [candle[4] for candle in ohlcv_data]
        volumes = [candle[5] for candle in ohlcv_data]

        # Find all swing highs and lows with their indices and volumes
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                swing_highs.append({"price": highs[i], "index": i, "volume": volumes[i]})

        swing_lows = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                swing_lows.append({"price": lows[i], "index": i, "volume": volumes[i]})

        # 1. Check for Distribution Pattern (multiple tests of resistance)
        distribution = self._detect_distribution(swing_highs)
        if distribution:
            return distribution

        # 2. Check for Accumulation Pattern (multiple tests of support)
        accumulation = self._detect_accumulation(swing_lows)
        if accumulation:
            return accumulation

        # 3. Check for Failed Breakouts (liquidity grabs)
        failed_breakout = self._detect_failed_breakout(ohlcv_data, swing_highs, swing_lows)
        if failed_breakout:
            return failed_breakout

        return None

    def _detect_distribution(self, swing_highs: List[Dict]) -> Optional[Dict]:
        """Detect distribution pattern - multiple tests of same resistance level"""
        if len(swing_highs) < 3:
            return None

        # Group swing highs by similar price levels (within 2% tolerance)
        tolerance = 0.02
        resistance_clusters = []

        for swing in swing_highs:
            price = swing["price"]
            added_to_cluster = False

            # Try to add to existing cluster
            for cluster in resistance_clusters:
                cluster_avg = sum(s["price"] for s in cluster) / len(cluster)
                if abs(price - cluster_avg) / cluster_avg <= tolerance:
                    cluster.append(swing)
                    added_to_cluster = True
                    break

            # Create new cluster if not added
            if not added_to_cluster:
                resistance_clusters.append([swing])

        # Find clusters with 3+ tests
        for cluster in resistance_clusters:
            if len(cluster) >= 3:
                # Check if volumes are declining (distribution signature)
                volumes = [s["volume"] for s in cluster]
                volume_declining = all(volumes[i] > volumes[i+1] for i in range(len(volumes)-1))

                resistance_level = sum(s["price"] for s in cluster) / len(cluster)
                return {
                    "pattern": "distribution",
                    "resistance_level": round(resistance_level, 2),
                    "test_count": len(cluster),
                    "volume_declining": volume_declining
                }

        return None

    def _detect_accumulation(self, swing_lows: List[Dict]) -> Optional[Dict]:
        """Detect accumulation pattern - multiple tests of same support level"""
        if len(swing_lows) < 3:
            return None

        # Group swing lows by similar price levels (within 2% tolerance)
        tolerance = 0.02
        support_clusters = []

        for swing in swing_lows:
            price = swing["price"]
            added_to_cluster = False

            # Try to add to existing cluster
            for cluster in support_clusters:
                cluster_avg = sum(s["price"] for s in cluster) / len(cluster)
                if abs(price - cluster_avg) / cluster_avg <= tolerance:
                    cluster.append(swing)
                    added_to_cluster = True
                    break

            # Create new cluster if not added
            if not added_to_cluster:
                support_clusters.append([swing])

        # Find clusters with 3+ tests
        for cluster in support_clusters:
            if len(cluster) >= 3:
                # Check if volumes are declining (accumulation signature)
                volumes = [s["volume"] for s in cluster]
                volume_declining = all(volumes[i] > volumes[i+1] for i in range(len(volumes)-1))

                support_level = sum(s["price"] for s in cluster) / len(cluster)
                return {
                    "pattern": "accumulation",
                    "support_level": round(support_level, 2),
                    "test_count": len(cluster),
                    "volume_declining": volume_declining
                }

        return None

    def _detect_failed_breakout(self, ohlcv_data: List[List], swing_highs: List[Dict], swing_lows: List[Dict]) -> Optional[Dict]:
        """Detect failed breakouts (bull traps and bear traps)"""
        if len(ohlcv_data) < 5:
            return None

        # Look at recent candles (last 10)
        recent_candles = ohlcv_data[-10:]

        # Get recent resistance and support levels
        if not swing_highs and not swing_lows:
            return None

        # Check for bull trap (break above resistance then immediate reversal)
        if swing_highs:
            recent_resistance = max(s["price"] for s in swing_highs[-3:]) if len(swing_highs) >= 3 else swing_highs[-1]["price"]

            for i in range(len(recent_candles) - 3):
                high = recent_candles[i][2]
                close = recent_candles[i][4]

                # Check if broke above resistance
                if high > recent_resistance * 1.005:  # 0.5% above resistance
                    # Check if next 1-3 candles reversed back below
                    for j in range(i+1, min(i+4, len(recent_candles))):
                        if recent_candles[j][4] < recent_resistance * 0.995:  # Closed back below
                            return {
                                "pattern": "failed_breakout",
                                "type": "bull_trap",
                                "trap_level": round(recent_resistance, 2)
                            }

        # Check for bear trap (break below support then immediate reversal)
        if swing_lows:
            recent_support = min(s["price"] for s in swing_lows[-3:]) if len(swing_lows) >= 3 else swing_lows[-1]["price"]

            for i in range(len(recent_candles) - 3):
                low = recent_candles[i][3]
                close = recent_candles[i][4]

                # Check if broke below support
                if low < recent_support * 0.995:  # 0.5% below support
                    # Check if next 1-3 candles reversed back above
                    for j in range(i+1, min(i+4, len(recent_candles))):
                        if recent_candles[j][4] > recent_support * 1.005:  # Closed back above
                            return {
                                "pattern": "failed_breakout",
                                "type": "bear_trap",
                                "trap_level": round(recent_support, 2)
                            }

        return None

    def analyze_liquidity_zones(self, ohlcv_data: List[List], current_price: float, volume_24h: float) -> Dict:
        """
        Analyze support/resistance levels and volume to identify potential liquidity zones.

        Args:
            ohlcv_data: OHLCV candle data [timestamp, open, high, low, close, volume]
            current_price: Current BTC price
            volume_24h: 24-hour volume

        Returns:
            Dictionary with liquidity analysis including support/resistance levels and volume metrics
        """
        if not ohlcv_data or len(ohlcv_data) < 20 or current_price <= 0:
            return {
                "nearest_support": None,
                "nearest_resistance": None,
                "distance_to_support_pct": None,
                "distance_to_resistance_pct": None,
                "volume_ratio": None,
                "volume_spike": False,
                "potential_liquidity_zone": False
            }

        # Extract high, low, and volume from OHLCV data
        highs = [candle[2] for candle in ohlcv_data]  # High prices
        lows = [candle[3] for candle in ohlcv_data]   # Low prices
        volumes = [candle[5] for candle in ohlcv_data]  # Volumes

        # Find swing highs (resistance levels)
        # A swing high is a high that's higher than the 2 candles before and after it
        swing_highs = []
        for i in range(2, len(highs) - 2):
            if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and
                highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                swing_highs.append(highs[i])

        # Find swing lows (support levels)
        # A swing low is a low that's lower than the 2 candles before and after it
        swing_lows = []
        for i in range(2, len(lows) - 2):
            if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and
                lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                swing_lows.append(lows[i])

        # Find nearest support (highest swing low below current price)
        nearest_support = None
        supports_below = [level for level in swing_lows if level < current_price]
        if supports_below:
            nearest_support = max(supports_below)

        # Find nearest resistance (lowest swing high above current price)
        nearest_resistance = None
        resistances_above = [level for level in swing_highs if level > current_price]
        if resistances_above:
            nearest_resistance = min(resistances_above)

        # Calculate distance to support/resistance as percentage
        distance_to_support_pct = None
        if nearest_support:
            distance_to_support_pct = round(((current_price - nearest_support) / current_price) * 100, 2)

        distance_to_resistance_pct = None
        if nearest_resistance:
            distance_to_resistance_pct = round(((nearest_resistance - current_price) / current_price) * 100, 2)

        # Calculate volume metrics
        # Use last 20 candles to calculate average volume
        avg_volume = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)

        volume_ratio = None
        volume_spike = False
        if avg_volume > 0 and volume_24h > 0:
            volume_ratio = round(volume_24h / avg_volume, 2)
            volume_spike = volume_ratio > 2.0  # Spike if volume is >2x average

        # Determine if we're at a potential liquidity zone
        # Criteria: Close to support/resistance (within 5%) OR volume spike
        potential_liquidity_zone = False
        if distance_to_support_pct and distance_to_support_pct < 5.0:
            potential_liquidity_zone = True
        elif distance_to_resistance_pct and distance_to_resistance_pct < 5.0:
            potential_liquidity_zone = True
        elif volume_spike:
            potential_liquidity_zone = True

        return {
            "nearest_support": round(nearest_support, 2) if nearest_support else None,
            "nearest_resistance": round(nearest_resistance, 2) if nearest_resistance else None,
            "distance_to_support_pct": distance_to_support_pct,
            "distance_to_resistance_pct": distance_to_resistance_pct,
            "volume_ratio": volume_ratio,
            "volume_spike": volume_spike,
            "potential_liquidity_zone": potential_liquidity_zone
        }

    def fetch_market_data(self) -> MarketData:
        """Fetch comprehensive market data including technical indicators with multi-exchange failover"""
        exchange_error = None
        exchange_available = False

        # Try to fetch ticker with multi-exchange failover
        ticker_result = self._try_fetch_ticker_with_failover()

        if not ticker_result:
            # ALL exchanges failed (including CoinGecko)
            exchange_error = "All exchange APIs unavailable (Coinbase, Kraken, Binance.US, CoinGecko)"
            return self._get_fallback_market_data(exchange_error)

        try:
            # Extract ticker data
            ticker = ticker_result["ticker"]
            exchange_name = ticker_result["exchange"]
            symbol = ticker_result["symbol"]

            current_price = ticker.get('last', 0) or ticker.get('close', 0)

            # Get 24-hour volume
            if "volume_24h" in ticker_result:
                # CoinGecko provides volume directly
                volume_24h = ticker_result["volume_24h"]
            else:
                # For CCXT exchanges, try to get volume from hourly candles
                try:
                    ohlcv_1h = self.get_btc_ohlcv(timeframe='1h', limit=24)
                    volume_24h = sum(candle[5] for candle in ohlcv_1h) if ohlcv_1h else 0
                except Exception:
                    volume_24h = 0

            price_change_24h = ticker.get('change', 0) or 0
            price_change_percentage_24h = ticker.get('percentage', 0) or 0

            # Fetch OHLCV for technical indicators
            ohlcv = self.get_btc_ohlcv(timeframe='1d', limit=100)

            rsi_14 = None
            sma_5 = None
            sma_10 = None
            sma_20 = None
            sma_50 = None
            ema_9 = None
            price_vs_sma20 = None
            price_vs_sma50 = None

            if ohlcv and len(ohlcv) >= 50:
                closing_prices = [candle[4] for candle in ohlcv]  # Close price is index 4

                # Calculate RSI
                rsi_14 = self.calculate_rsi(closing_prices, period=14)

                # Calculate SMAs (short-term and long-term)
                sma_5 = self.calculate_sma(closing_prices, period=5)
                sma_10 = self.calculate_sma(closing_prices, period=10)
                sma_20 = self.calculate_sma(closing_prices, period=20)
                sma_50 = self.calculate_sma(closing_prices, period=50)

                # Calculate EMA
                ema_9 = self.calculate_ema(closing_prices, period=9)

                # Determine price position relative to SMAs
                if sma_20:
                    price_vs_sma20 = "above" if current_price > sma_20 else "below"
                if sma_50:
                    price_vs_sma50 = "above" if current_price > sma_50 else "below"

            # Analyze liquidity zones (support/resistance and volume)
            liquidity_analysis = self.analyze_liquidity_zones(ohlcv, current_price, volume_24h)

            # Detect Wyckoff patterns (accumulation/distribution)
            wyckoff_pattern = self.detect_wyckoff_patterns(ohlcv)

            # Fetch Fear & Greed Index
            fear_greed_value, fear_greed_class, fear_greed_error = self.get_fear_greed_index()

            exchange_available = True
            return MarketData(
                current_price=current_price,
                volume_24h=volume_24h,
                price_change_24h=price_change_24h,
                price_change_percentage_24h=price_change_percentage_24h,
                fear_greed_index=fear_greed_value,
                fear_greed_classification=fear_greed_class,
                rsi_14=rsi_14,
                sma_5=sma_5,
                sma_10=sma_10,
                sma_20=sma_20,
                sma_50=sma_50,
                ema_9=ema_9,
                price_vs_sma20=price_vs_sma20,
                price_vs_sma50=price_vs_sma50,
                nearest_support=liquidity_analysis["nearest_support"],
                nearest_resistance=liquidity_analysis["nearest_resistance"],
                distance_to_support_pct=liquidity_analysis["distance_to_support_pct"],
                distance_to_resistance_pct=liquidity_analysis["distance_to_resistance_pct"],
                volume_ratio=liquidity_analysis["volume_ratio"],
                volume_spike=liquidity_analysis["volume_spike"],
                potential_liquidity_zone=liquidity_analysis["potential_liquidity_zone"],
                wyckoff_patterns=wyckoff_pattern,
                exchange_available=True,
                exchange_name=exchange_name,  # Use the exchange that actually worked
                exchange_error=None,
                fear_greed_available=fear_greed_value is not None,
                fear_greed_error=fear_greed_error
            )

        except Exception as e:
            # This should rarely happen now since we have multi-exchange failover
            exchange_error = f"Error processing market data: {type(e).__name__} - {str(e)}"
            return self._get_fallback_market_data(exchange_error)

    def _get_fallback_market_data(self, exchange_error: Optional[str] = None) -> MarketData:
        """Return fallback market data when exchange is unavailable"""
        # Still try to get Fear & Greed Index
        fear_greed_value, fear_greed_class, fear_greed_error = self.get_fear_greed_index()

        return MarketData(
            current_price=0.0,
            volume_24h=0.0,
            price_change_24h=0.0,
            price_change_percentage_24h=0.0,
            fear_greed_index=fear_greed_value,
            fear_greed_classification=fear_greed_class,
            rsi_14=None,
            sma_5=None,
            sma_10=None,
            sma_20=None,
            sma_50=None,
            ema_9=None,
            price_vs_sma20=None,
            price_vs_sma50=None,
            wyckoff_patterns=None,
            exchange_available=False,
            exchange_name=self.exchange_name,
            exchange_error=exchange_error,
            fear_greed_available=fear_greed_value is not None,
            fear_greed_error=fear_greed_error
        )


class BitcoinNewsAggregator:
    """Aggregates Bitcoin news from various sources"""

    def __init__(self, newsapi_key: Optional[str] = None):
        self.newsapi_key = newsapi_key
        self.newsapi_url = "https://newsapi.org/v2/everything"

    def fetch_news(self, query: str = "Bitcoin OR BTC", max_articles: int = 10) -> List[NewsArticle]:
        """Fetch recent Bitcoin news articles"""
        articles = []

        # If NewsAPI key is provided, use it
        if self.newsapi_key:
            articles.extend(self._fetch_from_newsapi(query, max_articles))

        # Add fallback to free RSS sources if needed
        if len(articles) < max_articles:
            articles.extend(self._fetch_from_rss_feeds(max_articles - len(articles)))

        return articles[:max_articles]

    def _fetch_from_newsapi(self, query: str, max_articles: int) -> List[NewsArticle]:
        """Fetch news from NewsAPI"""
        params = {
            "q": query,
            "apiKey": self.newsapi_key,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles
        }

        try:
            response = requests.get(self.newsapi_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            articles = []
            for article in data.get("articles", []):
                articles.append(NewsArticle(
                    title=article.get("title", ""),
                    description=article.get("description", ""),
                    url=article.get("url", ""),
                    published_at=article.get("publishedAt", ""),
                    source=article.get("source", {}).get("name", "Unknown")
                ))
            return articles
        except Exception as e:
            print(f"Error fetching from NewsAPI: {e}")
            return []

    def _fetch_from_rss_feeds(self, max_articles: int) -> List[NewsArticle]:
        """Fetch news from multiple RSS feeds, filtering for Bitcoin-specific articles"""
        # Fetch from all sources simultaneously
        rss_feeds = [
            ("https://bitcoinmagazine.com/.rss/full/", "Bitcoin Magazine"),
            ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
            ("https://cointelegraph.com/rss", "Cointelegraph")
        ]

        all_articles = []

        # Fetch from all sources
        for feed_url, source_name in rss_feeds:
            try:
                feed = feedparser.parse(feed_url)

                for entry in feed.entries:
                    # Extract description from entry
                    description = ""
                    if hasattr(entry, 'summary'):
                        description = entry.summary
                    elif hasattr(entry, 'description'):
                        description = entry.description

                    title = entry.get("title", "")

                    # Filter: Only include articles about Bitcoin
                    if not self._is_bitcoin_related(title, description):
                        continue

                    # Extract published date
                    published_at = ""
                    published_timestamp = None

                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_timestamp = entry.published_parsed
                        published_at = entry.published
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_timestamp = entry.updated_parsed
                        published_at = entry.updated

                    article = NewsArticle(
                        title=title,
                        description=description,
                        url=entry.get("link", ""),
                        published_at=published_at,
                        source=source_name
                    )

                    # Store timestamp for sorting
                    article._timestamp = published_timestamp
                    all_articles.append(article)

            except Exception as e:
                print(f"Error fetching from {source_name} RSS: {e}")
                continue

        # Sort by published date (most recent first)
        all_articles.sort(
            key=lambda x: x._timestamp if hasattr(x, '_timestamp') and x._timestamp else (0,),
            reverse=True
        )

        # Return the most recent articles (keep _timestamp for display)
        return all_articles[:max_articles]

    def _is_bitcoin_related(self, title: str, description: str) -> bool:
        """
        Check if article is primarily about Bitcoin (not just mentioned in passing).
        Strict filtering to exclude articles primarily about other cryptocurrencies.
        """
        title_lower = title.lower()
        description_lower = description.lower()

        # List of altcoin tickers to exclude (if they appear in title, article is likely about them)
        altcoin_tickers = [
            'xrp', 'eth', 'ethereum', 'sol', 'solana', 'ada', 'cardano',
            'doge', 'dogecoin', 'bnb', 'binance', 'avax', 'avalanche',
            'dot', 'polkadot', 'matic', 'polygon', 'link', 'chainlink',
            'atom', 'cosmos', 'xlm', 'stellar', 'algo', 'algorand',
            'shib', 'shiba', 'uni', 'uniswap', 'ltc', 'litecoin',
            'popcat', 'pepe', 'bonk', 'floki', 'usdt', 'tether',
            'usdc', 'dai', 'busd'
        ]

        # If title contains altcoin ticker, exclude it (title indicates primary topic)
        # Even if Bitcoin is mentioned, multi-coin articles are not Bitcoin-focused
        for altcoin in altcoin_tickers:
            # Check for whole word match to avoid false positives
            if f' {altcoin} ' in f' {title_lower} ' or title_lower.startswith(f'{altcoin} ') or title_lower.endswith(f' {altcoin}'):
                # Exception: if this is a comma-separated list and Bitcoin is clearly first/primary
                # Skip articles that mention Bitcoin alongside other coins (e.g., "Bitcoin, XRP, and ETH fall")
                return False
            # Also check for ticker format like "ETH, SOL, ADA"
            if f'{altcoin},' in title_lower or f', {altcoin}' in title_lower:
                return False

        # Bitcoin must be mentioned in title OR description
        bitcoin_keywords = ["bitcoin", "btc"]
        has_bitcoin = any(keyword in title_lower or keyword in description_lower for keyword in bitcoin_keywords)

        if not has_bitcoin:
            return False

        # For stronger filtering: prefer articles with Bitcoin/BTC in the TITLE
        # Articles with Bitcoin only in description are often about other topics
        has_bitcoin_in_title = any(keyword in title_lower for keyword in bitcoin_keywords)

        # If Bitcoin is only in description (not title), be more skeptical
        if not has_bitcoin_in_title:
            # Check if description talks about multiple coins (likely general crypto news)
            multi_coin_indicators = ['crypto', 'altcoin', 'tokens', 'coins']
            if any(indicator in description_lower for indicator in multi_coin_indicators):
                # Only accept if Bitcoin is mentioned prominently (at start of description)
                description_start = description_lower[:100]
                if not any(keyword in description_start for keyword in bitcoin_keywords):
                    return False

        return True


class BitcoinSentimentAnalyzer:
    """Analyzes Bitcoin sentiment using Ollama AI"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client

    def analyze_articles(self, articles: List[NewsArticle], market_data: Optional[MarketData] = None, reversal_signal: Optional[Dict] = None) -> SentimentAnalysis:
        """Analyze sentiment from multiple news articles and market data"""
        if not articles:
            return SentimentAnalysis(
                sentiment="neutral",
                confidence=0.0,
                key_points=["No articles to analyze"],
                reasoning="No news articles provided for analysis"
            )

        # Prepare articles summary for analysis
        articles_text = self._prepare_articles_text(articles)

        # Prepare market data context
        market_context = ""
        if market_data and market_data.current_price > 0:
            market_context = f"""
Current Market Data:
- Price: ${market_data.current_price:,.2f}
- 24h Change: {market_data.price_change_percentage_24h:+.2f}%
- 24h Volume: {market_data.volume_24h:,.0f} BTC
"""
            if market_data.fear_greed_index:
                market_context += f"- Fear & Greed Index: {market_data.fear_greed_index} ({market_data.fear_greed_classification})\n"
            if market_data.rsi_14:
                market_context += f"- RSI(14): {market_data.rsi_14}\n"

            # Short-term moving averages
            if market_data.sma_5:
                market_context += f"- SMA(5): ${market_data.sma_5:,.2f}\n"
            if market_data.sma_10:
                market_context += f"- SMA(10): ${market_data.sma_10:,.2f}\n"
            if market_data.ema_9:
                market_context += f"- EMA(9): ${market_data.ema_9:,.2f}\n"

            # Medium/Long-term moving averages
            if market_data.sma_20:
                market_context += f"- SMA(20): ${market_data.sma_20:,.2f} (Price is {market_data.price_vs_sma20})\n"
            if market_data.sma_50:
                market_context += f"- SMA(50): ${market_data.sma_50:,.2f} (Price is {market_data.price_vs_sma50})\n"

            # Liquidity zones (support/resistance and volume analysis)
            if market_data.nearest_support or market_data.nearest_resistance:
                market_context += "\nLiquidity Zones:\n"
                if market_data.nearest_support:
                    market_context += f"- Nearest Support: ${market_data.nearest_support:,.2f} ({market_data.distance_to_support_pct:.2f}% below)\n"
                if market_data.nearest_resistance:
                    market_context += f"- Nearest Resistance: ${market_data.nearest_resistance:,.2f} ({market_data.distance_to_resistance_pct:.2f}% above)\n"
                if market_data.volume_ratio:
                    volume_status = "SPIKE" if market_data.volume_spike else "Normal"
                    market_context += f"- Volume vs 20-day Avg: {market_data.volume_ratio}x ({volume_status})\n"
                if market_data.potential_liquidity_zone:
                    market_context += "- ⚠️ AT POTENTIAL LIQUIDITY ZONE - High probability of price reaction\n"

            # Wyckoff pattern analysis (multi-timeframe)
            if market_data.wyckoff_patterns:
                wyckoff_found = False

                # Priority 1: Short-term patterns (most relevant for 12h)
                if market_data.wyckoff_patterns.get("short_term"):
                    pattern = market_data.wyckoff_patterns["short_term"]
                    if not wyckoff_found:
                        market_context += "\n📊 WYCKOFF PATTERNS DETECTED:\n"
                        wyckoff_found = True
                    market_context += "🔴 SHORT-TERM (20 candles - HIGHEST PRIORITY for 12h):\n"
                    market_context += self._format_wyckoff_pattern(pattern)

                # Priority 2: Medium-term patterns
                if market_data.wyckoff_patterns.get("medium_term"):
                    pattern = market_data.wyckoff_patterns["medium_term"]
                    if not wyckoff_found:
                        market_context += "\n📊 WYCKOFF PATTERNS DETECTED:\n"
                        wyckoff_found = True
                    market_context += "🟡 MEDIUM-TERM (50 candles - MEDIUM PRIORITY):\n"
                    market_context += self._format_wyckoff_pattern(pattern)

                # Priority 3: Long-term patterns (lowest priority, note distance)
                if market_data.wyckoff_patterns.get("long_term"):
                    pattern = market_data.wyckoff_patterns["long_term"]
                    if not wyckoff_found:
                        market_context += "\n📊 WYCKOFF PATTERNS DETECTED:\n"
                        wyckoff_found = True
                    market_context += "🟢 LONG-TERM (100 candles - LOW PRIORITY for 12h):\n"
                    market_context += self._format_wyckoff_pattern(pattern)
                    # Calculate distance for long-term patterns
                    if pattern.get("pattern") in ["distribution", "accumulation"]:
                        level_key = "resistance_level" if pattern["pattern"] == "distribution" else "support_level"
                        level = pattern.get(level_key)
                        if level:
                            distance_pct = abs((level - market_data.current_price) / market_data.current_price * 100)
                            if distance_pct > 10:
                                market_context += f"  ⚠️ NOTE: {distance_pct:.1f}% from current price - LESS RELEVANT for 12h trading\n"

        # Add reversal context if provided
        reversal_context = ""
        if reversal_signal and reversal_signal.get("type") != "none":
            reversal_context = "\n⚠️ REVERSAL SIGNAL DETECTED:\n"
            if reversal_signal["type"] == "oversold_bounce":
                reversal_context += f"- Type: OVERSOLD BOUNCE (Potential Bottom)\n"
                reversal_context += f"- Strength: {reversal_signal['strength'].upper()}\n"
                reversal_context += "- Extreme fear + oversold conditions often precede price bounces\n"
            elif reversal_signal["type"] == "overbought_reversal":
                reversal_context += f"- Type: OVERBOUGHT REVERSAL (Potential Top)\n"
                reversal_context += f"- Strength: {reversal_signal['strength'].upper()}\n"
                reversal_context += "- Extreme greed + overbought conditions often precede corrections\n"

        # Create analysis prompt
        system_prompt = """You are an expert Bitcoin trading analyst. Analyze news articles and market data to provide sentiment analysis in JSON format.
Your response must be valid JSON with this structure:
{
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": <number 0-100>,
    "key_points": [<list of key findings>],
    "reasoning": "<detailed explanation>"
}

SIGNAL PRIORITY for 12h trading (evaluate signals in this order):
1. HIGHEST PRIORITY: BOUNCE PATTERNS - If detected, this OVERRIDES all conflicting signals. These combine Wyckoff + price position + volume for high-probability setups.
2. HIGH PRIORITY: Short-term MAs (SMA5/10, EMA9), short-term Wyckoff patterns (if present), volume spikes, reversal signals
3. MEDIUM PRIORITY: RSI, liquidity zones, medium-term Wyckoff patterns
4. LOW PRIORITY: Long-term Wyckoff patterns (ONLY if within 10% of current price), Fear & Greed Index (use contrarian), SMA50

Consider the news sentiment, technical indicators (RSI, moving averages, Fear & Greed Index), liquidity zones (support/resistance levels), volume analysis, Wyckoff patterns, bounce patterns, and any reversal signals in your analysis.
Weight short-term signals more heavily for 12h predictions. Long-term patterns far from current price should be de-emphasized.
Reversal signals and liquidity zones are important as they often precede significant price movements.
Wyckoff patterns indicate smart money activity: Distribution (bearish - selling into strength), Accumulation (bullish - buying into weakness), and Failed Breakouts (traps/liquidity grabs).
When price is near support with high volume, it may indicate capitulation and a buying opportunity.
When price is near resistance, it may face selling pressure."""

        prompt = f"""Analyze the following Bitcoin news articles and market data to determine the overall market sentiment:

{articles_text}
{market_context}
{reversal_context}

Provide your analysis in JSON format as specified."""

        try:
            response = self.ollama.generate(prompt, system_prompt, temperature=0.3)
            analysis_data = self._parse_json_response(response)

            return SentimentAnalysis(
                sentiment=analysis_data.get("sentiment", "neutral").lower(),
                confidence=float(analysis_data.get("confidence", 50)),
                key_points=analysis_data.get("key_points", []),
                reasoning=analysis_data.get("reasoning", "")
            )
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
            return SentimentAnalysis(
                sentiment="neutral",
                confidence=0.0,
                key_points=[f"Analysis error: {str(e)}"],
                reasoning="Failed to analyze sentiment due to an error"
            )

    def _format_wyckoff_pattern(self, pattern: Dict) -> str:
        """Format a Wyckoff pattern for display in market context"""
        if not pattern:
            return ""

        output = ""
        if pattern["pattern"] == "distribution":
            output += f"  - Pattern: DISTRIBUTION (Bearish)\n"
            output += f"  - Resistance Level: ${pattern['resistance_level']:,.2f} tested {pattern['test_count']} times\n"
            if pattern.get('volume_declining'):
                output += "  - Volume declining on tests (classic distribution signature)\n"
            output += "  - Indicates smart money selling into strength - Bearish signal\n"
        elif pattern["pattern"] == "accumulation":
            output += f"  - Pattern: ACCUMULATION (Bullish)\n"
            output += f"  - Support Level: ${pattern['support_level']:,.2f} tested {pattern['test_count']} times\n"
            if pattern.get('volume_declining'):
                output += "  - Volume declining on tests (classic accumulation signature)\n"
            output += "  - Indicates smart money buying into weakness - Bullish signal\n"
        elif pattern["pattern"] == "failed_breakout":
            if pattern['type'] == "bull_trap":
                output += f"  - Pattern: BULL TRAP (Bearish)\n"
                output += f"  - Failed breakout above ${pattern['trap_level']:,.2f}\n"
                output += "  - Liquidity grab above resistance, likely to reverse down\n"
            elif pattern['type'] == "bear_trap":
                output += f"  - Pattern: BEAR TRAP (Bullish)\n"
                output += f"  - Failed breakdown below ${pattern['trap_level']:,.2f}\n"
                output += "  - Liquidity grab below support, likely to reverse up\n"
        return output

    def _prepare_articles_text(self, articles: List[NewsArticle]) -> str:
        """Prepare articles for analysis"""
        text_parts = []
        for i, article in enumerate(articles, 1):
            text_parts.append(f"""Article {i}:
Title: {article.title}
Description: {article.description}
Source: {article.source}
Published: {article.published_at}
""")
        return "\n".join(text_parts)

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from model response"""
        # Try to find JSON in the response
        try:
            # Try direct parsing first
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            else:
                # Try to find JSON object in the text
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
                raise ValueError("No valid JSON found in response")


class BitcoinTradingAdvisor:
    """Generates trading recommendations based on sentiment analysis"""

    def __init__(self, ollama_client: OllamaClient):
        self.ollama = ollama_client

    def detect_reversal_conditions(self, market_data: Optional[MarketData] = None) -> Dict:
        """
        Detect potential reversal conditions based on extreme RSI and Fear & Greed values.

        Returns:
            Dict with keys:
                - type: "oversold_bounce", "overbought_reversal", or "none"
                - strength: "strong", "moderate", "weak" (only if reversal detected)
        """
        if not market_data or not market_data.rsi_14 or not market_data.fear_greed_index:
            return {"type": "none"}

        rsi = market_data.rsi_14
        fear_greed = market_data.fear_greed_index

        # Oversold Reversal (potential bottom)
        if rsi < 30 and fear_greed < 25:
            # Both extremely oversold - strong reversal signal
            if rsi < 25 and fear_greed < 20:
                return {"type": "oversold_bounce", "strength": "strong"}
            elif rsi < 27 or fear_greed < 22:
                return {"type": "oversold_bounce", "strength": "moderate"}
            else:
                return {"type": "oversold_bounce", "strength": "weak"}

        # Overbought Reversal (potential top)
        elif rsi > 70 and fear_greed > 75:
            # Both extremely overbought - strong reversal signal
            if rsi > 75 and fear_greed > 80:
                return {"type": "overbought_reversal", "strength": "strong"}
            elif rsi > 72 or fear_greed > 77:
                return {"type": "overbought_reversal", "strength": "moderate"}
            else:
                return {"type": "overbought_reversal", "strength": "weak"}

        return {"type": "none"}

    def detect_bounce_patterns(self, market_data: Optional[MarketData] = None) -> Dict:
        """
        Detect classic bounce setups combining Wyckoff patterns with price position vs MAs.

        PATTERN 1 - OVERSOLD BOUNCE (Bullish):
        - Wyckoff shows Accumulation or Spring phase (any timeframe)
        - Price below SMA(20) OR SMA(50) (oversold)
        - Volume > 1.5x average (capitulation/accumulation confirmation)
        Signal: Strong BUY - Smart money buying oversold dip

        PATTERN 2 - OVERBOUGHT REJECTION (Bearish):
        - Wyckoff shows Distribution or Upthrust phase (any timeframe)
        - Price above SMA(20) OR SMA(50) (overbought)
        - Volume > 1.5x average (distribution/exhaustion confirmation)
        Signal: Strong SELL - Smart money selling overbought rally

        PATTERN 3 - ACCUMULATION CONSOLIDATION (Bullish):
        - Wyckoff shows Accumulation phase (any timeframe)
        - Price above SMA(20) OR SMA(50) (consolidating)
        - Volume > 1.5x average (smart money accumulation confirmation)
        Signal: Strong BUY - Bullish continuation/breakout setup

        PATTERN 4 - DISTRIBUTION BREAKDOWN (Bearish):
        - Wyckoff shows Distribution phase (any timeframe)
        - Price below SMA(20) OR SMA(50) (breaking down)
        - Volume > 1.5x average (distribution confirmation)
        Signal: Strong SELL - Bearish continuation/breakdown setup

        Returns:
            Dict with keys:
                - pattern: "oversold_bounce", "overbought_rejection",
                          "accumulation_consolidation", "distribution_breakdown", or "none"
                - strength: "strong" or "moderate"
                - signal: "BUY", "SELL", or None
                - reasoning: explanation string
                - wyckoff_timeframe: which timeframe triggered (if applicable)
        """
        if not market_data:
            return {"pattern": "none", "strength": None, "signal": None, "reasoning": "No market data available"}

        # Check if we have required data
        if not market_data.wyckoff_patterns or not market_data.current_price:
            return {"pattern": "none", "strength": None, "signal": None, "reasoning": "Insufficient data for pattern detection"}

        # Check for Wyckoff accumulation or distribution across all timeframes
        wyckoff_accumulation = False
        wyckoff_distribution = False
        wyckoff_timeframe = None

        # Priority: short-term > medium-term > long-term
        for timeframe in ["short_term", "medium_term", "long_term"]:
            pattern = market_data.wyckoff_patterns.get(timeframe)
            if pattern:
                if pattern.get("pattern") == "accumulation":
                    wyckoff_accumulation = True
                    wyckoff_timeframe = timeframe
                    break
                elif pattern.get("pattern") == "distribution":
                    wyckoff_distribution = True
                    wyckoff_timeframe = timeframe
                    break
                elif pattern.get("pattern") == "failed_breakout":
                    # Bear trap = bullish (accumulation-like)
                    # Bull trap = bearish (distribution-like)
                    trap_type = pattern.get("type")
                    if trap_type == "bear_trap":
                        wyckoff_accumulation = True
                        wyckoff_timeframe = timeframe
                        break
                    elif trap_type == "bull_trap":
                        wyckoff_distribution = True
                        wyckoff_timeframe = timeframe
                        break

        # Check price position vs moving averages
        price_oversold = False
        price_overbought = False
        ma_reference = None

        if market_data.price_vs_sma20 == "below":
            price_oversold = True
            ma_reference = "SMA(20)"
        elif market_data.price_vs_sma50 == "below":
            price_oversold = True
            ma_reference = "SMA(50)"
        elif market_data.price_vs_sma20 == "above":
            price_overbought = True
            ma_reference = "SMA(20)"
        elif market_data.price_vs_sma50 == "above":
            price_overbought = True
            ma_reference = "SMA(50)"

        # Check volume confirmation (volume spike = high volume)
        volume_confirmation = market_data.volume_spike if market_data.volume_spike is not None else False
        volume_ratio = market_data.volume_ratio if market_data.volume_ratio else 1.0

        # PATTERN 1: OVERSOLD BOUNCE (Bullish)
        if wyckoff_accumulation and price_oversold:
            # RSI SAFETY CHECK for OVERSOLD BOUNCE
            if market_data.rsi_14 and market_data.rsi_14 > 60:
                # RSI is overbought - don't buy into it
                return {
                    "pattern": "none",
                    "strength": None,
                    "signal": None,
                    "reasoning": f"Oversold bounce pattern detected BUT RSI {market_data.rsi_14:.1f} is overbought (>60) - BLOCKED for safety"
                }

            # Pattern is valid
            strength = "strong" if volume_confirmation else "moderate"
            reasoning = f"OVERSOLD BOUNCE detected on {wyckoff_timeframe}: "
            reasoning += f"Wyckoff accumulation pattern + price below {ma_reference} (oversold)"
            if volume_confirmation:
                reasoning += f" + high volume ({volume_ratio:.1f}x avg) confirms capitulation/accumulation. "
                reasoning += "Smart money buying the dip - bounce highly probable."
            else:
                reasoning += ". Moderate setup - volume confirmation would strengthen signal."

            return {
                "pattern": "oversold_bounce",
                "strength": strength,
                "signal": "BUY",
                "reasoning": reasoning,
                "wyckoff_timeframe": wyckoff_timeframe
            }

        # PATTERN 2: OVERBOUGHT REJECTION (Bearish)
        if wyckoff_distribution and price_overbought:
            # RSI SAFETY CHECK for OVERBOUGHT REJECTION
            if market_data.rsi_14 and market_data.rsi_14 < 40:
                # RSI is oversold - don't sell into it
                return {
                    "pattern": "none",
                    "strength": None,
                    "signal": None,
                    "reasoning": f"Overbought rejection pattern detected BUT RSI {market_data.rsi_14:.1f} is oversold (<40) - BLOCKED for safety"
                }

            # Pattern is valid
            strength = "strong" if volume_confirmation else "moderate"
            reasoning = f"OVERBOUGHT REJECTION detected on {wyckoff_timeframe}: "
            reasoning += f"Wyckoff distribution pattern + price above {ma_reference} (overbought)"
            if volume_confirmation:
                reasoning += f" + high volume ({volume_ratio:.1f}x avg) confirms distribution/exhaustion. "
                reasoning += "Smart money selling the rally - rejection highly probable."
            else:
                reasoning += ". Moderate setup - volume confirmation would strengthen signal."

            return {
                "pattern": "overbought_rejection",
                "strength": strength,
                "signal": "SELL",
                "reasoning": reasoning,
                "wyckoff_timeframe": wyckoff_timeframe
            }

        # PATTERN 3: ACCUMULATION CONSOLIDATION (Bullish - price above MAs)
        if wyckoff_accumulation and price_overbought:
            # Accumulation happening at higher prices = retest/consolidation
            strength = "strong" if volume_confirmation else "moderate"
            reasoning = f"ACCUMULATION CONSOLIDATION detected on {wyckoff_timeframe}: "
            reasoning += f"Wyckoff accumulation pattern + price above {ma_reference} (consolidating gains)"
            if volume_confirmation:
                reasoning += f" + high volume ({volume_ratio:.1f}x avg) confirms smart money accumulation. "
                reasoning += "Bullish continuation setup - breakout likely."
            else:
                reasoning += ". Moderate setup - volume confirmation would strengthen signal."

            return {
                "pattern": "accumulation_consolidation",
                "strength": strength,
                "signal": "BUY",
                "reasoning": reasoning,
                "wyckoff_timeframe": wyckoff_timeframe
            }

        # PATTERN 4: DISTRIBUTION BREAKDOWN (Bearish - price below MAs)
        if wyckoff_distribution and price_oversold:
            # RSI SAFETY CHECK for DISTRIBUTION BREAKDOWN
            if market_data.rsi_14 and market_data.rsi_14 < 40:
                # RSI too oversold - risky to sell, might bounce
                return {
                    "pattern": "none",
                    "strength": None,
                    "signal": None,
                    "reasoning": f"Distribution breakdown pattern detected BUT RSI {market_data.rsi_14:.1f} is oversold (<40) - high bounce risk, BLOCKED"
                }

            # Pattern is valid
            # Distribution happening at lower prices = breakdown/capitulation
            strength = "strong" if volume_confirmation else "moderate"
            reasoning = f"DISTRIBUTION BREAKDOWN detected on {wyckoff_timeframe}: "
            reasoning += f"Wyckoff distribution pattern + price below {ma_reference} (breaking down)"
            if volume_confirmation:
                reasoning += f" + high volume ({volume_ratio:.1f}x avg) confirms distribution. "
                reasoning += "Bearish continuation setup - further downside likely."
            else:
                reasoning += ". Moderate setup - volume confirmation would strengthen signal."

            return {
                "pattern": "distribution_breakdown",
                "strength": strength,
                "signal": "SELL",
                "reasoning": reasoning,
                "wyckoff_timeframe": wyckoff_timeframe
            }

        # No pattern detected
        return {"pattern": "none", "strength": None, "signal": None, "reasoning": "No bounce pattern detected"}

    def _format_wyckoff_pattern(self, pattern: Dict) -> str:
        """Format a Wyckoff pattern for display in market context"""
        if not pattern:
            return ""

        output = ""
        if pattern["pattern"] == "distribution":
            output += f"  - Pattern: DISTRIBUTION (Bearish)\n"
            output += f"  - Resistance Level: ${pattern['resistance_level']:,.2f} tested {pattern['test_count']} times\n"
            if pattern.get('volume_declining'):
                output += "  - Volume declining on tests (classic distribution signature)\n"
            output += "  - Indicates smart money selling into strength - Bearish signal\n"
        elif pattern["pattern"] == "accumulation":
            output += f"  - Pattern: ACCUMULATION (Bullish)\n"
            output += f"  - Support Level: ${pattern['support_level']:,.2f} tested {pattern['test_count']} times\n"
            if pattern.get('volume_declining'):
                output += "  - Volume declining on tests (classic accumulation signature)\n"
            output += "  - Indicates smart money buying into weakness - Bullish signal\n"
        elif pattern["pattern"] == "failed_breakout":
            if pattern['type'] == "bull_trap":
                output += f"  - Pattern: BULL TRAP (Bearish)\n"
                output += f"  - Failed breakout above ${pattern['trap_level']:,.2f}\n"
                output += "  - Liquidity grab above resistance, likely to reverse down\n"
            elif pattern['type'] == "bear_trap":
                output += f"  - Pattern: BEAR TRAP (Bullish)\n"
                output += f"  - Failed breakdown below ${pattern['trap_level']:,.2f}\n"
                output += "  - Liquidity grab below support, likely to reverse up\n"
        return output

    def _print_wyckoff_pattern(self, pattern: Dict, indent: str = "  ") -> None:
        """Print a Wyckoff pattern with the specified indentation"""
        if not pattern:
            return

        if pattern["pattern"] == "distribution":
            print(f"{indent}Pattern: DISTRIBUTION (BEARISH)")
            print(f"{indent}Resistance: ${pattern['resistance_level']:,.2f} tested {pattern['test_count']}x")
            if pattern.get('volume_declining'):
                print(f"{indent}Volume: Declining (classic distribution)")
            print(f"{indent}Signal: Smart money selling - BEARISH")
        elif pattern["pattern"] == "accumulation":
            print(f"{indent}Pattern: ACCUMULATION (BULLISH)")
            print(f"{indent}Support: ${pattern['support_level']:,.2f} tested {pattern['test_count']}x")
            if pattern.get('volume_declining'):
                print(f"{indent}Volume: Declining (classic accumulation)")
            print(f"{indent}Signal: Smart money buying - BULLISH")
        elif pattern["pattern"] == "failed_breakout":
            trap_type = pattern.get('type', '').upper()
            print(f"{indent}Pattern: {trap_type}")
            print(f"{indent}Trap Level: ${pattern.get('trap_level', 0):,.2f}")
            if pattern.get('type') == "bull_trap":
                print(f"{indent}Signal: Failed breakout - BEARISH")
            else:
                print(f"{indent}Signal: Failed breakdown - BULLISH")

    def generate_recommendation(
        self,
        sentiment: SentimentAnalysis,
        current_price: float,
        portfolio_value: float,
        risk_tolerance: str = "medium",  # low, medium, high
        market_data: Optional[MarketData] = None
    ) -> TradingRecommendation:
        """Generate trading recommendation based on sentiment and market data"""

        # Detect reversal conditions first
        reversal = self.detect_reversal_conditions(market_data)

        # Detect bounce patterns (highest priority)
        bounce_pattern = self.detect_bounce_patterns(market_data)

        system_prompt = """You are an expert Bitcoin trading advisor. Generate trading recommendations in JSON format.
Your response must be valid JSON with this structure:
{
    "action": "buy" | "sell" | "hold",
    "confidence": <number 0-100>,
    "entry_price": <number or null>,
    "stop_loss": <number or null>,
    "take_profit": <number or null>,
    "position_size_percentage": <number 0-100>,
    "reasoning": "<detailed explanation>"
}

SIGNAL PRIORITY for 12h trading (evaluate signals in this order):
1. HIGHEST PRIORITY: BOUNCE PATTERNS - If detected, this OVERRIDES all conflicting signals. These combine Wyckoff + price position + volume for high-probability setups.
2. HIGH PRIORITY: Short-term MAs (SMA5/10, EMA9), short-term Wyckoff patterns (if present), volume spikes, reversal signals
3. MEDIUM PRIORITY: RSI, liquidity zones, medium-term Wyckoff patterns
4. LOW PRIORITY: Long-term Wyckoff patterns (ONLY if within 10% of current price), Fear & Greed Index (use contrarian), SMA50

Use technical indicators (RSI, moving averages), market sentiment (Fear & Greed), liquidity zones (support/resistance), volume analysis, Wyckoff patterns, bounce patterns, and reversal signals to inform your decision.
Weight short-term signals more heavily for 12h predictions. Long-term patterns far from current price (>10% away) should be de-emphasized or ignored.
Pay special attention to reversal conditions, short-term Wyckoff patterns, and liquidity zones as they often precede significant price movements.
Wyckoff patterns reveal smart money activity: Distribution (bearish), Accumulation (bullish), Failed Breakouts (traps).
Use support levels for stop-loss placement and resistance levels for take-profit targets.
Consider volume spikes as potential capitulation or exhaustion signals."""

        # Prepare market data context
        market_context = ""
        if market_data and market_data.current_price > 0:
            market_context = f"\nTechnical Indicators:\n"
            if market_data.rsi_14:
                market_context += f"- RSI(14): {market_data.rsi_14} (>70 overbought, <30 oversold)\n"

            # Short-term moving averages for 4-hour signals
            if market_data.sma_5:
                market_context += f"- SMA(5): ${market_data.sma_5:,.2f}\n"
            if market_data.sma_10:
                market_context += f"- SMA(10): ${market_data.sma_10:,.2f}\n"
            if market_data.ema_9:
                market_context += f"- EMA(9): ${market_data.ema_9:,.2f}\n"

            # Medium/Long-term moving averages
            if market_data.sma_20:
                market_context += f"- SMA(20): ${market_data.sma_20:,.2f}\n"
            if market_data.sma_50:
                market_context += f"- SMA(50): ${market_data.sma_50:,.2f}\n"

            if market_data.price_vs_sma20:
                market_context += f"- Price vs SMA(20): {market_data.price_vs_sma20}\n"
            if market_data.price_vs_sma50:
                market_context += f"- Price vs SMA(50): {market_data.price_vs_sma50}\n"
            if market_data.fear_greed_index:
                market_context += f"- Fear & Greed Index: {market_data.fear_greed_index} ({market_data.fear_greed_classification})\n"
            if market_data.price_change_percentage_24h != 0:
                market_context += f"- 24h Price Change: {market_data.price_change_percentage_24h:+.2f}%\n"

            # Liquidity zones
            if market_data.nearest_support or market_data.nearest_resistance:
                market_context += "\nLiquidity Zones & Market Structure:\n"
                if market_data.nearest_support:
                    market_context += f"- Nearest Support: ${market_data.nearest_support:,.2f} ({market_data.distance_to_support_pct:.2f}% below) - Consider for stop-loss\n"
                if market_data.nearest_resistance:
                    market_context += f"- Nearest Resistance: ${market_data.nearest_resistance:,.2f} ({market_data.distance_to_resistance_pct:.2f}% above) - Consider for take-profit\n"
                if market_data.volume_ratio:
                    volume_status = "HIGH VOLUME SPIKE" if market_data.volume_spike else "Normal"
                    market_context += f"- Volume vs 20-day Avg: {market_data.volume_ratio}x ({volume_status})\n"
                if market_data.potential_liquidity_zone:
                    market_context += "- ⚠️ AT LIQUIDITY ZONE - Price likely to react here (bounce or break)\n"

            # Bounce pattern analysis (HIGHEST PRIORITY)
            if bounce_pattern["pattern"] != "none":
                market_context += "\n🚨 BOUNCE PATTERN DETECTED (HIGHEST PRIORITY - OVERRIDES CONFLICTING SIGNALS):\n"
                market_context += f"- Pattern: {bounce_pattern['pattern'].upper().replace('_', ' ')}\n"
                market_context += f"- Signal: {bounce_pattern['signal']} ({bounce_pattern['strength'].upper()})\n"
                market_context += f"- Timeframe: {bounce_pattern['wyckoff_timeframe'].upper().replace('_', ' ')}\n"
                market_context += f"- Reasoning: {bounce_pattern['reasoning']}\n"
                market_context += "- ⚠️ This is a high-probability setup - strongly consider this signal!\n"

            # Wyckoff pattern analysis (multi-timeframe)
            if market_data.wyckoff_patterns:
                wyckoff_found = False

                # Priority 1: Short-term patterns (most relevant for 12h)
                if market_data.wyckoff_patterns.get("short_term"):
                    pattern = market_data.wyckoff_patterns["short_term"]
                    if not wyckoff_found:
                        market_context += "\n📊 WYCKOFF PATTERNS DETECTED:\n"
                        wyckoff_found = True
                    market_context += "🔴 SHORT-TERM (20 candles - HIGHEST PRIORITY for 12h):\n"
                    market_context += self._format_wyckoff_pattern(pattern)

                # Priority 2: Medium-term patterns
                if market_data.wyckoff_patterns.get("medium_term"):
                    pattern = market_data.wyckoff_patterns["medium_term"]
                    if not wyckoff_found:
                        market_context += "\n📊 WYCKOFF PATTERNS DETECTED:\n"
                        wyckoff_found = True
                    market_context += "🟡 MEDIUM-TERM (50 candles - MEDIUM PRIORITY):\n"
                    market_context += self._format_wyckoff_pattern(pattern)

                # Priority 3: Long-term patterns (lowest priority, note distance)
                if market_data.wyckoff_patterns.get("long_term"):
                    pattern = market_data.wyckoff_patterns["long_term"]
                    if not wyckoff_found:
                        market_context += "\n📊 WYCKOFF PATTERNS DETECTED:\n"
                        wyckoff_found = True
                    market_context += "🟢 LONG-TERM (100 candles - LOW PRIORITY for 12h):\n"
                    market_context += self._format_wyckoff_pattern(pattern)
                    # Calculate distance for long-term patterns
                    if pattern.get("pattern") in ["distribution", "accumulation"]:
                        level_key = "resistance_level" if pattern["pattern"] == "distribution" else "support_level"
                        level = pattern.get(level_key)
                        if level:
                            distance_pct = abs((level - market_data.current_price) / market_data.current_price * 100)
                            if distance_pct > 10:
                                market_context += f"  ⚠️ NOTE: {distance_pct:.1f}% from current price - LESS RELEVANT for 12h trading\n"

        # Add reversal context
        reversal_context = ""
        if reversal["type"] != "none":
            reversal_context = f"\n⚠️ REVERSAL SIGNAL DETECTED:\n"
            if reversal["type"] == "oversold_bounce":
                reversal_context += f"- Type: OVERSOLD BOUNCE (Potential Bottom)\n"
                reversal_context += f"- Strength: {reversal['strength'].upper()}\n"
                reversal_context += f"- Market is in extreme fear with oversold RSI - potential buying opportunity\n"
            elif reversal["type"] == "overbought_reversal":
                reversal_context += f"- Type: OVERBOUGHT REVERSAL (Potential Top)\n"
                reversal_context += f"- Strength: {reversal['strength'].upper()}\n"
                reversal_context += f"- Market is in extreme greed with overbought RSI - potential selling opportunity\n"

        prompt = f"""Based on the following sentiment analysis and market data, generate a trading recommendation for Bitcoin:

Sentiment Analysis:
- Sentiment: {sentiment.sentiment}
- Confidence: {sentiment.confidence}%
- Key Points: {', '.join(sentiment.key_points)}
- Reasoning: {sentiment.reasoning}

Current Market Conditions:
- BTC Current Price: ${current_price:,.2f}
- Portfolio Value: ${portfolio_value:,.2f}
- Risk Tolerance: {risk_tolerance}
{market_context}
{reversal_context}

Provide a trading recommendation with proper risk management (stop loss, take profit, position sizing).
Consider the risk tolerance when determining position size:
- Low risk: 1-3% of portfolio
- Medium risk: 3-7% of portfolio
- High risk: 7-15% of portfolio

CRITICAL: If a BOUNCE PATTERN is detected, this is the HIGHEST PRIORITY signal and OVERRIDES all conflicting signals. These are high-probability setups combining Wyckoff smart money activity + price position + volume confirmation.
IMPORTANT: If a reversal signal is detected, consider it carefully in your recommendation as these often precede significant price movements.

Provide your recommendation in JSON format as specified."""

        try:
            response = self.ollama.generate(prompt, system_prompt, temperature=0.3)
            rec_data = self._parse_json_response(response)

            return TradingRecommendation(
                action=rec_data.get("action", "hold").lower(),
                confidence=float(rec_data.get("confidence", 50)),
                entry_price=rec_data.get("entry_price"),
                stop_loss=rec_data.get("stop_loss"),
                take_profit=rec_data.get("take_profit"),
                position_size_percentage=float(rec_data.get("position_size_percentage", 5)),
                reasoning=rec_data.get("reasoning", "")
            )
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return TradingRecommendation(
                action="hold",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size_percentage=0.0,
                reasoning=f"Failed to generate recommendation: {str(e)}"
            )

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from model response"""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                json_str = response[json_start:json_end].strip()
                return json.loads(json_str)
            else:
                start = response.find("{")
                end = response.rfind("}") + 1
                if start != -1 and end > start:
                    return json.loads(response[start:end])
                raise ValueError("No valid JSON found in response")

    def validate_recommendation_consistency(
        self,
        recommendation: TradingRecommendation,
        bounce_pattern: Dict,
        wyckoff_patterns: Dict,
        reversal_signal: Dict,
        market_data: MarketData
    ) -> Dict:
        """
        Validate that AI recommendation is consistent with detected patterns.
        Prevents hallucinations where AI claims patterns that don't exist.

        Returns dict with:
            - is_valid: bool
            - warnings: List[str] of inconsistencies found
            - overridden: bool (if recommendation was changed)
        """
        warnings = []

        # Check bounce pattern consistency
        if bounce_pattern.get('pattern') == 'none':
            # AI should NOT claim bounce pattern in reasoning
            reasoning_lower = recommendation.reasoning.lower()
            if any(phrase in reasoning_lower for phrase in ['bounce pattern detected', 'bounce pattern is', 'oversold bounce', 'overbought rejection']):
                warnings.append("AI claimed bounce pattern but none detected - hallucination!")

        # Check Wyckoff pattern consistency
        has_bullish_wyckoff = False
        has_bearish_wyckoff = False

        for timeframe, pattern in wyckoff_patterns.items():
            if pattern:
                if pattern.get('pattern') in ['accumulation', 'bear_trap']:
                    has_bullish_wyckoff = True
                elif pattern.get('pattern') in ['distribution', 'bull_trap']:
                    has_bearish_wyckoff = True

        # If AI recommends BUY but no bullish signals exist, flag it
        if recommendation.action == "buy" and not has_bullish_wyckoff and bounce_pattern.get('pattern') == 'none':
            if reversal_signal.get('type') == 'none':
                warnings.append(f"BUY recommendation but no bullish Wyckoff, bounce, or reversal detected")

        # If AI recommends SELL but no bearish signals exist, flag it
        if recommendation.action == "sell" and not has_bearish_wyckoff and bounce_pattern.get('pattern') == 'none':
            if reversal_signal.get('type') == 'none':
                warnings.append(f"SELL recommendation but no bearish Wyckoff, bounce, or reversal detected")

        return {
            'is_valid': len(warnings) == 0,
            'warnings': warnings,
            'overridden': False
        }

    def validate_and_fix_recommendation(
        self,
        recommendation: TradingRecommendation,
        market_data: MarketData,
        atr_hourly: Optional[float] = None
    ) -> tuple[TradingRecommendation, Dict]:
        """
        Validate and fix AI-generated recommendation to ensure realistic targets.

        Args:
            recommendation: The AI-generated recommendation
            market_data: Current market data
            atr_hourly: ATR calculated from hourly data (if available)

        Returns:
            Tuple of (fixed_recommendation, validation_info)
        """
        validation_info = {
            "validation_applied": False,
            "fixes_applied": [],
            "ai_original_target": None,
            "target_overridden": False
        }

        # Skip validation for HOLD actions
        if recommendation.action.lower() == "hold":
            return recommendation, validation_info

        validation_info["validation_applied"] = True
        current_price = market_data.current_price

        # 1. Fix Missing Entry Price
        if recommendation.entry_price is None or recommendation.entry_price <= 0:
            recommendation.entry_price = current_price
            validation_info["fixes_applied"].append("entry_price_set_to_current")

        entry_price = recommendation.entry_price

        # 2. Calculate ATR-Based Target Constraints
        atr_based_target = None
        if atr_hourly and atr_hourly > 0:
            if recommendation.action.lower() == "sell":
                atr_based_target = entry_price - (4 * atr_hourly)
            else:  # buy
                atr_based_target = entry_price + (4 * atr_hourly)

        # 3. Validate and Fix Take-Profit Target
        if recommendation.take_profit and recommendation.take_profit > 0:
            # Calculate percentage change from entry to target
            pct_change = abs((recommendation.take_profit - entry_price) / entry_price) * 100

            # Check if target is unrealistic (>15% from entry in 12h window)
            if pct_change > 15.0:
                validation_info["ai_original_target"] = recommendation.take_profit

                if atr_based_target:
                    recommendation.take_profit = round(atr_based_target, 2)
                    validation_info["fixes_applied"].append("target_overridden_atr_based")
                    validation_info["target_overridden"] = True
                else:
                    # Fallback: cap at 15% if no ATR available
                    if recommendation.action.lower() == "sell":
                        recommendation.take_profit = round(entry_price * 0.85, 2)
                    else:  # buy
                        recommendation.take_profit = round(entry_price * 1.15, 2)
                    validation_info["fixes_applied"].append("target_capped_at_15_percent")
                    validation_info["target_overridden"] = True

            # Validate target is in correct direction
            if recommendation.action.lower() == "buy" and recommendation.take_profit <= entry_price:
                if atr_based_target:
                    recommendation.take_profit = round(atr_based_target, 2)
                else:
                    recommendation.take_profit = round(entry_price * 1.05, 2)
                validation_info["fixes_applied"].append("buy_target_fixed_direction")

            elif recommendation.action.lower() == "sell" and recommendation.take_profit >= entry_price:
                if atr_based_target:
                    recommendation.take_profit = round(atr_based_target, 2)
                else:
                    recommendation.take_profit = round(entry_price * 0.95, 2)
                validation_info["fixes_applied"].append("sell_target_fixed_direction")

        # 4. Validate and Fix Stop-Loss Direction
        if recommendation.stop_loss and recommendation.stop_loss > 0:
            stop_fixed = False

            # For SELL: stop must be > entry (protect against upside)
            if recommendation.action.lower() == "sell" and recommendation.stop_loss <= entry_price:
                if atr_hourly:
                    recommendation.stop_loss = round(entry_price + (2.5 * atr_hourly), 2)
                else:
                    recommendation.stop_loss = round(entry_price * 1.03, 2)
                validation_info["fixes_applied"].append("sell_stop_fixed_direction")
                stop_fixed = True

            # For BUY: stop must be < entry (protect against downside)
            elif recommendation.action.lower() == "buy" and recommendation.stop_loss >= entry_price:
                if atr_hourly:
                    recommendation.stop_loss = round(entry_price - (2.5 * atr_hourly), 2)
                else:
                    recommendation.stop_loss = round(entry_price * 0.97, 2)
                validation_info["fixes_applied"].append("buy_stop_fixed_direction")
                stop_fixed = True

        return recommendation, validation_info

    def calculate_position_size(
        self,
        recommendation: TradingRecommendation,
        portfolio_value: float,
        current_price: float
    ) -> Dict[str, float]:
        """Calculate specific position size details"""
        position_value = portfolio_value * (recommendation.position_size_percentage / 100)
        btc_amount = position_value / current_price

        return {
            "position_value_usd": position_value,
            "btc_amount": btc_amount,
            "percentage_of_portfolio": recommendation.position_size_percentage,
            "entry_price": recommendation.entry_price or current_price,
            "stop_loss": recommendation.stop_loss,
            "take_profit": recommendation.take_profit,
            "max_loss_usd": abs(position_value * ((current_price - (recommendation.stop_loss or current_price)) / current_price)) if recommendation.stop_loss else None,
            "potential_profit_usd": abs(position_value * (((recommendation.take_profit or current_price) - current_price) / current_price)) if recommendation.take_profit else None
        }


class BitcoinTradingBot:
    """Main trading bot that orchestrates the entire analysis process"""

    def __init__(
        self,
        ollama_model: str = "qwen2.5-coder:32b",
        newsapi_key: Optional[str] = None,
        ollama_base_url: str = "http://localhost:11434"
    ):
        self.ollama = OllamaClient(model=ollama_model, base_url=ollama_base_url)
        self.news_aggregator = BitcoinNewsAggregator(newsapi_key)
        self.sentiment_analyzer = BitcoinSentimentAnalyzer(self.ollama)
        self.trading_advisor = BitcoinTradingAdvisor(self.ollama)
        self.market_data_fetcher = MarketDataFetcher()

    def run_analysis(
        self,
        portfolio_value: float,
        risk_tolerance: str = "medium",
        max_articles: int = 10
    ) -> Dict:
        """Run complete trading analysis"""
        print("=" * 80)
        print("Bitcoin Trading Sentiment Analyzer")
        print("=" * 80)
        print()

        # Check if Ollama model is available
        print(f"Checking Ollama model availability: {self.ollama.model}...")
        if not self.ollama.check_model_availability():
            print(f"�  Warning: Model {self.ollama.model} may not be available")
            print(f"   Make sure Ollama is running and the model is pulled:")
            print(f"   ollama pull {self.ollama.model}")
        else:
            print(f" Model {self.ollama.model} is available")
        print()

        # Fetch market data (includes price, volume, technical indicators, Fear & Greed)
        print("Fetching real-time market data...")
        market_data = self.market_data_fetcher.fetch_market_data()
        current_price = market_data.current_price

        # Determine and display analysis mode
        mode = self._determine_mode(market_data)

        print()
        print("=" * 80)
        if mode == "FULL":
            print("[GREEN] Running in FULL MODE (Exchange + Fear & Greed + News)")
        elif mode == "PARTIAL":
            print("[YELLOW] Running in PARTIAL MODE (Fear & Greed + News only)")
        else:
            print("[ORANGE] Running in NEWS ONLY MODE")
        print("=" * 80)

        if mode == "FULL":
            print(f"OK Exchange: {market_data.exchange_name} connected")
            print(f"OK Fear & Greed Index: Available")
            print(f"OK Technical Indicators: RSI, SMA available")
            print(f"OK News Sources: RSS feeds active")
        elif mode == "PARTIAL":
            print(f"OK Fear & Greed Index: Available")
            print(f"OK News Sources: RSS feeds active")
            print()
            print("WARNINGS:")
            if market_data.exchange_error:
                print(f"  X Exchange: {market_data.exchange_error}")
                self._print_exchange_suggestions(market_data.exchange_error)
        else:  # NEWS_ONLY
            print(f"OK News Sources: RSS feeds active")
            print()
            print("WARNINGS:")
            if market_data.exchange_error:
                print(f"  X Exchange: {market_data.exchange_error}")
                self._print_exchange_suggestions(market_data.exchange_error)
            if market_data.fear_greed_error:
                print(f"  X Fear & Greed: {market_data.fear_greed_error}")
                self._print_fear_greed_suggestions(market_data.fear_greed_error)

        print("=" * 80)
        print()

        if current_price == 0.0:
            print("�  Warning: Could not fetch BTC price, using demo mode")
            current_price = 50000.0  # Fallback price
            market_data.current_price = current_price

        print(f"Current BTC Price: ${current_price:,.2f}")
        if market_data.price_change_percentage_24h != 0:
            print(f"24h Change: {market_data.price_change_percentage_24h:+.2f}%")
        if market_data.fear_greed_index:
            print(f"Fear & Greed Index: {market_data.fear_greed_index} ({market_data.fear_greed_classification})")
        if market_data.rsi_14:
            print(f"RSI(14): {market_data.rsi_14}")
        if market_data.sma_5 or market_data.sma_10 or market_data.ema_9:
            indicators = []
            if market_data.sma_5:
                indicators.append(f"SMA(5): ${market_data.sma_5:,.2f}")
            if market_data.sma_10:
                indicators.append(f"SMA(10): ${market_data.sma_10:,.2f}")
            if market_data.ema_9:
                indicators.append(f"EMA(9): ${market_data.ema_9:,.2f}")
            print(f"Short-term MAs: {', '.join(indicators)}")

        # Display liquidity zone information if available
        if market_data.nearest_support or market_data.nearest_resistance:
            print()
            print("Liquidity Zones:")
            if market_data.nearest_support:
                print(f"  Support: ${market_data.nearest_support:,.2f} ({market_data.distance_to_support_pct:.2f}% below)")
            if market_data.nearest_resistance:
                print(f"  Resistance: ${market_data.nearest_resistance:,.2f} ({market_data.distance_to_resistance_pct:.2f}% above)")
            if market_data.volume_ratio:
                volume_status = "SPIKE" if market_data.volume_spike else "Normal"
                print(f"  Volume: {market_data.volume_ratio:.2f}x 20-day avg ({volume_status})")
            if market_data.potential_liquidity_zone:
                print(f"  WARNING: AT LIQUIDITY ZONE")
        print()

        # Detect reversal conditions (after technical indicators are calculated)
        print("Detecting reversal conditions...")
        reversal_signal = self.trading_advisor.detect_reversal_conditions(market_data)
        if reversal_signal["type"] != "none":
            print(f" WARNING: REVERSAL SIGNAL DETECTED!")
            if reversal_signal["type"] == "oversold_bounce":
                print(f"  - Type: OVERSOLD BOUNCE (Potential Bottom)")
            elif reversal_signal["type"] == "overbought_reversal":
                print(f"  - Type: OVERBOUGHT REVERSAL (Potential Top)")
            print(f"  - Strength: {reversal_signal['strength'].upper()}")
        else:
            print(f" No reversal conditions detected")
        print()

        # Detect bounce patterns (HIGHEST PRIORITY)
        print("Detecting bounce patterns (Wyckoff + MA + Volume)...")
        bounce_pattern = self.trading_advisor.detect_bounce_patterns(market_data)
        if bounce_pattern["pattern"] != "none":
            print(f" 🚨 BOUNCE PATTERN DETECTED (HIGHEST PRIORITY)!")
            print(f"  - Pattern: {bounce_pattern['pattern'].upper().replace('_', ' ')}")
            print(f"  - Signal: {bounce_pattern['signal']} ({bounce_pattern['strength'].upper()})")
            print(f"  - Timeframe: {bounce_pattern['wyckoff_timeframe'].upper().replace('_', ' ')}")
            print(f"  - Reasoning: {bounce_pattern['reasoning']}")
            print(f"  - ⚠️ This signal OVERRIDES conflicting indicators!")
        else:
            print(f" No bounce pattern detected")
        print()

        # Fetch news articles
        print(f"Fetching Bitcoin news articles (max {max_articles})...")
        articles = self.news_aggregator.fetch_news(max_articles=max_articles)
        print(f" Fetched {len(articles)} articles")
        print()

        if articles:
            print("Recent Headlines:")
            for i, article in enumerate(articles, 1):
                print(f"  {i}. {article.title[:80]}{'...' if len(article.title) > 80 else ''}")
                # Show publication time and source
                time_str = format_relative_time(getattr(article, '_timestamp', None))
                print(f"     Published {time_str} | Source: {article.source}")
            print()

        # Analyze sentiment (including reversal signal)
        print("Analyzing sentiment with AI...")
        sentiment = self.sentiment_analyzer.analyze_articles(articles, market_data, reversal_signal)
        print(f" Sentiment Analysis Complete")
        print(f"  - Sentiment: {sentiment.sentiment.upper()}")
        print(f"  - Confidence: {sentiment.confidence:.1f}%")
        print()

        # Generate trading recommendation
        print("Generating trading recommendation...")
        recommendation = self.trading_advisor.generate_recommendation(
            sentiment, current_price, portfolio_value, risk_tolerance, market_data
        )
        print(f" Recommendation Generated")
        print(f"  - Action: {recommendation.action.upper()}")
        print(f"  - Confidence: {recommendation.confidence:.1f}%")
        print()

        # Validate recommendation consistency (check for hallucinations)
        consistency_validation = self.trading_advisor.validate_recommendation_consistency(
            recommendation=recommendation,
            bounce_pattern=bounce_pattern,
            wyckoff_patterns=market_data.wyckoff_patterns,
            reversal_signal=reversal_signal,
            market_data=market_data
        )

        # If validation found issues, determine smart override
        if not consistency_validation['is_valid'] and len(consistency_validation['warnings']) > 0:
            print(f"⚠️  Recommendation validation warnings:")
            for warning in consistency_validation['warnings']:
                print(f"    - {warning}")

            # Check if hallucination is critical
            critical_hallucination = any('hallucination' in w.lower() for w in consistency_validation['warnings'])

            if critical_hallucination:
                print(f"🚫 CRITICAL HALLUCINATION DETECTED - Using pattern-based override")

                # Store original recommendation
                original_action = recommendation.action

                # SMART OVERRIDE: Use actual detected patterns in priority order
                override_action = None
                override_reasoning = ""
                override_confidence = 65.0  # Medium confidence for overrides

                # PRIORITY 1: Bounce Pattern (Highest)
                if bounce_pattern.get('pattern') != 'none':
                    override_action = bounce_pattern['signal'].lower()
                    override_reasoning = f"Override based on {bounce_pattern['pattern']}: {bounce_pattern['reasoning']}"
                    override_confidence = 75.0 if bounce_pattern['strength'] == 'strong' else 65.0
                    print(f"  → Using BOUNCE PATTERN: {override_action.upper()}")

                # PRIORITY 2: Reversal Signal
                elif reversal_signal.get('type') != 'none':
                    if 'bullish' in reversal_signal['type'] or 'oversold' in reversal_signal['type']:
                        override_action = 'buy'
                        override_reasoning = f"Override based on reversal: {reversal_signal['type']}"
                        override_confidence = 70.0
                        print(f"  → Using REVERSAL SIGNAL: BUY")
                    elif 'bearish' in reversal_signal['type'] or 'overbought' in reversal_signal['type']:
                        override_action = 'sell'
                        override_reasoning = f"Override based on reversal: {reversal_signal['type']}"
                        override_confidence = 70.0
                        print(f"  → Using REVERSAL SIGNAL: SELL")

                # PRIORITY 3: Short-term Wyckoff
                elif market_data.wyckoff_patterns.get('short_term'):
                    pattern = market_data.wyckoff_patterns['short_term']
                    if pattern['pattern'] in ['accumulation', 'bear_trap']:
                        override_action = 'buy'
                        override_reasoning = f"Override based on short-term Wyckoff {pattern['pattern']}"
                        override_confidence = 70.0
                        print(f"  → Using SHORT-TERM WYCKOFF: BUY")
                    elif pattern['pattern'] in ['distribution', 'bull_trap']:
                        override_action = 'sell'
                        override_reasoning = f"Override based on short-term Wyckoff {pattern['pattern']}"
                        override_confidence = 70.0
                        print(f"  → Using SHORT-TERM WYCKOFF: SELL")

                # PRIORITY 4: Medium-term Wyckoff
                elif market_data.wyckoff_patterns.get('medium_term'):
                    pattern = market_data.wyckoff_patterns['medium_term']
                    if pattern['pattern'] in ['accumulation', 'bear_trap']:
                        override_action = 'buy'
                        override_reasoning = f"Override based on medium-term Wyckoff {pattern['pattern']}"
                        override_confidence = 65.0
                        print(f"  → Using MEDIUM-TERM WYCKOFF: BUY")
                    elif pattern['pattern'] in ['distribution', 'bull_trap']:
                        override_action = 'sell'
                        override_reasoning = f"Override based on medium-term Wyckoff {pattern['pattern']}"
                        override_confidence = 65.0
                        print(f"  → Using MEDIUM-TERM WYCKOFF: SELL")

                # DEFAULT: HOLD if no patterns detected
                else:
                    override_action = 'hold'
                    override_reasoning = "No clear patterns detected after hallucination - defaulting to HOLD"
                    override_confidence = 0.0
                    print(f"  → No patterns available: HOLD")

                # Apply override
                recommendation = TradingRecommendation(
                    action=override_action,
                    confidence=override_confidence,
                    entry_price=current_price if override_action != 'hold' else None,
                    stop_loss=None,  # Will be calculated by validate_and_fix_recommendation
                    take_profit=None,  # Will be calculated by validate_and_fix_recommendation
                    position_size_percentage=5.0 if override_action != 'hold' else 0.0,
                    reasoning=f"OVERRIDDEN: AI hallucinated. Original: {original_action}. {override_reasoning}. Warnings: {consistency_validation['warnings']}"
                )

                consistency_validation['overridden'] = True
                consistency_validation['original_action'] = original_action
                consistency_validation['override_action'] = override_action
            print()

        # Fetch ATR hourly data early for validation (before any other processing)
        atr_hourly_value = None
        ohlcv_hourly = None
        if recommendation.action.lower() in ["buy", "sell"] and market_data.exchange_available:
            ohlcv_hourly = self.market_data_fetcher.get_btc_ohlcv(timeframe='1h', limit=100)
            if ohlcv_hourly:
                atr_hourly_value = self.market_data_fetcher.calculate_atr(
                    ohlcv_hourly, period=14, timeframe='1h'
                )

        # VALIDATE AND FIX RECOMMENDATION (before saving or further processing)
        print("Validating and fixing AI recommendation...")
        recommendation, validation_info = self.trading_advisor.validate_and_fix_recommendation(
            recommendation, market_data, atr_hourly_value
        )

        if validation_info.get("validation_applied"):
            if validation_info.get("fixes_applied"):
                print(f"  Validation fixes applied: {len(validation_info['fixes_applied'])}")
                for fix in validation_info["fixes_applied"]:
                    print(f"    - {fix}")
                if validation_info.get("target_overridden"):
                    print(f"    - AI original target: ${validation_info['ai_original_target']:,.2f}")
                    print(f"    - New validated target: ${recommendation.take_profit:,.2f}")
            else:
                print(f"  Validation passed - no fixes needed")
        print()

        # DISTRIBUTION DOMINANCE OVERRIDE
        # When bounce BUY blocked but 2+ timeframes show distribution → SELL
        if bounce_pattern.get('pattern') != 'none' and bounce_pattern.get('signal', '').lower() == 'buy' and recommendation.action == 'hold':
            # Count distribution vs accumulation across timeframes
            distribution_count = 0
            accumulation_count = 0

            for timeframe in ['short_term', 'medium_term', 'long_term']:
                pattern = market_data.wyckoff_patterns.get(timeframe)
                if pattern:
                    if pattern.get('pattern') in ['distribution', 'bull_trap']:
                        distribution_count += 1
                    elif pattern.get('pattern') in ['accumulation', 'bear_trap']:
                        accumulation_count += 1

            # If 2+ timeframes bearish and distribution dominates
            if distribution_count >= 2 and accumulation_count < distribution_count:
                # RSI SAFETY CHECK - Don't SELL into oversold conditions
                rsi = market_data.rsi_14 if hasattr(market_data, 'rsi_14') and market_data.rsi_14 else 50
                fear_greed = market_data.fear_greed_index if hasattr(market_data, 'fear_greed_index') and market_data.fear_greed_index else 50

                # Block SELL if oversold or extreme fear + low RSI
                if rsi < 35:
                    print(f"⚠️  Distribution dominance detected but RSI {rsi:.1f} too oversold - SKIPPING")
                    print(f"→ High bounce risk - keeping HOLD")
                elif rsi < 40 and fear_greed < 25:
                    print(f"⚠️  Distribution dominance detected but RSI {rsi:.1f} + F&G {fear_greed} (extreme oversold + fear) - SKIPPING")
                    print(f"→ Extreme fear with low RSI - keeping HOLD")
                else:
                    print(f"🎯 DISTRIBUTION DOMINANCE: {distribution_count} timeframes bearish")
                    print(f"→ Overriding HOLD to SELL")

                    # Calculate momentum target (2x ATR or 1.5% drop)
                    atr = market_data.atr if hasattr(market_data, 'atr') else None
                    if atr and atr > 0:
                        atr_target = current_price - (atr * 2)
                    else:
                        atr_target = current_price * 0.985

                    percent_target = current_price * 0.985  # 1.5% drop
                    target_price = min(atr_target, percent_target)

                    # Create SELL recommendation
                    recommendation = TradingRecommendation(
                        action='sell',
                        confidence=65.0,
                        entry_price=current_price,
                        stop_loss=market_data.nearest_resistance,
                        take_profit=target_price,
                        position_size_percentage=5.0,
                        reasoning=f"Distribution dominance override: {distribution_count} timeframes show distribution. Using momentum target (2x ATR or 1.5%)."
                    )

                    print(f"- Distribution SELL target: ${target_price:,.2f}")
                print()

        # SAFETY CHECK: Force HOLD if all exchanges failed
        blocked_recommendation = None
        if not market_data.exchange_available:
            print()
            print("=" * 80)
            print("CRITICAL SAFETY OVERRIDE")
            print("=" * 80)
            print("All exchange APIs unavailable - cannot generate safe trading recommendation")
            print()
            print(f"Exchanges tried: {', '.join(self.market_data_fetcher.exchanges_tried)}")
            print()
            print("Original recommendation overridden to HOLD for safety")
            print("=" * 80)
            print()

            # Save original recommendation
            blocked_recommendation = {
                "action": recommendation.action,
                "confidence": recommendation.confidence,
                "entry_price": recommendation.entry_price,
                "stop_loss": recommendation.stop_loss,
                "take_profit": recommendation.take_profit,
                "reasoning": recommendation.reasoning
            }

            # Override to HOLD
            recommendation = TradingRecommendation(
                action="hold",
                confidence=0.0,
                entry_price=None,
                stop_loss=None,
                take_profit=None,
                position_size_percentage=0.0,
                reasoning="All exchange APIs unavailable - cannot generate safe trading recommendation with real-time price data"
            )

        # Calculate hybrid stop-loss with ATR and leverage optimization
        print("Calculating hybrid stop-loss and leverage optimization...")
        hybrid_stop = None
        if recommendation.action.lower() in ["buy", "sell"] and market_data.exchange_available:
            # Reuse OHLCV data fetched earlier for validation (already fetched at line ~1868)
            # If not available, fetch it now
            if not ohlcv_hourly:
                ohlcv_hourly = self.market_data_fetcher.get_btc_ohlcv(timeframe='1h', limit=100)
            if ohlcv_hourly:
                # Check if this is a bounce pattern trade
                is_bounce_pattern = bounce_pattern.get('pattern') != 'none'

                hybrid_stop = self.market_data_fetcher.calculate_hybrid_stop_loss(
                    ohlcv_data=ohlcv_hourly,
                    entry_price=recommendation.entry_price or current_price,
                    target_price=recommendation.take_profit,
                    action=recommendation.action,
                    portfolio_value=portfolio_value,
                    timeframe='1h',
                    is_bounce_pattern=is_bounce_pattern,
                    market_data=market_data
                )
                if hybrid_stop and hybrid_stop.get("atr_value"):
                    print(f"  Hybrid Stop-Loss Calculated")
                    timeframe_label = hybrid_stop.get('atr_timeframe', '1h')
                    print(f"  - ATR(14, {timeframe_label}): ${hybrid_stop['atr_value']:,.2f}")
                    print(f"  - ATR Stop Loss: ${hybrid_stop['atr_stop_loss']:,.2f}")
                    if hybrid_stop.get('risk_reward_ratio'):
                        rr_status = "GOOD" if hybrid_stop['meets_rr_minimum'] else "LOW"
                        print(f"  - Risk/Reward: 1:{hybrid_stop['risk_reward_ratio']} ({rr_status})")
                    print(f"  - Recommended Leverage: {hybrid_stop['recommended_leverage']}x")
                    print(f"  - Max Safe Position: {hybrid_stop['max_safe_position_pct']:.1f}%")

                    # Safety filter: Block trades with poor R/R ratio
                    if hybrid_stop.get('risk_reward_ratio') and not hybrid_stop['meets_rr_minimum']:
                        # Determine minimum R/R based on trade type
                        min_rr = 1.0 if is_bounce_pattern else 2.0
                        min_rr_label = "1:1" if is_bounce_pattern else "1:2"
                        trade_type = "bounce pattern" if is_bounce_pattern else "regular"

                        print()
                        print(f"  WARNING: TRADE BLOCKED: Risk/Reward ratio too low ({hybrid_stop['risk_reward_ratio']}:1 < {min_rr_label} minimum for {trade_type} trades)")

                        # Save original recommendation (if not already blocked by exchange failure)
                        if not blocked_recommendation:
                            blocked_recommendation = {
                                "action": recommendation.action,
                                "confidence": recommendation.confidence,
                                "entry_price": recommendation.entry_price,
                                "stop_loss": recommendation.stop_loss,
                                "take_profit": recommendation.take_profit,
                                "reasoning": recommendation.reasoning
                            }

                        # Calculate required target for minimum R/R
                        entry = recommendation.entry_price or current_price
                        atr_stop = hybrid_stop['atr_stop_loss']
                        risk_amount = abs(entry - atr_stop)
                        required_profit = risk_amount * min_rr  # Use appropriate minimum

                        if recommendation.action.lower() == "buy":
                            required_target = entry + required_profit
                        else:  # sell
                            required_target = entry - required_profit

                        # Override recommendation to HOLD
                        new_reasoning = (
                            f"Trade blocked - Risk/Reward ratio of {hybrid_stop['risk_reward_ratio']}:1 is below the {min_rr_label} minimum threshold for {trade_type} trades. "
                            f"The ATR-based stop loss at ${atr_stop:,.2f} would require a target of ${required_target:,.2f} to meet minimum R/R. "
                            f"Current target of ${recommendation.take_profit:,.2f} is insufficient. Waiting for better setup with improved risk/reward."
                        )

                        # Create new recommendation object with HOLD action
                        recommendation = TradingRecommendation(
                            action="hold",
                            confidence=recommendation.confidence * 0.5,  # Reduce confidence
                            entry_price=None,
                            stop_loss=None,
                            take_profit=None,
                            position_size_percentage=0.0,
                            reasoning=new_reasoning
                        )

                        print(f"  - Required target for {min_rr_label} R/R: ${required_target:,.2f}")
                        print(f"  - Recommendation changed to: HOLD")
        print()

        # Calculate position size
        position_details = self.trading_advisor.calculate_position_size(
            recommendation, portfolio_value, current_price
        )

        # Convert numeric confidence to string for evaluation framework
        def get_confidence_level(confidence: float) -> str:
            if confidence >= 80:
                return "HIGH"
            elif confidence >= 50:
                return "MEDIUM"
            else:
                return "LOW"

        # Prepare results in evaluation framework format
        evaluation_format = {
            "timestamp": datetime.now().isoformat(),
            "recommendation": recommendation.action.upper(),  # Ensure uppercase for evaluation framework
            "confidence_level": get_confidence_level(recommendation.confidence),
            "sentiment_analysis": {
                "sentiment": sentiment.sentiment,
                "confidence": sentiment.confidence,  # This should be numeric for evaluation framework
                "key_points": sentiment.key_points,
                "overall_score": sentiment.confidence / 100.0  # Normalize to 0-1 scale
            },
            "position_sizing": {
                "recommended_size": position_details["percentage_of_portfolio"]
            },
            "risk_management": {
                "stop_loss": recommendation.stop_loss,
                "take_profit": recommendation.take_profit
            }
        }

        # Also save the full results for display
        full_results = {
            "timestamp": datetime.now().isoformat(),
            "analysis_mode": mode,
            "current_price": current_price,
            "portfolio_value": portfolio_value,
            "risk_tolerance": risk_tolerance,
            "articles_analyzed": len(articles),
            "data_sources": {
                "exchange_available": market_data.exchange_available,
                "exchange_name": market_data.exchange_name,
                "exchange_error": market_data.exchange_error,
                "fear_greed_available": market_data.fear_greed_available,
                "fear_greed_error": market_data.fear_greed_error,
                "news_sources_count": 3  # RSS feeds
            },
            "market_data": {
                "price": market_data.current_price,
                "volume_24h": market_data.volume_24h,
                "price_change_24h": market_data.price_change_percentage_24h,
                "fear_greed_index": market_data.fear_greed_index,
                "fear_greed_classification": market_data.fear_greed_classification,
                "rsi_14": market_data.rsi_14,
                "sma_5": market_data.sma_5,
                "sma_10": market_data.sma_10,
                "ema_9": market_data.ema_9,
                "sma_20": market_data.sma_20,
                "sma_50": market_data.sma_50,
                "price_vs_sma20": market_data.price_vs_sma20,
                "price_vs_sma50": market_data.price_vs_sma50,
                "nearest_support": market_data.nearest_support,
                "nearest_resistance": market_data.nearest_resistance,
                "distance_to_support_pct": market_data.distance_to_support_pct,
                "distance_to_resistance_pct": market_data.distance_to_resistance_pct,
                "volume_ratio": market_data.volume_ratio,
                "volume_spike": market_data.volume_spike,
                "potential_liquidity_zone": market_data.potential_liquidity_zone,
                "wyckoff_patterns": market_data.wyckoff_patterns
            },
            "reversal_signal": reversal_signal,
            "bounce_pattern": bounce_pattern,
            "sentiment": {
                "sentiment": sentiment.sentiment,
                "confidence": sentiment.confidence,
                "key_points": sentiment.key_points,
                "reasoning": sentiment.reasoning
            },
            "recommendation": {
                "action": recommendation.action,
                "confidence": recommendation.confidence,
                "entry_price": recommendation.entry_price,
                "stop_loss": recommendation.stop_loss,
                "take_profit": recommendation.take_profit,
                "reasoning": recommendation.reasoning
            },
            "position_sizing": position_details,
            "hybrid_stop_loss": hybrid_stop if hybrid_stop else None,
            "blocked_recommendation": blocked_recommendation if blocked_recommendation else None,
            "validation_info": validation_info if validation_info else None,
            "consistency_validation": consistency_validation
        }

        # Display results
        self._display_results(full_results)

        # Save both formats
        output_dir = "data/analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full results
        output_file = os.path.join(output_dir, f"btc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w') as f:
            json.dump(full_results, f, indent=2)
        print(f" Full results saved to {output_file}")

        # Save evaluation format
        evaluation_file = os.path.join(output_dir, f"evaluation_format_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(evaluation_file, 'w') as f:
            json.dump(evaluation_format, f, indent=2)
        print(f" Evaluation format saved to {evaluation_file}")

        return evaluation_format

    def _print_wyckoff_pattern(self, pattern: Dict, indent: str = "  ") -> None:
        """Print a Wyckoff pattern with the specified indentation"""
        if not pattern:
            return

        if pattern["pattern"] == "distribution":
            print(f"{indent}Pattern: DISTRIBUTION (BEARISH)")
            print(f"{indent}Resistance: ${pattern['resistance_level']:,.2f} tested {pattern['test_count']}x")
            if pattern.get('volume_declining'):
                print(f"{indent}Volume: Declining (classic distribution)")
            print(f"{indent}Signal: Smart money selling - BEARISH")
        elif pattern["pattern"] == "accumulation":
            print(f"{indent}Pattern: ACCUMULATION (BULLISH)")
            print(f"{indent}Support: ${pattern['support_level']:,.2f} tested {pattern['test_count']}x")
            if pattern.get('volume_declining'):
                print(f"{indent}Volume: Declining (classic accumulation)")
            print(f"{indent}Signal: Smart money buying - BULLISH")
        elif pattern["pattern"] == "failed_breakout":
            trap_type = pattern.get('type', '').upper()
            print(f"{indent}Pattern: {trap_type}")
            print(f"{indent}Trap Level: ${pattern.get('trap_level', 0):,.2f}")
            if pattern.get('type') == "bull_trap":
                print(f"{indent}Signal: Failed breakout - BEARISH")
            else:
                print(f"{indent}Signal: Failed breakdown - BULLISH")

    def _display_results(self, results: Dict):
        """Display analysis results in a formatted way"""
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
        print()

        # Display reversal signal if present
        if "reversal_signal" in results and results["reversal_signal"]["type"] != "none":
            print("WARNING: REVERSAL SIGNAL")
            print("-" * 80)
            reversal = results["reversal_signal"]
            if reversal["type"] == "oversold_bounce":
                print("Type: OVERSOLD BOUNCE (Potential Bottom)")
                print("Description: Extreme fear + oversold RSI often precede price bounces")
            elif reversal["type"] == "overbought_reversal":
                print("Type: OVERBOUGHT REVERSAL (Potential Top)")
                print("Description: Extreme greed + overbought RSI often precede corrections")
            print(f"Strength: {reversal['strength'].upper()}")
            print()

        # Display bounce pattern if present (HIGHEST PRIORITY)
        if "bounce_pattern" in results and results["bounce_pattern"]["pattern"] != "none":
            print("🚨 BOUNCE PATTERN DETECTED (HIGHEST PRIORITY)")
            print("=" * 80)
            bounce = results["bounce_pattern"]
            print(f"Pattern: {bounce['pattern'].upper().replace('_', ' ')}")
            print(f"Signal: {bounce['signal']} ({bounce['strength'].upper()})")
            print(f"Timeframe: {bounce['wyckoff_timeframe'].upper().replace('_', ' ')}")
            print(f"Reasoning: {bounce['reasoning']}")
            print()
            print("⚠️  THIS SIGNAL OVERRIDES ALL CONFLICTING INDICATORS")
            print("    Bounce patterns combine Wyckoff smart money activity,")
            print("    price position vs moving averages, and volume confirmation.")
            print("    These are high-probability setups that should be strongly weighted.")
            print()

        # Display market data first (only if we have valid data)
        if "market_data" in results:
            mkt = results["market_data"]
            # Only show section if we have at least price or some indicators
            has_data = (mkt['price'] > 0 or mkt['fear_greed_index'] or
                       mkt['rsi_14'] or mkt['sma_20'] or mkt['sma_50'])

            if has_data:
                print("MARKET DATA & TECHNICAL INDICATORS")
                print("-" * 80)

                if mkt['price'] > 0:
                    print(f"Price: ${mkt['price']:,.2f}")
                    if mkt['price_change_24h'] != 0:
                        print(f"24h Change: {mkt['price_change_24h']:+.2f}%")
                    if mkt['volume_24h'] > 0:
                        print(f"24h Volume: {mkt['volume_24h']:,.0f} BTC")

                if mkt.get('fear_greed_index'):
                    print(f"Fear & Greed Index: {mkt['fear_greed_index']} - {mkt['fear_greed_classification']}")

                if mkt.get('rsi_14'):
                    rsi_signal = "Overbought" if mkt['rsi_14'] > 70 else ("Oversold" if mkt['rsi_14'] < 30 else "Neutral")
                    print(f"RSI(14): {mkt['rsi_14']} ({rsi_signal})")

                # Short-term moving averages
                if mkt.get('sma_5'):
                    print(f"SMA(5): ${mkt['sma_5']:,.2f}")
                if mkt.get('sma_10'):
                    print(f"SMA(10): ${mkt['sma_10']:,.2f}")
                if mkt.get('ema_9'):
                    print(f"EMA(9): ${mkt['ema_9']:,.2f}")

                # Medium/Long-term moving averages
                if mkt.get('sma_20'):
                    print(f"SMA(20): ${mkt['sma_20']:,.2f} (Price is {mkt['price_vs_sma20']})")

                if mkt.get('sma_50'):
                    print(f"SMA(50): ${mkt['sma_50']:,.2f} (Price is {mkt['price_vs_sma50']})")

                # Liquidity zones section
                if mkt.get('nearest_support') or mkt.get('nearest_resistance'):
                    print()
                    print("Liquidity Zones & Market Structure:")
                    if mkt.get('nearest_support'):
                        print(f"  Support: ${mkt['nearest_support']:,.2f} ({mkt['distance_to_support_pct']:.2f}% below)")
                    if mkt.get('nearest_resistance'):
                        print(f"  Resistance: ${mkt['nearest_resistance']:,.2f} ({mkt['distance_to_resistance_pct']:.2f}% above)")
                    if mkt.get('volume_ratio'):
                        volume_status = "SPIKE" if mkt.get('volume_spike') else "Normal"
                        print(f"  Volume vs 20-day Avg: {mkt['volume_ratio']:.2f}x ({volume_status})")
                    if mkt.get('potential_liquidity_zone'):
                        print(f"  WARNING: AT LIQUIDITY ZONE - High probability of price reaction")

                # Wyckoff pattern section (multi-timeframe)
                wyckoff_patterns = mkt.get('wyckoff_patterns')
                if wyckoff_patterns:
                    # Check if any patterns exist
                    has_patterns = any(wyckoff_patterns.get(tf) for tf in ["short_term", "medium_term", "long_term"])

                    if has_patterns:
                        print()
                        print("Wyckoff Patterns (Multi-Timeframe):")

                        # Short-term (highest priority)
                        if wyckoff_patterns.get('short_term'):
                            print(f"  🔴 SHORT-TERM (20 candles - HIGHEST PRIORITY for 12h):")
                            self._print_wyckoff_pattern(wyckoff_patterns['short_term'], indent="    ")

                        # Medium-term
                        if wyckoff_patterns.get('medium_term'):
                            print(f"  🟡 MEDIUM-TERM (50 candles - MEDIUM PRIORITY):")
                            self._print_wyckoff_pattern(wyckoff_patterns['medium_term'], indent="    ")

                        # Long-term
                        if wyckoff_patterns.get('long_term'):
                            print(f"  🟢 LONG-TERM (100 candles - LOW PRIORITY for 12h):")
                            self._print_wyckoff_pattern(wyckoff_patterns['long_term'], indent="    ")
                            # Note distance for long-term patterns
                            pattern = wyckoff_patterns['long_term']
                            current_price = mkt.get('current_price', 0)
                            if pattern.get('pattern') in ['distribution', 'accumulation'] and current_price > 0:
                                level_key = "resistance_level" if pattern['pattern'] == "distribution" else "support_level"
                                level = pattern.get(level_key)
                                if level:
                                    distance_pct = abs((level - current_price) / current_price * 100)
                                    if distance_pct > 10:
                                        print(f"    ⚠️ NOTE: {distance_pct:.1f}% from current price - LESS RELEVANT for 12h")

                print()

        print("SENTIMENT ANALYSIS")
        print("-" * 80)
        sentiment = results["sentiment"]
        print(f"Overall Sentiment: {sentiment['sentiment'].upper()} ({sentiment['confidence']:.1f}% confidence)")
        print()
        print("Key Points:")
        for point in sentiment['key_points']:
            print(f"  • {point}")
        print()
        print(f"Reasoning: {sentiment['reasoning']}")
        print()

        print("TRADING RECOMMENDATION")
        print("-" * 80)
        rec = results["recommendation"]
        print(f"Action: {rec['action'].upper()} ({rec['confidence']:.1f}% confidence)")
        if rec['entry_price']:
            print(f"Entry Price: ${rec['entry_price']:,.2f}")
        if rec['stop_loss']:
            print(f"Stop Loss: ${rec['stop_loss']:,.2f}")
        if rec['take_profit']:
            print(f"Take Profit: ${rec['take_profit']:,.2f}")
        print()
        print(f"Reasoning: {rec['reasoning']}")
        print()

        print("POSITION SIZING")
        print("-" * 80)
        pos = results["position_sizing"]
        print(f"Position Value: ${pos['position_value_usd']:,.2f} ({pos['percentage_of_portfolio']:.1f}% of portfolio)")
        print(f"BTC Amount: {pos['btc_amount']:.6f} BTC")
        if pos['max_loss_usd']:
            print(f"Maximum Loss (if stop loss hit): ${pos['max_loss_usd']:,.2f}")
        if pos['potential_profit_usd']:
            print(f"Potential Profit (if take profit hit): ${pos['potential_profit_usd']:,.2f}")
            if pos['max_loss_usd']:
                risk_reward = pos['potential_profit_usd'] / pos['max_loss_usd']
                print(f"Risk/Reward Ratio: 1:{risk_reward:.2f}")
        print()

        # Display hybrid stop-loss analysis if available
        if "hybrid_stop_loss" in results and results["hybrid_stop_loss"]:
            hybrid = results["hybrid_stop_loss"]
            if hybrid.get("atr_value"):
                print("HYBRID STOP-LOSS & LEVERAGE ANALYSIS")
                print("-" * 80)
                timeframe_label = hybrid.get('atr_timeframe', '1h')
                print(f"ATR(14, {timeframe_label}): ${hybrid['atr_value']:,.2f}")
                print(f"ATR-Based Stop Loss: ${hybrid['atr_stop_loss']:,.2f}")

                if hybrid.get('risk_reward_ratio'):
                    rr_status = "MEETS MINIMUM" if hybrid['meets_rr_minimum'] else "BELOW MINIMUM"
                    print(f"Risk/Reward Ratio: 1:{hybrid['risk_reward_ratio']} ({rr_status})")

                print(f"\nRecommended Leverage: {hybrid['recommended_leverage']}x")
                print(f"Max Safe Position Size: {hybrid['max_safe_position_pct']:.1f}%")

                # Display leverage analysis table
                if hybrid.get('leverage_analysis'):
                    print("\nLeverage Safety Analysis:")
                    print(f"  {'Leverage':<10} {'Liq Price':<15} {'Safe':<8} {'Max Position'}")
                    print("  " + "-" * 50)
                    for lev, analysis in hybrid['leverage_analysis'].items():
                        liq_str = f"${analysis['liq_price']:,.0f}" if analysis['liq_price'] else "None"
                        safe_str = "Yes" if analysis['safe'] else "No"
                        print(f"  {lev:<10} {liq_str:<15} {safe_str:<8} {analysis['max_position']:.1f}%")
                print()

    def _determine_mode(self, market_data) -> str:
        """Determine analysis mode based on available data"""
        if market_data.exchange_available and market_data.fear_greed_available:
            return "FULL"
        elif market_data.fear_greed_available:
            return "PARTIAL"
        else:
            return "NEWS_ONLY"

    def _print_exchange_suggestions(self, error_msg: str):
        """Print suggestions based on exchange error"""
        print()
        print("  Suggestions:")
        if "451" in error_msg or "Geo-restricted" in error_msg:
            print("    - Use a VPN to bypass geo-restrictions")
            print("    - Try a different exchange (modify MarketDataFetcher in code)")
            print("    - Use CoinGecko API as alternative (free, no restrictions)")
        elif "Timeout" in error_msg:
            print("    - Check your internet connection")
            print("    - Try again in a few moments")
            print("    - Increase timeout in code if connection is slow")
        elif "Connection" in error_msg:
            print("    - Verify internet connection is active")
            print("    - Check firewall settings")
        else:
            print("    - Check exchange status (may be down for maintenance)")
            print("    - Review error message above for specific issue")

    def _print_fear_greed_suggestions(self, error_msg: str):
        """Print suggestions based on Fear & Greed error"""
        print()
        print("  Suggestions:")
        if "Timeout" in error_msg:
            print("    - API is slow, try again later")
        elif "Connection" in error_msg:
            print("    - Check internet connection")
        else:
            print("    - Alternative.me API may be down")
            print("    - Analysis will continue with news data only")


def main():
    """Example usage of the Bitcoin Trading Bot"""

    # Configuration
    OLLAMA_MODEL = "qwen3-coder:30b"
    NEWSAPI_KEY = None  # Set your NewsAPI key here or leave as None to use free sources
    PORTFOLIO_VALUE = 10000.0  # Your portfolio value in USD
    RISK_TOLERANCE = "medium"  # low, medium, or high

    # Initialize the bot
    bot = BitcoinTradingBot(
        ollama_model=OLLAMA_MODEL,
        newsapi_key=NEWSAPI_KEY
    )

    # Run analysis
    try:
        results = bot.run_analysis(
            portfolio_value=PORTFOLIO_VALUE,
            risk_tolerance=RISK_TOLERANCE,
            max_articles=10
        )

        print("\nAnalysis complete!")
        print(f"Results saved in evaluation format for framework compatibility")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n\nL Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
