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
    sma_20: Optional[float]
    sma_50: Optional[float]
    price_vs_sma20: Optional[str]  # "above" or "below"
    price_vs_sma50: Optional[str]  # "above" or "below"
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
        # Use Coinbase as primary, with Kraken as fallback
        try:
            self.exchange = ccxt.coinbase()
            self.exchange_name = "Coinbase"
        except Exception:
            try:
                self.exchange = ccxt.kraken()
                self.exchange_name = "Kraken"
            except Exception:
                self.exchange = None
                self.exchange_name = "None"

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

    def get_btc_ohlcv(self, timeframe: str = '1h', limit: int = 100) -> List[List]:
        """Fetch OHLCV data from exchange"""
        if not self.exchange:
            return []

        try:
            # Try BTC/USD first (Coinbase), then BTC/USDT (others)
            try:
                ohlcv = self.exchange.fetch_ohlcv('BTC/USD', timeframe=timeframe, limit=limit)
            except Exception:
                ohlcv = self.exchange.fetch_ohlcv('BTC/USDT', timeframe=timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            print(f"Error fetching OHLCV data from {self.exchange_name}: {e}")
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

    def fetch_market_data(self) -> MarketData:
        """Fetch comprehensive market data including technical indicators"""
        exchange_error = None
        exchange_available = False

        if not self.exchange:
            exchange_error = "No exchange initialized - Both Coinbase and Kraken failed"
            return self._get_fallback_market_data(exchange_error)

        try:
            # Fetch ticker data - try BTC/USD first (Coinbase), then BTC/USDT
            try:
                ticker = self.exchange.fetch_ticker('BTC/USD')
            except Exception:
                ticker = self.exchange.fetch_ticker('BTC/USDT')

            current_price = ticker.get('last', 0) or ticker.get('close', 0)
            volume_24h = ticker.get('quoteVolume', 0) or ticker.get('baseVolume', 0) or 0
            price_change_24h = ticker.get('change', 0) or 0
            price_change_percentage_24h = ticker.get('percentage', 0) or 0

            # Fetch OHLCV for technical indicators
            ohlcv = self.get_btc_ohlcv(timeframe='1d', limit=100)

            rsi_14 = None
            sma_20 = None
            sma_50 = None
            price_vs_sma20 = None
            price_vs_sma50 = None

            if ohlcv and len(ohlcv) >= 50:
                closing_prices = [candle[4] for candle in ohlcv]  # Close price is index 4

                # Calculate RSI
                rsi_14 = self.calculate_rsi(closing_prices, period=14)

                # Calculate SMAs
                sma_20 = self.calculate_sma(closing_prices, period=20)
                sma_50 = self.calculate_sma(closing_prices, period=50)

                # Determine price position relative to SMAs
                if sma_20:
                    price_vs_sma20 = "above" if current_price > sma_20 else "below"
                if sma_50:
                    price_vs_sma50 = "above" if current_price > sma_50 else "below"

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
                sma_20=sma_20,
                sma_50=sma_50,
                price_vs_sma20=price_vs_sma20,
                price_vs_sma50=price_vs_sma50,
                exchange_available=True,
                exchange_name=self.exchange_name,
                exchange_error=None,
                fear_greed_available=fear_greed_value is not None,
                fear_greed_error=fear_greed_error
            )

        except requests.exceptions.HTTPError as e:
            if '451' in str(e) or 'Unavailable For Legal Reasons' in str(e):
                exchange_error = f"{self.exchange_name}: HTTP 451 - Geo-restricted (try VPN or different exchange)"
            else:
                exchange_error = f"{self.exchange_name}: HTTP {e.response.status_code} - {str(e)}"
        except requests.exceptions.Timeout:
            exchange_error = f"{self.exchange_name}: Timeout - API too slow (check connection)"
        except requests.exceptions.ConnectionError:
            exchange_error = f"{self.exchange_name}: Connection error (check internet)"
        except Exception as e:
            exchange_error = f"{self.exchange_name}: {type(e).__name__} - {str(e)}"

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
            sma_20=None,
            sma_50=None,
            price_vs_sma20=None,
            price_vs_sma50=None,
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

    def analyze_articles(self, articles: List[NewsArticle], market_data: Optional[MarketData] = None) -> SentimentAnalysis:
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
- 24h Volume: ${market_data.volume_24h:,.0f}
"""
            if market_data.fear_greed_index:
                market_context += f"- Fear & Greed Index: {market_data.fear_greed_index} ({market_data.fear_greed_classification})\n"
            if market_data.rsi_14:
                market_context += f"- RSI(14): {market_data.rsi_14}\n"
            if market_data.sma_20:
                market_context += f"- SMA(20): ${market_data.sma_20:,.2f} (Price is {market_data.price_vs_sma20})\n"
            if market_data.sma_50:
                market_context += f"- SMA(50): ${market_data.sma_50:,.2f} (Price is {market_data.price_vs_sma50})\n"

        # Create analysis prompt
        system_prompt = """You are an expert Bitcoin trading analyst. Analyze news articles and market data to provide sentiment analysis in JSON format.
Your response must be valid JSON with this structure:
{
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": <number 0-100>,
    "key_points": [<list of key findings>],
    "reasoning": "<detailed explanation>"
}

Consider both the news sentiment and technical indicators (RSI, moving averages, Fear & Greed Index) in your analysis."""

        prompt = f"""Analyze the following Bitcoin news articles and market data to determine the overall market sentiment:

{articles_text}
{market_context}

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

    def generate_recommendation(
        self,
        sentiment: SentimentAnalysis,
        current_price: float,
        portfolio_value: float,
        risk_tolerance: str = "medium",  # low, medium, high
        market_data: Optional[MarketData] = None
    ) -> TradingRecommendation:
        """Generate trading recommendation based on sentiment and market data"""

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

Use technical indicators (RSI, moving averages) and market sentiment (Fear & Greed) to inform your decision."""

        # Prepare market data context
        market_context = ""
        if market_data and market_data.current_price > 0:
            market_context = f"\nTechnical Indicators:\n"
            if market_data.rsi_14:
                market_context += f"- RSI(14): {market_data.rsi_14} (>70 overbought, <30 oversold)\n"
            if market_data.price_vs_sma20:
                market_context += f"- Price vs SMA(20): {market_data.price_vs_sma20}\n"
            if market_data.price_vs_sma50:
                market_context += f"- Price vs SMA(50): {market_data.price_vs_sma50}\n"
            if market_data.fear_greed_index:
                market_context += f"- Fear & Greed Index: {market_data.fear_greed_index} ({market_data.fear_greed_classification})\n"
            if market_data.price_change_percentage_24h != 0:
                market_context += f"- 24h Price Change: {market_data.price_change_percentage_24h:+.2f}%\n"

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

Provide a trading recommendation with proper risk management (stop loss, take profit, position sizing).
Consider the risk tolerance when determining position size:
- Low risk: 1-3% of portfolio
- Medium risk: 3-7% of portfolio
- High risk: 7-15% of portfolio

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

        # Analyze sentiment
        print("Analyzing sentiment with AI...")
        sentiment = self.sentiment_analyzer.analyze_articles(articles, market_data)
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

        # Calculate position size
        position_details = self.trading_advisor.calculate_position_size(
            recommendation, portfolio_value, current_price
        )

        # Prepare results
        results = {
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
                "sma_20": market_data.sma_20,
                "sma_50": market_data.sma_50,
                "price_vs_sma20": market_data.price_vs_sma20,
                "price_vs_sma50": market_data.price_vs_sma50
            },
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
            "position_sizing": position_details
        }

        # Display results
        self._display_results(results)

        return results

    def _display_results(self, results: Dict):
        """Display analysis results in a formatted way"""
        print("=" * 80)
        print("ANALYSIS RESULTS")
        print("=" * 80)
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
                        print(f"24h Volume: ${mkt['volume_24h']:,.0f}")

                if mkt.get('fear_greed_index'):
                    print(f"Fear & Greed Index: {mkt['fear_greed_index']} - {mkt['fear_greed_classification']}")

                if mkt.get('rsi_14'):
                    rsi_signal = "Overbought" if mkt['rsi_14'] > 70 else ("Oversold" if mkt['rsi_14'] < 30 else "Neutral")
                    print(f"RSI(14): {mkt['rsi_14']} ({rsi_signal})")

                if mkt.get('sma_20'):
                    print(f"SMA(20): ${mkt['sma_20']:,.2f} (Price is {mkt['price_vs_sma20']})")

                if mkt.get('sma_50'):
                    print(f"SMA(50): ${mkt['sma_50']:,.2f} (Price is {mkt['price_vs_sma50']})")

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

        print("=� POSITION SIZING")
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

        # Optionally save results to file
        output_dir = "data/analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"btc_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f" Results saved to {output_file}")

    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
    except Exception as e:
        print(f"\n\nL Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
