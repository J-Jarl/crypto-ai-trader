"""
Data Cache - Persistent storage for historical market data

Caches:
- OHLCV candles (by symbol, timeframe, date)
- Fear & Greed Index (by date)

Benefits:
- Consistent backtest results
- Faster execution (no repeated API calls)
- Immutable historical data
"""

import pandas as pd
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import ccxt


class DataCache:
    """Persistent cache for market data"""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize data cache

        Args:
            cache_dir: Root directory for cached data
        """
        self.cache_dir = Path(cache_dir)
        self.ohlcv_dir = self.cache_dir / "ohlcv"
        self.fear_greed_dir = self.cache_dir / "fear_greed"

        # Create directories
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        self.fear_greed_dir.mkdir(parents=True, exist_ok=True)

        # Initialize exchange
        self.exchange = ccxt.coinbase({'enableRateLimit': True})

    def get_ohlcv_cache_path(self, symbol: str, timeframe: str, date: datetime) -> Path:
        """Get cache file path for OHLCV data"""
        # Format: BTC_USDT_1h_2025-12-02.csv
        symbol_clean = symbol.replace('/', '_')
        date_str = date.strftime('%Y-%m-%d')
        filename = f"{symbol_clean}_{timeframe}_{date_str}.csv"
        return self.ohlcv_dir / filename

    def get_fear_greed_cache_path(self, date: datetime) -> Path:
        """Get cache file path for Fear & Greed data"""
        # Format: 2025-12-02.json
        date_str = date.strftime('%Y-%m-%d')
        filename = f"{date_str}.json"
        return self.fear_greed_dir / filename

    def fetch_ohlcv(self, symbol: str, timeframe: str, since: int, limit: int) -> List[List]:
        """
        Fetch OHLCV data with smart caching (permanent for old, TTL for recent)

        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Candle timeframe (e.g., '1h')
            since: Start timestamp in milliseconds
            limit: Number of candles

        Returns:
            List of OHLCV candles
        """
        # Determine date range
        start_date = datetime.fromtimestamp(since / 1000)
        days_ago = (datetime.now() - start_date).days

        # Determine cache strategy
        is_finalized = days_ago >= 7
        ttl_hours = 2  # TTL for recent data

        cache_path = self.get_ohlcv_cache_path(symbol, timeframe, start_date)
        meta_path = cache_path.with_suffix('.meta.json')

        # Check cache
        if cache_path.exists():
            # Check if cache is valid
            cache_valid = False

            if is_finalized:
                # Old data - cache is always valid (immutable)
                cache_valid = True
                print(f"📦 Using permanent cache: {cache_path.name}")
            elif meta_path.exists():
                # Recent data - check TTL
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    cached_at = datetime.fromisoformat(meta['cached_at'])
                    age_hours = (datetime.now() - cached_at).total_seconds() / 3600

                    if age_hours < ttl_hours:
                        cache_valid = True
                        print(f"📦 Using TTL cache ({age_hours:.1f}h old): {cache_path.name}")
                    else:
                        print(f"⏰ Cache expired ({age_hours:.1f}h > {ttl_hours}h TTL): {cache_path.name}")

            if cache_valid:
                df = pd.read_csv(cache_path)
                return df.values.tolist()

        # Fetch from API
        print(f"🌐 Fetching from API: {symbol} {timeframe} (limit={limit})")
        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)

        # Save to cache
        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df.to_csv(cache_path, index=False)

            # Save metadata for recent data
            if not is_finalized:
                with open(meta_path, 'w') as f:
                    json.dump({
                        'cached_at': datetime.now().isoformat(),
                        'ttl_hours': ttl_hours,
                        'finalized': False
                    }, f)
                print(f"💾 Cached with {ttl_hours}h TTL: {cache_path.name}")
            else:
                print(f"💾 Cached permanently: {cache_path.name}")

        return ohlcv

    def fetch_fear_greed(self, date: datetime) -> Optional[int]:
        """
        Fetch Fear & Greed Index with smart caching (permanent for old, TTL for recent)

        Args:
            date: Date to fetch index for

        Returns:
            Fear & Greed value (0-100) or None
        """
        # Determine cache strategy
        days_ago = (datetime.now() - date).days
        is_finalized = days_ago >= 7
        ttl_hours = 2

        cache_path = self.get_fear_greed_cache_path(date)

        # Check cache
        if cache_path.exists():
            with open(cache_path, 'r') as f:
                data = json.load(f)

            cache_valid = False

            if is_finalized:
                cache_valid = True
                print(f"📦 Using permanent cache: {cache_path.name}")
            elif 'cached_at' in data:
                cached_at = datetime.fromisoformat(data['cached_at'])
                age_hours = (datetime.now() - cached_at).total_seconds() / 3600

                if age_hours < ttl_hours:
                    cache_valid = True
                    print(f"📦 Using TTL cache ({age_hours:.1f}h old): {cache_path.name}")
                else:
                    print(f"⏰ Cache expired ({age_hours:.1f}h > {ttl_hours}h TTL)")

            if cache_valid:
                return data.get('value')

        # Fetch from API
        print(f"🌐 Fetching Fear & Greed from API: {date.strftime('%Y-%m-%d')}")

        try:
            import requests
            url = f"https://api.alternative.me/fng/?limit=30"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                api_data = response.json()

                # Find matching date
                target_timestamp = int(date.timestamp())
                for entry in api_data.get('data', []):
                    entry_timestamp = int(entry.get('timestamp', 0))

                    # Match within same day
                    if abs(entry_timestamp - target_timestamp) < 86400:
                        value = int(entry.get('value', 50))

                        # Save to cache
                        cache_data = {
                            'date': date.strftime('%Y-%m-%d'),
                            'value': value
                        }

                        if not is_finalized:
                            cache_data['cached_at'] = datetime.now().isoformat()
                            cache_data['ttl_hours'] = ttl_hours
                            cache_data['finalized'] = False

                        with open(cache_path, 'w') as f:
                            json.dump(cache_data, f)

                        if is_finalized:
                            print(f"💾 Cached permanently: {cache_path.name}")
                        else:
                            print(f"💾 Cached with {ttl_hours}h TTL: {cache_path.name}")

                        return value
        except Exception as e:
            print(f"Error fetching Fear & Greed: {e}")

        return None

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cached data

        Args:
            older_than_days: Only clear data older than N days (None = clear all)
        """
        if older_than_days is None:
            # Clear all
            for file in self.ohlcv_dir.glob("*.csv"):
                file.unlink()
            for file in self.fear_greed_dir.glob("*.json"):
                file.unlink()
            print("🗑️  Cleared all cached data")
        else:
            cutoff = datetime.now() - timedelta(days=older_than_days)
            count = 0

            # Clear old OHLCV
            for file in self.ohlcv_dir.glob("*.csv"):
                # Extract date from filename (format: BTC_USDT_1h_2025-12-02.csv)
                parts = file.stem.split('_')
                if len(parts) >= 4:
                    date_str = parts[-1]
                    try:
                        file_date = datetime.strptime(date_str, '%Y-%m-%d')
                        if file_date < cutoff:
                            file.unlink()
                            count += 1
                    except ValueError:
                        pass

            # Clear old Fear & Greed
            for file in self.fear_greed_dir.glob("*.json"):
                date_str = file.stem
                try:
                    file_date = datetime.strptime(date_str, '%Y-%m-%d')
                    if file_date < cutoff:
                        file.unlink()
                        count += 1
                except ValueError:
                    pass

            print(f"🗑️  Cleared {count} cached files older than {older_than_days} days")

    def get_cache_stats(self) -> Dict:
        """Get statistics about cached data"""
        ohlcv_files = list(self.ohlcv_dir.glob("*.csv"))
        fear_greed_files = list(self.fear_greed_dir.glob("*.json"))

        # Calculate total size
        ohlcv_size = sum(f.stat().st_size for f in ohlcv_files)
        fg_size = sum(f.stat().st_size for f in fear_greed_files)

        return {
            'ohlcv_count': len(ohlcv_files),
            'fear_greed_count': len(fear_greed_files),
            'ohlcv_size_mb': ohlcv_size / (1024 * 1024),
            'fear_greed_size_mb': fg_size / (1024 * 1024),
            'total_size_mb': (ohlcv_size + fg_size) / (1024 * 1024)
        }


def main():
    """Test cache functionality"""
    cache = DataCache()

    # Test OHLCV caching
    print("\n=== Testing OHLCV Cache ===")
    test_date = datetime(2025, 12, 2, 9, 0, 0)
    since = int(test_date.timestamp() * 1000)

    ohlcv = cache.fetch_ohlcv('BTC/USDT', '1h', since, 24)
    print(f"Fetched {len(ohlcv)} candles")

    # Test Fear & Greed caching
    print("\n=== Testing Fear & Greed Cache ===")
    fg_value = cache.fetch_fear_greed(test_date)
    print(f"Fear & Greed: {fg_value}")

    # Show stats
    print("\n=== Cache Statistics ===")
    stats = cache.get_cache_stats()
    print(f"OHLCV files: {stats['ohlcv_count']}")
    print(f"Fear & Greed files: {stats['fear_greed_count']}")
    print(f"Total size: {stats['total_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
