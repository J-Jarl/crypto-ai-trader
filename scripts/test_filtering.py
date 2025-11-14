"""Test the improved Bitcoin filtering and timestamp display"""
import feedparser
import time
from datetime import datetime

def format_relative_time(timestamp_struct):
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

def is_bitcoin_related(title: str, description: str) -> bool:
    """Check if article is primarily about Bitcoin"""
    title_lower = title.lower()
    description_lower = description.lower()

    # List of altcoin tickers to exclude
    altcoin_tickers = [
        'xrp', 'eth', 'ethereum', 'sol', 'solana', 'ada', 'cardano',
        'doge', 'dogecoin', 'bnb', 'binance', 'avax', 'avalanche',
        'dot', 'polkadot', 'matic', 'polygon', 'link', 'chainlink',
        'atom', 'cosmos', 'xlm', 'stellar', 'algo', 'algorand',
        'shib', 'shiba', 'uni', 'uniswap', 'ltc', 'litecoin',
        'popcat', 'pepe', 'bonk', 'floki', 'usdt', 'tether',
        'usdc', 'dai', 'busd'
    ]

    # If title contains altcoin ticker, exclude it
    for altcoin in altcoin_tickers:
        if f' {altcoin} ' in f' {title_lower} ' or title_lower.startswith(f'{altcoin} ') or title_lower.endswith(f' {altcoin}'):
            return False
        # Also check for ticker format like "ETH, SOL, ADA"
        if f'{altcoin},' in title_lower or f', {altcoin}' in title_lower:
            return False

    # Bitcoin must be mentioned
    bitcoin_keywords = ["bitcoin", "btc"]
    has_bitcoin = any(keyword in title_lower or keyword in description_lower for keyword in bitcoin_keywords)

    if not has_bitcoin:
        return False

    # Prefer Bitcoin in title
    has_bitcoin_in_title = any(keyword in title_lower for keyword in bitcoin_keywords)

    if not has_bitcoin_in_title:
        multi_coin_indicators = ['crypto', 'altcoin', 'tokens', 'coins']
        if any(indicator in description_lower for indicator in multi_coin_indicators):
            description_start = description_lower[:100]
            if not any(keyword in description_start for keyword in bitcoin_keywords):
                return False

    return True

# Test with CoinDesk feed
print("Testing Bitcoin filtering with CoinDesk RSS feed\n")
print("=" * 80)

feed = feedparser.parse("https://www.coindesk.com/arc/outboundfeeds/rss/")

bitcoin_articles = []
excluded_articles = []

for entry in feed.entries[:20]:  # Check first 20 articles
    title = entry.get("title", "")
    description = entry.get("summary", "")

    if is_bitcoin_related(title, description):
        timestamp = entry.published_parsed if hasattr(entry, 'published_parsed') else None
        bitcoin_articles.append((title, timestamp))
    else:
        excluded_articles.append(title)

print(f"\n[GREEN] INCLUDED ({len(bitcoin_articles)} Bitcoin articles):")
print("=" * 80)
for i, (title, timestamp) in enumerate(bitcoin_articles, 1):
    time_str = format_relative_time(timestamp)
    print(f"{i}. {title}")
    print(f"   Published {time_str}")
    print()

print(f"\n[RED] EXCLUDED ({len(excluded_articles)} non-Bitcoin articles):")
print("=" * 80)
for i, title in enumerate(excluded_articles, 1):
    print(f"{i}. {title}")

print("\n" + "=" * 80)
print(f"Summary: {len(bitcoin_articles)} Bitcoin articles, {len(excluded_articles)} excluded")
