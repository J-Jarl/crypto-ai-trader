"""
Quick test script to verify RSS feeds are working
"""

import sys
import os

# Add parent directory to path so we can import from trading_ai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from trading_ai import BitcoinNewsAggregator

    print("=" * 80)
    print("Testing RSS Feed News Aggregator")
    print("=" * 80)
    print()

    # Create aggregator without NewsAPI key (will use RSS feeds only)
    aggregator = BitcoinNewsAggregator(newsapi_key=None)

    print("Fetching articles from RSS feeds...")
    print("Sources: CoinDesk, Cointelegraph, Bitcoin Magazine")
    print()

    articles = aggregator.fetch_news(max_articles=10)

    print(f"Successfully fetched {len(articles)} articles")
    print()
    print("=" * 80)
    print("RECENT BITCOIN NEWS HEADLINES")
    print("=" * 80)
    print()

    for i, article in enumerate(articles, 1):
        print(f"{i}. [{article.source}] {article.title}")
        if article.description:
            # Limit description to 150 chars
            desc = article.description[:150] + "..." if len(article.description) > 150 else article.description
            print(f"   {desc}")
        print(f"   Published: {article.published_at}")
        print(f"   URL: {article.url}")
        print()

    print("=" * 80)
    print("RSS Feed Test Complete!")
    print("=" * 80)

except ImportError as e:
    print(f"Error importing required modules: {e}")
    print()
    print("Make sure to install dependencies first:")
    print("pip install -r requirements.txt")
except Exception as e:
    print(f"Error during test: {e}")
    import traceback
    traceback.print_exc()
