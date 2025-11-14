"""Debug RSS feed parsing to check timestamps and filtering"""
import feedparser
import time

rss_feeds = [
    ("https://bitcoinmagazine.com/.rss/full/", "Bitcoin Magazine"),
    ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk"),
    ("https://cointelegraph.com/rss", "Cointelegraph")
]

print("Testing RSS feed parsing...\n")

for feed_url, source_name in rss_feeds:
    print(f"\n{'='*80}")
    print(f"Source: {source_name}")
    print(f"URL: {feed_url}")
    print(f"{'='*80}")

    try:
        feed = feedparser.parse(feed_url)

        # Show first 3 entries
        for i, entry in enumerate(feed.entries[:3], 1):
            print(f"\nArticle {i}:")
            print(f"  Title: {entry.get('title', 'N/A')}")

            # Check timestamp fields
            print(f"  Has published_parsed: {hasattr(entry, 'published_parsed')}")
            if hasattr(entry, 'published_parsed'):
                print(f"  published_parsed value: {entry.published_parsed}")
                print(f"  published string: {entry.get('published', 'N/A')}")

            print(f"  Has updated_parsed: {hasattr(entry, 'updated_parsed')}")
            if hasattr(entry, 'updated_parsed'):
                print(f"  updated_parsed value: {entry.updated_parsed}")
                print(f"  updated string: {entry.get('updated', 'N/A')}")

            # Test our time formatting
            if hasattr(entry, 'published_parsed') and entry.published_parsed:
                timestamp = entry.published_parsed
                published_time = time.mktime(timestamp)
                now = time.time()
                diff_seconds = now - published_time
                hours = int(diff_seconds / 3600)
                print(f"  Time ago: {hours} hours")

    except Exception as e:
        print(f"Error: {e}")

print("\n" + "="*80)
print("Testing complete")
