# import libraries
import asyncio

# import custom functions
from utils import fetch_and_save_articles

if __name__ == "__main__":
    # specifying rss feeds
    rss_feeds = [
        # "https://www.reddit.com/r/investing/.rss",
        # "https://www.reddit.com/r/stocks/.rss",
        "https://finance.yahoo.com/news/rssindex",
    ]

    # saving the texts into a text file
    asyncio.run(fetch_and_save_articles(rss_feeds))