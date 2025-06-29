# import libaries
# standard library imports
import os

# third-party imports
import requests
import feedparser
from newspaper import Article

# typing imports
from typing import Union

def get_marketaux_news(symbols: list[str], api_key: str) -> tuple[int, Union[dict, list]]:
    """
    Retrieve news articles for specified stock symbols using MarketAux API.

    Parameters:
    - symbols (List[str]): A list of stock symbols to search for.
    - api_key (str): API key for MarketAux.

    Returns:
    - Tuple[int, Union[dict, list]]: A tuple containing the response status code and the parsed JSON response.
    """

    # convert symbols list into a string
    tickers = ",".join(symbols)
    
    # create url for request
    url = f"https://api.marketaux.com/v1/news/all?symbols={tickers}&filter_entities=true&language=en&api_token={api_key}"

    # call marketaux for news
    response = requests.get(url)

    return response.status_code, response.json()

def get_raw_news_rss(rss: list[str], output: str, charas: int = -1) -> None:
    """
    Retreives news articles from specified RSS feeds using feedparser and newspaper3k 
    and dumps it into a txt file for future use.

    Parameters:
    - rss (List[str]): A list of RSS feed links to search for.
    - output (str): output filename for raw news to current directory.
    - charas (int): Number of charas to save (default to everything).
    """
    # determine the output path
    output_path = os.path.join(os.getcwd(), output)

    # dumping the texts into a text file
    with open(output_path, "w", encoding="utf-8") as f:
        for feed_url in rss:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                url = entry.link
                print(f"Processing: {url}")

                try:
                    article = Article(url)
                    article.download()
                    article.parse()

                    # Determine text length
                    text = article.text
                    if charas != -1:
                        text = text[:charas]

                    # Write to file with header separator
                    f.write("="*80 + "\n")
                    f.write(f"Title: {article.title}\n")
                    f.write(f"Publish Date: {article.publish_date}\n")
                    f.write(f"URL: {url}\n\n")
                    f.write(text + "\n\n")

                    print(f"Saved: {article.title}")
                    print("-" * 100)

                except Exception as e:
                    print(f"Failed to process {url}. Reason: {e}")

    print(f"\nNews dump completed. Saved to current directory as: {output}")