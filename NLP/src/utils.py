# import libaries
# standard library imports
import os
import re

# third-party imports
import requests
import feedparser
from newspaper import Article
from playwright.async_api import async_playwright
from tqdm import tqdm
from bs4 import BeautifulSoup

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

async def scrape_with_playwright(url: str, output_file: str) -> None:
    """
    Scrape and save the fully rendered article text from a Yahoo Finance page using Playwright and BeautifulSoup.

    Parameters:
    - url (str): The target URL of the web page to scrape.
    - output_file (str): The file name for saving the extracted text in the current working directory.

    Returns:
    - None
    """
    # build output path
    output_path = os.path.join(os.getcwd(), output_file)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        await page.goto(url, timeout=60000)
        # wait for the article container
        await page.wait_for_selector('[data-testid="article-content-wrapper"]')

        # grab the rendered HTML
        html = await page.content()
        await browser.close()

    # parse it
    soup = BeautifulSoup(html, 'html.parser')
    # find ONLY the paragraphs in the article
    paras = soup.find_all('p', class_='yf-1090901')
    text = "\n\n".join(p.get_text(strip=True) for p in paras)

    # write out
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved {len(paras)} paragraphs from {url} to {output_file}")

async def fetch_and_save_articles(rss: list[str]) -> None:
    """
    Fetch article URLs from RSS feeds, scrape each with Playwright + BeautifulSoup,
    and save each article as a separate .txt under data/articles/ in the current directory.

    Parameters:
    - rss (List[str]): A list of RSS feed URLs to pull links from.

    Returns:
    - None
    """
    # Ensure the output directory exists
    articles_dir = os.path.join(os.getcwd(), "data", "articles")
    os.makedirs(articles_dir, exist_ok=True)

    # Spin up Playwright once
    async with async_playwright() as p:
        for feed_url in rss:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries:
                url = entry.link

                # Build a filesystem-safe filename from the title
                title = entry.get("title", "article")
                slug = re.sub(r"[^\w\-]+", "_", title)[:50].strip("_")
                filename = f"{slug}.txt"
                output_path = os.path.join(articles_dir, filename)

                print(f"Fetching {url} → {filename}")
                try:
                    # Scrape with Playwright
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, timeout=60000)
                    await page.wait_for_selector('[data-testid="article-content-wrapper"]')
                    html = await page.content()
                    await browser.close()

                    # Parse and extract only the article paragraphs
                    soup = BeautifulSoup(html, 'html.parser')
                    paras = soup.find_all('p', class_='yf-1090901')
                    text = "\n\n".join(p.get_text(strip=True) for p in paras)

                    # Write out the cleaned text
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text)

                    print(f"  ✓ Saved {len(paras)} paragraphs")
                except Exception as e:
                    print(f"  ✗ Failed to fetch {url}: {e}")

async def fetch_and_save_articles_tqdm(rss: list[str]) -> None:
    """
    Fetch article URLs from RSS feeds, scrape each with Playwright + BeautifulSoup,
    and save each article as a separate .txt under data/articles/ in the current directory,
    with progress bars via tqdm.

    Parameters:
    - rss (List[str]): A list of RSS feed URLs to pull links from.

    Returns:
    - None
    """
    # 1. Ensure the output directory exists
    articles_dir = os.path.join(os.getcwd(), "data", "articles")
    os.makedirs(articles_dir, exist_ok=True)

    # 2. Launch Playwright once
    async with async_playwright() as p:
        # Progress bar over feeds
        for feed_url in tqdm(rss, desc="RSS feeds"):
            feed = feedparser.parse(feed_url)

            # Progress bar over entries in this feed
            for entry in tqdm(feed.entries, desc="Articles", leave=False):
                url = entry.link

                # Build a filesystem-safe filename from the title
                title = entry.get("title", "article")
                slug = re.sub(r"[^\w\-]+", "_", title)[:50].strip("_")
                filename = f"{slug}.txt"
                output_path = os.path.join(articles_dir, filename)

                try:
                    # Scrape with Playwright
                    browser = await p.chromium.launch(headless=True)
                    page = await browser.new_page()
                    await page.goto(url, timeout=60000)
                    await page.wait_for_selector('[data-testid="article-content-wrapper"]')
                    html = await page.content()
                    await browser.close()

                    # Parse and extract only the article paragraphs
                    soup = BeautifulSoup(html, 'html.parser')
                    paras = soup.find_all('p', class_='yf-1090901')
                    text = "\n\n".join(p.get_text(strip=True) for p in paras)

                    # Write out the cleaned text
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(text)

                except Exception as e:
                    tqdm.write(f"Failed to fetch {url}: {e}")