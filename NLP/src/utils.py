# Standard library imports
import os
import re
from pathlib import Path

# Third-party imports
from bs4 import BeautifulSoup
import feedparser

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from newspaper import Article
from playwright.async_api import async_playwright
import requests

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm

try:
    from bertopic import BERTopic
    _HAS_BERTOPIC = True
except ImportError:
    _HAS_BERTOPIC = False

# Typing imports
from typing import Dict, List, Optional, Union


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

class TopicModeling:
    """
    A class to load text data from a directory, preprocess it, build a document-term matrix,
    train a topic model (LDA or BERTopic), and retrieve & label top keywords per topic.
    """

    def __init__(
        self,
        data_dir: str = "data/articles",
        n_topics: int = 10,
        n_top_words: int = 10,
        model_type: str = "lda"
    ):
        """
        Parameters:
        - data_dir: str — Path to directory containing .txt articles
        - n_topics: int — Number of topics to extract
        - n_top_words: int — Number of top words to display per topic
        - model_type: str — "lda" or "bertopic"
        """
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('wordnet', quiet=True)

        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        self.file_paths = sorted(self.data_dir.glob("*.txt"))
        if not self.file_paths:
            raise ValueError(f"No .txt files found in {data_dir}")

        self.docs: List[str] = [fp.read_text(encoding='utf-8') for fp in self.file_paths]
        self.filenames: List[str] = [fp.stem for fp in self.file_paths]

        self.n_topics = n_topics
        self.n_top_words = n_top_words
        self.model_type = model_type.lower()

        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        self.cleaned_docs: List[str] = []
        self.vectorizer: Optional[CountVectorizer] = None
        self.dtm = None
        self.feature_names: List[str] = []
        self.lda_model: Optional[LatentDirichletAllocation] = None
        self.bert_model = None
        self.bert_topics = None

    def preprocess(self) -> List[str]:
        """
        Tokenize, lowercase, remove non-alpha, remove stop words, and lemmatize.
        Returns a list of cleaned document strings.
        """
        cleaned = []
        for doc in self.docs:
            tokens = word_tokenize(doc.lower())
            tokens = [re.sub(r'[^a-z]', '', t) for t in tokens]
            tokens = [t for t in tokens if t and t not in self.stop_words]
            lemmas = [self.lemmatizer.lemmatize(t) for t in tokens]
            cleaned.append(" ".join(lemmas))

        self.cleaned_docs = cleaned
        return self.cleaned_docs

    def build_dtm(self, docs: Optional[List[str]] = None) -> None:
        """
        Builds a document-term matrix from preprocessed documents.
        """
        if docs is None:
            docs = self.cleaned_docs
        self.vectorizer = CountVectorizer(
            max_df=0.95,
            min_df=2,
            stop_words='english'
        )
        self.dtm = self.vectorizer.fit_transform(docs)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def train_lda(self) -> None:
        """
        Trains an LDA model on the document-term matrix.
        """
        if self.dtm is None:
            raise ValueError("Document-term matrix not built yet. Call build_dtm() first.")
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=10,
            learning_method='online',
            random_state=0
        )
        self.lda_model.fit(self.dtm)

    def train_bertopic(self) -> None:
        """
        Trains a BERTopic model on the preprocessed documents.
        Requires BERTopic installed.
        """
        if not _HAS_BERTOPIC:
            raise ImportError("Please install BERTopic to use this method.")
        self.bert_model = BERTopic(nr_topics=self.n_topics)
        self.bert_topics, _ = self.bert_model.fit_transform(self.cleaned_docs)

    def get_top_keywords(self) -> Dict[int, List[str]]:
        """
        Returns the top keywords for each topic.
        """
        topics = {}
        if self.model_type == 'lda':
            if self.lda_model is None:
                raise ValueError("LDA model not trained yet.")
            for idx, comp in enumerate(self.lda_model.components_):
                top_indices = comp.argsort()[:-self.n_top_words - 1:-1]
                topics[idx] = [self.feature_names[i] for i in top_indices]
        elif self.model_type == 'bertopic':
            if self.bert_model is None:
                raise ValueError("BERTopic model not trained yet.")
            for idx in range(self.n_topics):
                topic_info = self.bert_model.get_topic(idx)
                topics[idx] = [word for word, _ in topic_info]
        else:
            raise ValueError("Unknown model_type: choose 'lda' or 'bertopic'")
        return topics

    def label_topics(self, manual_labels: Dict[int, str]) -> Dict[int, str]:
        """
        Accepts a mapping from topic index to human-assigned label.
        Returns that mapping for easier reference.
        """
        return manual_labels
