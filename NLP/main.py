# import libraries
import asyncio

# import custom functions
from utils import scrape_with_playwright

if __name__ == "__main__":
    url = "https://finance.yahoo.com/news/amd-vs-arista-networks-artificial-111700421.html"
    output_file = "financial_news.txt"
    asyncio.run(scrape_with_playwright(url, output_file))


