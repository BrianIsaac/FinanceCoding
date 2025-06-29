# import libraries
import requests
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