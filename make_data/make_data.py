import os 
from sys import path
path.append(os.getcwd())
from src.data.reddit import make_reddit_comment_dataset


if __name__ == "__main__":
    client_id = os.environ["PRAW_CLIENT_ID"]
    client_secret = os.environ["PRAW_CLIENT_SECRET"] 
    user_agent = os.environ["PRAW_USER_AGENT"]
    
    searches = {
        # "wallstreetbets": "Daily Discussion Thread",
        "CryptoCurrency": "Daily Crypto Discussion",
        # "soccer": "Daily Discussion",
        # "movies": "Official Discussion",
        "formula1": "Daily Discussion",
    }
    
    make_reddit_comment_dataset(
        client_id, 
        client_secret, 
        user_agent, 
        searches,
        100,
        100,
        (.7, .2, .1),
        "data"
    )
