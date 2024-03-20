import os 
import re
import string
from tqdm import tqdm
from src.data.reddit import RedditClient

client_id = os.environ["PRAW_CLIENT_ID"]
client_secret = os.environ["PRAW_CLIENT_SECRET"] 
user_agent = os.environ["PRAW_USER_AGENT"]
reddit_client = RedditClient(client_id, client_secret, user_agent)

def clean_comment(comment):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251"  # Enclosed characters
                               "]+", flags=re.UNICODE)

    comment_text = comment.lower().strip()
    comment_text = re.sub(f"[{re.escape(string.punctuation)}]", '', comment_text)
    comment_text = re.sub("\n", " ", comment_text)
    comment_text = emoji_pattern.sub("", comment_text).strip()
    comment_text = re.sub(r"\w*emote\w*", "", comment_text).strip()
    comment_text = re.sub(r'\b\w{31,}\b', "", comment_text)
    return comment_text

query = {"wallstreetbets": "Daily Discussion"}

for subreddit_name in query:
    submissions = reddit_client.get_subreddit_submissions_by_key(
        subreddit_name, query[subreddit_name], num_submissions=2
    )
    for submission in tqdm(submissions):
        comments = submission.comments.list()[: 50]
        for comment in comments:
            comment_text = clean_comment(comment.body)
            with open("data/wsb.txt", "a") as write:
                _ = write.write(comment_text + "\n")
