from collections import defaultdict
import string
from src.data.reddit import RedditClient

client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
user_agent = 'phython_act'

reddit_client = RedditClient(client_id, client_secret, user_agent)

query = {"wallstreetbets": "Daily Discussion"}

sr_and_subs = {} 
sr_and_coms = defaultdict(list)
for subreddit_name in query:
    sr_and_subs[subreddit_name] = reddit_client.get_subreddit_submissions_by_key(
        subreddit_name, query[subreddit_name], num_submissions=2
    )
    for submission in sr_and_subs[subreddit_name]:
        for comment in submission.comments.list()[: 50]:
            sr_and_coms[subreddit_name].append(
                comment.body.lower().strip().translate(
                    str.maketrans("", "", string.punctuation)
                ).replace("\n", " ")
            )

words = sr_and_coms["wallstreetbets"][0]

print(words)
