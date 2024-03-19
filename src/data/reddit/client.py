from collections.abc import Iterator
from praw.reddit import Submission
from .utils import (
    _get_reddit, 
)

class RedditClient:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent

        self._reddit = _get_reddit(client_id, client_secret, user_agent)


    def get_subreddit_submissions_by_key(self,
                                         subreddit_name: str,
                                         search: str,
                                         num_submissions: int,
                                         sort: str = "revelance") -> list[Submission]:

        subreddit = self._reddit.subreddit(subreddit_name)
        submissions:Iterator[Submission] = subreddit.search(search, sort=sort)

        return [*submissions][: num_submissions]
    

if __name__ == "__main__":
    client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
    client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
    user_agent = 'phython_act'
    
    reddit_client = RedditClient(client_id, client_secret, user_agent)
    
    subs = reddit_client.get_subreddit_submissions_by_key(
        "wallstreetbets", "Daily Discussion", 2
    )
    
    coms = {}
    for i, sub in enumerate(subs):
        print(i)
        coms[len(coms)] = sub.comments.list()[5: 10]
    
    for com in coms[0]:
        print(com.body)
        print("\n\n\n\n")
