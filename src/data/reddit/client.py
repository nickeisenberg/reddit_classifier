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
    pass

