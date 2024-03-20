from .utils import RedditClient

def make_reddit_comment_dataset(client_id: str, 
                                client_secret: str, 
                                user_agent: str, 
                                searches: dict[str, str],
                                num_submissions_per_search: int,
                                num_comments_per_submission: int,
                                save_root: str):
    reddit_client = RedditClient(client_id, client_secret, user_agent)
    for subreddit_name in searches:
        reddit_client.get_submission_comments_by_subreddit_search(
            subreddit_name, 
            searches[subreddit_name], 
            num_submissions_per_search, 
            num_comments_per_submission, 
            save_root
        )
