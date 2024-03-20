import os
from praw.models import Comment
from tqdm import tqdm
from .utils import RedditClient, lower_text_and_remove_all_non_asci


def make_reddit_comment_dataset(client_id: str, 
                                client_secret: str, 
                                user_agent: str, 
                                searches: dict[str, str],
                                num_submissions_per_search: int,
                                num_comments_per_submission: int,
                                save_root: str):
    reddit_client = RedditClient(client_id, client_secret, user_agent)

    for subreddit_name in searches:
        os.makedirs(os.path.join(save_root, subreddit_name))

        print(f"Fetching submissions for {subreddit_name}")
        submissions = reddit_client.get_submissions_by_subreddit_search(
            subreddit_name, searches[subreddit_name], num_submissions_per_search
        )

        pbar = tqdm(submissions)        
        for i, submission in enumerate(pbar):
            pbar.set_postfix(
                sub_num=f"{i + 1}/{num_submissions_per_search}", 
                com_num=f"fetching"
            )

            _ = submission.comments.replace_more(limit=0)
            comments: list[Comment] = submission.comments.list()[: num_comments_per_submission]

            for j, comment in enumerate(comments):
                pbar.set_postfix(
                    sub_num=f"{i + 1}/{num_submissions_per_search}", 
                    com_num=f"{j + 1} / {len(comments)}"
                )

                clean_comment_body = lower_text_and_remove_all_non_asci(comment.body)

                save_to = os.path.join(
                    save_root, 
                    subreddit_name, 
                    f"{submission.id}_{comment.id}.txt"
                )
                with open(save_to, "a") as af:
                    af.write(clean_comment_body)

