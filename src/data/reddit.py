import re
import os
import string

from tqdm import tqdm
from random import seed, shuffle

from praw import Reddit
from praw.models import Comment
from praw.reddit import Comment, Submission, Subreddit
from collections.abc import Iterator


def make_reddit_comment_dataset(client_id: str, 
                                client_secret: str, 
                                user_agent: str, 
                                searches: dict[str, str],
                                num_submissions_per_search: int,
                                num_comments_per_submission: int,
                                train_val_test: tuple,
                                save_root: str):

    assert sum([int(x * 10) for x in train_val_test])
    reddit_client = RedditClient(client_id, client_secret, user_agent)

    for subreddit_name in searches:
        os.makedirs(os.path.join(save_root, subreddit_name, "train"))
        os.makedirs(os.path.join(save_root, subreddit_name, "val"))
        os.makedirs(os.path.join(save_root, subreddit_name, "test"))

        print(f"Fetching submissions for {subreddit_name}")
        submissions = reddit_client.get_submissions_by_subreddit_search(
            subreddit_name, searches[subreddit_name], num_submissions_per_search
        )

        seed(1)
        shuffle(submissions)

        pbar = tqdm(submissions)        
        for i, submission in enumerate(pbar):
            if i < len(pbar) * train_val_test[0]:
                which = "train"
            elif i > len(pbar) * train_val_test[0] and i < len(pbar) * sum(train_val_test[:2]):
                which = "val"
            else:
                which = "test"

            pbar.set_postfix(
                sub_num=f"{i + 1}/{num_submissions_per_search}", 
                com_num=f"fetching"
            )

            _ = submission.comments.replace_more(limit=0)
            comments: list[Comment] = submission.comments.list()

            for j, comment in enumerate(comments[: num_comments_per_submission]):
                pbar.set_postfix(
                    sub_num=f"{i + 1}/{num_submissions_per_search}", 
                    com_num=f"{j + 1} / {len(comments)}"
                )

                clean_comment_body = lower_text_and_remove_all_non_asci(comment.body)

                save_to = os.path.join(
                    save_root, 
                    subreddit_name, 
                    which,
                    f"{submission.id}_{comment.id}.txt"
                )
                with open(save_to, "a") as af:
                    af.write(clean_comment_body)


def get_comments_from_submission_id_list(reddit: Reddit,
                                         submission_ids: list[str],
                                         num_comments_per_submission: int,
                                         save_root: str):

    os.makedirs(save_root)

    pbar = tqdm(submission_ids)        
    for i, submission_id in enumerate(pbar):
        pbar.set_postfix(
            sub_num=f"{i + 1}/{len(submission_ids)}", 
            com_num=f"fetching"
        )

        submission: Submission = reddit.submission(submission_id)
        _ = submission.comments.replace_more(limit=0)
        comments: list[Comment] = submission.comments.list()

        for j, comment in enumerate(comments[: num_comments_per_submission]):
            pbar.set_postfix(
                sub_num=f"{i + 1}/{len(submission_ids)}", 
                com_num=f"{j + 1} / {len(comments)}"
            )

            clean_comment_body = lower_text_and_remove_all_non_asci(comment.body)

            save_to = os.path.join(
                save_root, 
                f"{submission.id}_{comment.id}.txt"
            )
            with open(save_to, "a") as af:
                af.write(clean_comment_body)


class RedditClient:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent

        self._reddit = _get_reddit(client_id, client_secret, user_agent)


    def get_submissions_by_subreddit_search(self,
                                           subreddit_name: str, 
                                           search: str,
                                           num_submissions: int,
                                           sort: str = "revelance") -> list[Submission]:

        subreddit: Subreddit = self._reddit.subreddit(subreddit_name)
        submissions:Iterator[Submission] = subreddit.search(search, sort=sort)

        return [*submissions][: num_submissions]


    def get_submissions_by_id(self, submission_ids: list[str]) -> list[Submission]:
        return [self._reddit.submission(id) for id in submission_ids]


def remove_emojis_by_type(comment):
    emoji_pattern = re.compile(
        "["
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
        "]+", flags=re.UNICODE
    )
    comment = emoji_pattern.sub("", comment)
    return comment


def lower_text_and_remove_all_non_asci(text):
    text = text.lower().strip()
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub("\n", " ", text)
    text = re.sub(r'[^\x00-\x7F]', '', text)
    text = re.sub(r"\w*emote\w*", "", text)
    text = re.sub(r'\b\w{31,}\b', "", text).strip()
    return text


def _get_submission_list_from_subreddit(subreddit: Subreddit,
                                       sort_by: str ='top',
                                       search: str | None  = None,
                                       search_sort_by: str = 'relevance',
                                       no_of_submissions: int = 10):
    """
    Returns a list of praw Submissions from a praw Subreddit
    """

    print('starting submission getter')

    if isinstance(search, str):
        submissions = subreddit.search(search, sort=search_sort_by)
    elif sort_by == 'top':
        submissions = subreddit.top(limit=no_of_submissions)
    elif sort_by == 'hot':
        submissions = subreddit.hot(limit=no_of_submissions)
    elif sort_by == 'new':
        submissions = subreddit.new(limit=no_of_submissions)
    elif sort_by == 'rising':
        submissions = subreddit.rising(limit=no_of_submissions)
    else:
        raise Exception("")

    submission_list: list[Submission] = []
    count = 1
    for sub in submissions:
        submission_list.append(sub)
        if count == no_of_submissions:
            break
        count += 1

    return submission_list


def _get_comments_from_submission(submission: Submission,
                                 num_comments=10):
    """
    Retrieve comments from a submission
    """
    comment_list: list[Comment] = submission.comments.list()[: num_comments]
    return comment_list


def _get_comments_from_submission_list(submission_list: list[Submission],
                                      num_comments=10):

    submission_coms: dict[Submission, list[Comment]] = {
        submission: [] for submission in submission_list
    }

    for i, submission in enumerate(list(submission_coms.keys())):
        print(f'{i} / {len(submission_list)}: {submission.title}')
        submission_coms[submission] = _get_comments_from_submission(
            submission, num_comments
        )

    return submission_coms


def _get_replies_of_comment(submission_list: list[Subreddit],
                           submission_comments,
                           no_of_replies=10):

    print('starting comments replies')

    submissions_comments_replies = {sub: {} for sub in submission_list}

    for sub in submission_list:
        comments_replies = {com: [] for com in submission_comments[sub]}
        count_c = 1
        for com in submission_comments[sub]:
            print(f'COMMENT {count_c}')
            replies = com.replies
            replies.replace_more(limit=None)
            replies = replies[: no_of_replies]
            for reply in replies:
                comments_replies[com].append(reply)
            count_c += 1

        submissions_comments_replies[sub] = comments_replies

    return submissions_comments_replies


def _get_reddit(client_id: str, client_secret: str, user_agent: str):
    return Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent)

if __name__ == "__main__":
    import os
    
    x = Reddit(
        client_id=os.environ["PRAW_CLIENT_ID"],
        client_secret=os.environ["PRAW_CLIENT_SECRET"],
        user_agent=os.environ["PRAW_USER_AGENT"])
    wsb: Subreddit = x.subreddit("wallstreetbets")
    subs = wsb.search("Daily")
    sub = list(subs)[0]
    print(sub.id)
    x.submission(sub.id)
