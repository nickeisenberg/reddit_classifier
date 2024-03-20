import re
import string
import praw
import os

from praw.reddit import Comment, Reddit, Submission, Subreddit
from collections.abc import Iterator
from tqdm import tqdm


class RedditClient:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent

        self._reddit = _get_reddit(client_id, client_secret, user_agent)


    def get_submission_comments_by_subreddit_search(self,
                                                   subreddit_name: str, 
                                                   search: str,
                                                   num_submissions: int,
                                                   num_comments_per_submission: int,
                                                   save_root: str):
        print(f"Fetching submissions for {subreddit_name}")
        submissions = self.get_submissions_by_subreddit_search(
            subreddit_name, search, num_submissions=num_submissions
        )
        pbar = tqdm(submissions)
        for i, submission in enumerate(pbar):
            pbar.set_postfix(name=subreddit_name, num=f"{i + 1}/{num_submissions}")
            _ = submission.comments.replace_more(limit=0)
            with open(os.path.join(save_root, f"{subreddit_name}.txt"), "a") as af:
                for comment in submission.comments.list()[: num_comments_per_submission]:
                    _ = af.write(
                        lower_text_and_remove_all_non_asci(comment.body) + "\n"
                    )
        return None


    def get_submissions_by_subreddit_search(self,
                                           subreddit_name: str, 
                                           search: str,
                                           num_submissions: int,
                                           sort: str = "revelance") -> list[Submission]:

        subreddit = self._reddit.subreddit(subreddit_name)
        submissions:Iterator[Submission] = subreddit.search(search, sort=sort)

        return [*submissions][: num_submissions]


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
    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent)
