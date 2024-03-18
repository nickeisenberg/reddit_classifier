import praw
from praw.reddit import Comment, Reddit, Submission, Subreddit


class RedditClient:
    def __init__(self, client_id: str, client_secret: str, user_agent: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.user_agent = user_agent

        self.reddit = _get_reddit(client_id, client_secret, user_agent)


    def get_comments_from_submission_list(self, 
                                          subreddit_name: str, 
                                          num_submissions: int, 
                                          num_comments: int,
                                          search: str):
    
        subreddit = _get_subreddit(self.reddit, subreddit_name)
    
        subs = _get_submission_list_from_subreddit(
            subreddit=subreddit,
            search=search,
            no_of_submissions=num_submissions
        )
        
        coms = _get_comments_from_submission_list(
            submission_list=subs,
            num_comments=num_comments
        )
    
        return coms


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

    print('Getting comments for...')

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


def _get_subreddit(reddit: Reddit, subreddit_name: str):
    return reddit.subreddit(subreddit_name)


if __name__ == "__main__":
    client_id = 'NA7yHLvDpSwcAEA1nmibNQ'
    client_secret = 'qq_Odz0Vq6vIOPziOz2JYnoUJkMPog'
    user_agent = 'phython_act'
    
    reddit_client = RedditClient(client_id, client_secret, user_agent)
    
    coms = reddit_client.get_comments_from_submission_list(
        "wallstreetbets",
        1,
        5,
        "Daily"
    )
