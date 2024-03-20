import os 
from tqdm import tqdm
from transformers import BertTokenizer
from src.data.reddit import RedditClient
from src.data.utils import lower_text_and_remove_all_non_asci

client_id = os.environ["PRAW_CLIENT_ID"]
client_secret = os.environ["PRAW_CLIENT_SECRET"] 
user_agent = os.environ["PRAW_USER_AGENT"]

reddit_client = RedditClient(client_id, client_secret, user_agent)

query = {"wallstreetbets": "Daily Discussion"}
num_submissions = 5
num_comments = 300

for subreddit_name in query:
    submissions = reddit_client.get_subreddit_submissions_by_key(
        subreddit_name, query[subreddit_name], num_submissions=num_submissions
    )
    for submission in tqdm(submissions):
        _ = submission.comments.replace_more(limit=0)
        with open("data/wsb.txt", "a") as af:
            for comment in submission.comments.list()[: num_comments]:
                _ = af.write(
                    lower_text_and_remove_all_non_asci(comment.body) + "\n"
                )

with open("data/wsb.txt", "r") as af:
    wsb = af.readlines()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

encoded_inputs = tokenizer(
    wsb, padding=True, truncation=True, max_length=256, return_tensors="pt"
)

enc = encoded_inputs["input_ids"][0]

tokenizer.decode(enc)

tokenizer.convert_ids_to_tokens([3681, 13334, 2102, 20915, 2015])

tokenizer.decode(tokenizer.encode("wallstreetbets"))
