# About
A transformer classifier trained on comments from 10 popular subreddits. The
subreddits choosen all have a popular "discussion" chat post and 100 comments
were scraped using `praw` from the last 100 "discussion" posts of the
subreddits. All of the discussion posts except the `movies` posts shared a
common theme. The `movies` "discussion" posts were all "discussions" about a
particular movie, like [this
post](https://www.reddit.com/r/movies/comments/1b3jo9s/official_discussion_dune_part_two_spoilers/)
, while the "discussion" posts from the other subreddits were all general
"daily discussion" type posts [this this soccer
post](https://www.reddit.com/r/soccer/comments/1cdcxww/daily_discussion/) for
examples. The latest comments in the dataset occured on April 11, so any
comment created after April 11 will be new to the model.

To download and unzip the data, use
```bash
curl -o data.tar.gz -L 'https://drive.google.com/uc?export=download&confirm=yes&id=1slCTIg_iJ3ggcMyHTq7yuCcZGqp96-at'
tar -xzvf data.tar.gz
```
Alternatively, you can down download the data directly from [here](https://drive.google.com/drive/folders/1MdYmlTaZhMoeRNwAw3jc0pDtTFLvWMH-).


# Some Results
Below are the results of the transformer on the testing data after 40 epochs of
training and validating. The transformer had 6 "multihead attention" layers
with 4 heads each and an embedded dimension of 256. A max token length of 256
was also chosen. The full configuration of the experiment is in
`./experiment/config.py`. I saved the best model based on
validation accuracy, which occured on the 29th epoch.

![](./experiment/metrics/evaluation_ep0.png) 

![](./experiment/loss_logs/loss_and_accuracy.png) 


# Try it out
There is an app in `./app`. To try it out you will need the weights
[found here](https://drive.google.com/drive/folders/1MdYmlTaZhMoeRNwAw3jc0pDtTFLvWMH-).

```bash
git clone https://github.com/nickeisenberg/reddit_classifier.git
cd reddit_classifier
gdown 1R7JVUWz5h02T8c7UqsWI8pZjkDjfKvKa
mv validation_ckp.pth app
pip install -r rec.app.txt
```
Then install the `cpu` version of pytorch for you OS type found [here](https://pytorch.org/) 
and run
```bash
python app
```
