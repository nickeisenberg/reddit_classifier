from flask import Flask, request, render_template

from src.classifier import RedditClassifier

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        classification = reddit_classifier(text)
        return render_template(
            'index.html', text=text, classification=classification
        )
    return render_template('index.html', text="", classification="")

if __name__ == '__main__':
    reddit_classifier = RedditClassifier()
    app.run(host='0.0.0.0', port=5000, debug=True)
