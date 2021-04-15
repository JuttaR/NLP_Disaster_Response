from flask import Flask, render_template, request, url_for
import joblib
import json
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import pandas as pd
import plotly
from plotly.graph_objs import Bar, Heatmap
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
print("Connecting to SQL database...")
engine = create_engine('sqlite:///../data/Disaster.db')

print("Reading in SQL table as df...")
df = pd.read_sql_table('df', engine)

# load model
print("Loading model as model...")
model = joblib.load("../models/classifier.pickle")

print("Completed loading data and model.")

# index web page displays visuals and receives user input text for model


@app.route('/')
@app.route('/index')
def index():
    # extract data for visualizations
    genre_counts = df.groupby('genre').count()['message'].sort_values(ascending=False)
    genre_names = list(genre_counts.index)

    category_sums = df[df.columns[4:]].sum().sort_values(ascending=False)
    category_names = list(category_sums.index)

    correlation_list = []
    correlation = df.corr().values
    for c in correlation:
        correlation_list.append(list(c))

    # create 3 visualizations: Genres, Categories, Correlations
    graphs = [
        # 1: Genres
        {'data': [Bar(x=genre_names, y=genre_counts)],
         'layout': {'title': 'Distribution of Genres',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'title': "Genre"}}},
        # 2: Categories
        {'data': [Bar(x=category_names, y=category_sums)],
         'layout': {'template': 'plotly_dark',
                    'title': 'Distribution of Categories',
                    'yaxis': {'title': "Count"},
                    'xaxis': {'tickangle': 30}}},
        # 3: Correlations
        {'data': [Heatmap(z=correlation_list, x=category_names, y=category_names)],
         'layout': {'title': 'Correlation of Categories',
                    'height': 500,
                    'xaxis': {'tickangle': 30}}}
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # render go.html
    return render_template('go.html', query=query, classification_result=classification_results)


def main():
    app.run(host='127.0.0.1', port=5000, debug=True)


if __name__ == '__main__':
    main()
