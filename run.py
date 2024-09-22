import json
import plotly
import pandas as pd
import nltk
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine

sys.path.insert(0, '../models')
from verbextractor import StartingVerbExtractor


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
engine = create_engine('sqlite:///../data/DisasterResponseDB.db')
df = pd.read_sql_table('DisasterResponseTable', con=engine.connect())

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    #visuals:
    # - bar chart with the message counts for the different genre type 
    # - bar chart with the message counts for the different category types
    # - bar chart with the message counts for the different category types within the genre direct  
    # - bar chart with the message counts for the different category types within the genre news
    # - bar chart with the message counts for the different category types within the genre social

    #Message count for the genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    #Message counts for the categories
    category_counts = df.iloc[:,4:].sum().sort_values(ascending=False)
    category_names = list(category_counts.index)

    # Get the top ten categories in the 3 diffrent genres
    direct_category_counts = df[df['genre'] == 'direct'].iloc[:,4:].sum().sort_values(ascending=False)[0:10]
    direct_category_names = list(direct_category_counts.index)

    news_category_counts = df[df['genre'] == 'news'].iloc[:,4:].sum().sort_values(ascending=False)[0:10]
    news_category_names = list(news_category_counts.index)

    social_category_counts = df[df['genre'] == 'social'].iloc[:,4:].sum().sort_values(ascending=False)[0:10]
    social_category_names = list(social_category_counts.index)


    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],
            
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=direct_category_names,
                    y=direct_category_counts
                )
            ],
            
            'layout': {
                'title': 'TOP 10 Message Categories In The Direct Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=news_category_names,
                    y=news_category_counts
                )
            ],
            
            'layout': {
                'title': 'TOP 10 Message Categories In The News Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=social_category_names,
                    y=social_category_counts
                )
            ],
            
            'layout': {
                'title': 'TOP 10 Message Categories In The Social Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
        

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

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()