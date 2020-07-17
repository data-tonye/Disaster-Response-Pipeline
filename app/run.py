#imports libraries
import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__) #stores the name the current python module 

def tokenize(text):
    '''
    This function processes text data making it appropriate for transformation and classification.
    
    INPUT: text data
    OUTPUT: clean text data
    '''
    tokens = word_tokenize(text) #tokenizes text data
    lemmatizer = WordNetLemmatizer() #coverts words to base form

    clean_tokens = [] #empty list
    for tok in tokens:
        #iterates through tokens, cleans text data and appends it to an empty list
        clean_tok = lemmatizer.lemmatize(tok).lower().strip() 
        clean_tokens.append(clean_tok) #retruns the cleaned text data

    return clean_tokens

engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db') #creates home base for the data
df = pd.read_sql_table('DisasterResponse', engine) #reads sql table from home base stores in df variable

model = joblib.load("/home/workspace/models/classifier.pkl") #loads model


#index webpage displays cool visuals and receives user input text for model
@app.route('/') #decorator to tell Flask what URL should trigger the function
@app.route('/index') #decorator to tell Flask what URL should trigger the function

def index():
    
    '''
    Extracts data needed for visuals
    '''
    #extracts specified data from different columns of the dataframe and their values
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    weather_counts = df.groupby('weather_related').count()['message']
    weather_names = list(weather_counts.index)
    
    request_counts = df.groupby('request').count()['message']
    request_names = list(request_counts.index)
    
    # creates visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts 
                ) #values for the axis in the bar chart
            ],

            'layout': {
                'title': 'Distribution Based on Genre',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                } #sets layout of the visual
            }
        },
        {
            'data': [
                Bar(
                    x=weather_names,
                    y=weather_counts
                ) #values for the axis in the bar chart
            ],

            'layout': {
                'title': 'Distribution Based on Weather Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Weather Related"
                } #sets layout of the visual
            }
        },
        {
            'data': [
                Bar(
                    x=request_names,
                    y=request_counts
                ) #values for the axis in the bar chart
            ],

            'layout': {
                'title': 'Distribution on Related Messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Related"
                } #sets layout of the visual
            }
        }
    ]
    
    #encodes plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    #renders web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


#web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # uses model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This renders the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

#executes code in the file
def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()