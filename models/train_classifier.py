#imports libraries
import sys
import nltk
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def load_data(database_filepath):
    '''
    This function loads data from a database filepath and filters variables needed for modeling
    
    INPUT: Database file path
    OUTPUT: X, Y varibales and category names
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath)) #creates home base for the data
    df = pd.read_sql_table('DisasterResponse', engine ) #reads sql table from home base stores in df variable
    category_names = list(df.columns[4:]) #checks for specified column names and saves in a list variable
    X = df.message.values #selects all values in 'message' column
    Y = df[df.columns[4:]].values #selects all values in specified columns
    Y = np.where(Y==2, 1, Y) #replaces values with number 2 by number 1 in the variable
    
    return X, Y, category_names #returns variables
    pass


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
        clean_tokens.append(clean_tok)

    return clean_tokens #retruns the cleaned text data
    pass


def build_model():
    '''
    This function creates machine learning model by createing a pipeline and enhancing it with GridSearch
    
    INPUT: None
    OUTPUT: Model
    
    '''
    #creates pipline for transforming and classiying text data
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())
    ])
    #parameters to improve model
    parameters = {
        'clf__n_estimators': [10],
        'clf__min_samples_split': [2]
    }
    #Searches over specified parameter values for an estimator
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model #returns model
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    '''Uses existing data to predict and prints out precision, recall, f1-scores
    INPUT: Model, test data and category names
    OUTPUT: Returns the result for each line of column
    '''
    X, Y, category_names = load_data('/home/workspace/data/DisasterResponse.db') #uses load_data function to load data needed
    y_pred = model.predict(X_test) #predicts with model
    for i, c in enumerate(Y[0]):
        #iterates through data to provide a classification report for the data
        evaluate = classification_report(Y_test[i],  y_pred[i])
    return
    pass


def save_model(model, model_filepath):
    '''This function saves the model as a pickle file
    
    INPUT: model, model's file path
    OUTPUT: saves the model to file path
    '''
    #saves model in specified file path
    pickle.dump(model, open('/home/workspace/models/{}'.format(model_filepath), 'wb'))
    return
    pass


def main():
    '''
    Runs all functions above
    
    INPUT: None
    OUTPUT: Runs all functions
    '''
    #condition for length for lists of arguements
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath)) #prints strings
        X, Y, category_names = load_data(database_filepath) #runs the load_data function
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2) #splits the data into train and test sets
        
        print('Building model...') #prints strings
        model = build_model() #runs the build_model function
        
        print('Training model...') #prints strings
        model.fit(X_train, Y_train) #trains pipeline
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names) #runs the evaluate_model function

        print('Saving model...\n    MODEL: {}'.format(model_filepath)) #prints strings
        save_model(model, model_filepath) #runs the save_model function

        print('Trained model saved!') #prints strings

    else:
        #prints message if input is not correct
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl') #prints strings

#executes code in the file
if __name__ == '__main__':
    main()