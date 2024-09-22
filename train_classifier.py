
# import libraries
import sys

import pandas as pd
import os
import re
from joblib import dump
import numpy as np
from verbextractor import StartingVerbExtractor, tokenize

#import nltk
#nltk.download(['punkt', 'wordnet'])

#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sqlalchemy import create_engine


def load_data(database_filepath):
    """Load the data from the database given in filepath
    and store it in a dataframe

    Arguments:
        database_filepath -- Databse to load

    Returns:
        X, y: dataframes with independent und dependent variable(s)
        category_names: list of column names of y
    """
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponseTable', con=engine.connect())

    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = list(y.columns) 

    return X,y,category_names


def build_model():
    """Build the model with the machine learning algorithm and grid search

    Returns:
        model with the machine learning algorithm
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters =  {
        'clf__estimator__n_estimators': [10],
        'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters) 

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Makes a prediction and prints a classification report for the model 

    Arguments:
        model -- model to evaluate
        X_test -- test data dataframe with the independant variables
        Y_test -- test data dataframe with the dependant variable
        category_names -- list of column names of X_test
    """
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('=========================================================')
        print("Category:",category_names[i])
        print('=========================================================')
        print(classification_report(Y_test.iloc[:, i].values, y_pred[:, i], zero_division=0))
    


def save_model(model, model_filepath):
    """_summary_

    Arguments:
        model -- Model to save
        model_filepath -- File for saving the model
    """
    # save model with joblib dump
    dump(model, model_filepath) 
    #with open(model_filepath, 'wb') as f:
    #    pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()