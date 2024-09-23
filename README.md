# udacity_data_science_project_disaster

Table of Contents
Description
Dependencies
Installing and Executing


Description

This project "Disaster Response Pipeline" is the one of important parts of the Udacity Data Scientist Nanodegree with Figure Eight contribution. The aim of the project is to classify disaster messages based on model data from the people who are real disasters with the help of a Natural Language Processing Model (NLP).

Proceeding with the help of machine learning has the advantage that disaster messages coming with the various channels can be classified quickly without having to read through the entire text manually. This means that they can then be tagged by several disaster response agencies immediately and in a targeted manner to the right aid organizations.

This will help the disaster victims to receive prompt medical aid and speed recovery from the effects of the disasters.

This project is clustered in three Python scripts:

./process_data.py: Processing the data from real disaster messages, that came from Figure Eight in 2 csv files with an ETL pipeline: read the data from the csv-files, clean and wrangle it and save the data in a SQLite database.
./train_classifier.py: Build a NLP machine learning pipeline based on the data from the SQLite database to classify new disater messages.
./run.py Run a flask-based web app with a user interface, into which new messages can be entered, which are classified on the basis of the NLP model.

Dependencies
Python ver. 3.9 minimum
Data Processing: NumPy, Pandas
Machine Learning: Sciki-Learn
Natural Language Processing: NLTK
Model Saving and Loading: Scikit-Learn / Pickle
Web App and Visualization: Flask, Plotly
All of these modules can be installed by using the Anaconda package.
SQLlite Database: SQLalchemy

Installing and Executing:
Clone this Github repository to your local computer with python and the needed packages installed.

To run ETL pipeline to clean data and store the processed data in the database run from the project's root directory: python ./process_data.py ./messages.csv ./categories.csv ./DisasterResponse.db

To run the ML pipeline that loads data from DB, trains classifier and saves the classifier as a pickle file run from the project's root directory: python ./train_classifier.py ./DisasterResponse.db ./train_classifier.pkl

Go to directory:  run the web app: python run.py

Go to the displayed address in your web browser to see the web user interface, where you can enter a new disaster message to be categorized.

