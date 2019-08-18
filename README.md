# Disaster Response Pipeline Project

Installations
-----
The below installations are required for this notebook on IPL Data Analysis: Pandas, NumPy, MatPlotLib, seaborn, time, sklearn (SciKit) for the learning models, sklearn metrics- make_scorer, accuracy_score, f1_score, fbeta_score, classification_report, GridSearchCV, Pipeline, FeatureUnion,  etc. (please refer mode/train_classifier.py for all sklearn models used), plotly for visualizations, sqlalchemy for create_engine, NLTK for NLP processing using word_tokenize, stopwords and lemmatizer, os, re, JSON, Flask

Project Motivation
-----
In this exercise, we build on data engineering skills to expand your opportunities and potential as a data scientist. In this project, we apply these skills to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages.

In this project we have a data set containing real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that you we send the messages to an appropriate disaster relief agency.

Project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

File Descriptions
-----
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- DSND_Features_Class_GridSearch.txt # GridSearcghCV Logs and Feature Revall, Precision scores

- README.md

This project involves couples of steps:

### ETL

The first part of the data pipeline is the Extract, Transform, and Load process where we read the dataset, clean the data, and then store it in a SQLite database.process_data.py

### Machine Learning
For the machine learning portion, we split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). 

Finally, we export the model to a pickle file.

How to Interact with your project
-----

### Instructions:
Note: Please refer to Images folder for details screen shots for executing the scripts and running the web app
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Run at the Terminal ---> env|grep WORK

4. Not down the SPACE-ID & Space Domain frmo the above step

5. Go to http://SPACEID-3001.SPACEDOMAIN/ to view the web app


Acknowledgements
-----
1. FigureEight for providing the Disaster Data
2. Udacity for the awesome Data Scientist for Enterprise Nano Science program
