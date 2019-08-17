# Disaster Response Pipeline Project

This project involves couples of steps:

### ETL

The first part of the data pipeline is the Extract, Transform, and Load process where we read the dataset, clean the data, and then store it in a SQLite database.process_data.py

### Machine Learning
For the machine learning portion, we split the data into a training set and a test set. Then, create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model that uses the message column to predict classifications for 36 categories (multi-output classification). 

Finally, we export the model to a pickle file.



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
