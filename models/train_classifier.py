
# How to run. E.g. below
# > python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

import sys
import pandas as pd
import numpy as np
import pickle
import re

from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import fbeta_score, classification_report

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(['punkt', 'wordnet', 'stopwords', 'averaged_perceptron_tagger'])
from scipy.stats.mstats import gmean

def load_data(database_filepath):
    '''
    FUNC
        As the name suggests, this function is to load the dat
    INPUT
        databasefilepath
    OUTPUT
        X feature matrix, y target vector & category names
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('df', engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
    '''
    FUNC
        Tokenize custom function. It involves multiple steps:
        Step1: Replace All URL's with 'urlplaceholder' text
        Step2: Call word_tokenize method on the text to convert to individual tokens
        Step3: Instantiate WordNet Lemmatizer, which converts tokens to their root form and lower the case
        
    INPUT
        text for tokenizing, lemmatizing & lower case conversion
    OUTPUT
        clean tokens
    '''        
    
    # the below will help idenitfy any possible URL's in the text
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    detected_urls = re.findall(url_regex, text)
    
    # replace URL with static text - "urlplaceholder"
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # tokenize the text
    tokens =  word_tokenize(text)

    # Instantiate Lemmatizer
    lem = WordNetLemmatizer()
    
    # loop through the tokens and lemmetize to root form e.g. rides, riding to 'ride'
    clean_tokens = []
    for tok in tokens:
        clean_tok = lem.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    
    """
    Starting Verb Extractor class
    
    This class extracts the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        # sentence tokenize
        sentence_list = nltk.sent_tokenize(text)
        
        #loop through sentence list
        for sentence in sentence_list:
            # Parts of Speech Tagging using NLTK pos_tag method
            pos_tags = nltk.pos_tag(tokenize(sentence))
            
            first_word, first_tag = pos_tags[0]
            
            # VB, VBP--> Verb base form and singular
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Model building using Pipeline and Feature Union
    CountVectorizer, TFIDF and MultiOutput Classifier are run here sequentially
    '''
    
    model = Pipeline([
        ('features', FeatureUnion([
            
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            
            ])),
            ('starting_verb', StartingVerbExtractor()),
        ])),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    return model

def multioutput_fscore(y_true,y_pred,beta=1):
    score_list = []
    if isinstance(y_pred, pd.DataFrame) == True:
        y_pred = y_pred.values
       
    if isinstance(y_true, pd.DataFrame) == True:
        y_true = y_true.values
        
    for column in range(0,y_true.shape[1]):
        score = fbeta_score(y_true[:,column],y_pred[:,column],beta,average='weighted')
        score_list.append(score)
    
    f1score_numpy = np.asarray(score_list)
    
    f1score_numpy = f1score_numpy[f1score_numpy<1]
    
    f1score = gmean(f1score_numpy)
    
    return  f1score

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Model Evaluation
    INPUT
        model, X_test, Y_test, categories
    OUTPUT
        Prints Accuracy and F1 scores
    '''
    Y_pred = model.predict(X_test)
    multi_f1 = multioutput_fscore(Y_test,Y_pred, beta = 1)
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
    pass

                         
def save_model(model, model_filepath):
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass
    
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
