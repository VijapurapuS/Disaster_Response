
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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from sklearn.model_selection import GridSearchCV

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
        '''
        FUNC 
            Checks for text being in starting verb and returns boolean values
        INPUT
            text
        OUTPUT
            boolean value
        '''
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
        '''
        INPUT
            X feature matrix, y = None
        OUTPUT
            returns self
        '''
        return self

    def transform(self, X):
        '''
        FUNC
            Transform Function
        INPUT
            X feature matrix
        OUTPUT
            X_tagged pandas dataframe after applying starting_verb defined above    
        '''
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)

def build_model():
    '''
    Model building using Pipeline and Feature Union
    CountVectorizer, TFIDF and MultiOutput Classifier are run here sequentially
    '''
# Adding the first model and commenting for mentor feedback. Updated model below
#     model = Pipeline([
#         ('vect', CountVectorizer(tokenizer=tokenize)),
#         ('tfidf', TfidfTransformer()),
#         ('clf', MultiOutputClassifier(RandomForestClassifier())),
#     ])

#     return pipeline
    
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
    '''
    FUNC
        Function defition for multioutput fscore
    INPUT
        Takes in three arguments, y_true (Actual result), y_pred (predicted value) and beta
    OUTPUT
        Returns the F1 score
    '''
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

def evaluate_model(model, X_test, y_test, category_names):
    '''
    Model Evaluation
    INPUT
        Takes 4 arguments model, X_test, y_test, categories
    OUTPUT
        Prints Accuracy and F1 scores
    '''
    y_pred = model.predict(X_test)
    multi_f1 = multioutput_fscore(y_test,y_pred, beta = 1)
    overall_accuracy = (y_pred == y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(multi_f1*100))
    
    # Mentor Feedback - adding in Classification Report for each category
    y_pred_pd = pd.DataFrame(y_pred, columns = y_test.columns)
    for col in y_test.columns:
            print('******************************************************\n')
            print('FEATURE: {}\n'.format(col))
            print(classification_report(y_test[col],y_pred_pd[col]))    
    pass

                         
def save_model(model, model_filepath):
    '''
    FUNC
        Saves the pickle model
    INPUT
        Takes in two args model and model file path
    OUTPUT
        Saves the model.
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))
    pass
    
def main():
    '''
    FUNC
        Main function. Program processing begins from here
    
    INPUT
        None
    
    OUTPUT
        Build, Train, Evaluate and Saves the trained model
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        #GridSearch for Parameter tuning and improving the model
        ### Mentor feedback ot include grid searchcv
        parameters = {
            'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            'features__text_pipeline__vect__max_df': (0.75, 1.0),
            'features__text_pipeline__vect__max_features': (None, 5000),
            'features__text_pipeline__tfidf__use_idf': (True, False),
        #    'clf__n_estimators': [10, 100],
        #    'clf__learning_rate': [0.01, 0.1],
        #    'features__transformer_weights': (
        #        {'text_pipeline': 1, 'starting_verb': 0.5},
        #        {'text_pipeline': 0.5, 'starting_verb': 1},
        #        {'text_pipeline': 0.8, 'starting_verb': 1},
        #    )
            }

        scorer = make_scorer(multioutput_fscore,greater_is_better = True)

        cv = GridSearchCV(model, param_grid=parameters, scoring = scorer,verbose = 2, n_jobs = -1)

        cv.fit(X_train, y_train)        
        
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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
