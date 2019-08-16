#########################################
# Execution Step:
# > python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
#########################################

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(message_filepath, categories_filepath):
    
    '''
    
    
    load_data function --> takes two inputs messages_file path and categories_filepath
    and loads the datasets for concatenation and input to next steps as defined in main()
    
    INPUT
        Messages and categories csv filepaths
    
    OUTPUT
        Concatenated data frame 'df'merged on 'id'
    
    '''
    
    # load messages dataset
    messages = pd.read_csv(message_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge the messages and categories datasets using the common id
    df = pd.merge(messages, categories, on = 'id')
          
    return df


def clean_data(df):

    '''
    DOC STRING
    
    clean_data function --> Cleasn the data frame by renaming the columns and splitting in to 36 categories 

    
    INPUT
        'df' data frame from the load_data function above
    
    OUTPUT
        returns cleaned 'df' by renaming columns, dropping duplicates
    
    '''
    # Split categories in to 36 different categories
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:].values
    
    # use this row to extract a list of new column names for categories.
    category_colnames = [r[:-2] for r in row]
    
    # rename the columns of `categories`
    for column in categories:
        #renaming the columns in categories
        categories[column] = categories[column].str[-1]
    
        # converting the string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the categories column from the df dataframe since it is no longer needed
    df.drop('categories', axis=1, inplace=True)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    
    # Remove duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    return df


def save_data(df, database_filename):
    """
    save_data function--> saves the cleaned data in to sqlite database
    
    INPUT
        cleaned 'df' dataframe
        database_filename for saving
    
    OUTPUT
    
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('df', engine, index=False)
    
    pass  


def main():
    """
    main function, program execution starts here
    
    Steps:
        1) Checks if four mandatory parameters (arguments) are provided by the user at command prompt. If not, prompts user
        to make the correction
        2)Reads in the file paths for messages, categories and database from the arguments provided
        3) load_data functions is called
        4) clean_data function is called
        5) save_data function is called saving cleaned data to database sqllite.
        
    """
    
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()