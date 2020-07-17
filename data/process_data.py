#imports libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads dataframes and merges them
    
    INPUT: File path to messages dataframe and categories dataframe
    OUTPUT: Merged file
    '''
    messages = pd.read_csv(messages_filepath) #reads dataframe and saves in variable
    categories = pd.read_csv(categories_filepath) #reads dataframe and saves in variable
    df = messages.merge(categories, how = 'outer', on = ['id']) #merges both dataframes
    return df #returns merged dataframe
    pass


def clean_data(df):
    
   '''
   This function cleans the data by reading, spliting, renaming, droping duplicates and NA values in the categories dataframe and concatenating it to the main dataframe.
   
   INPUT: Dataframe
   OUTPUT: Clean data
   '''
    categories = pd.read_csv('disaster_categories.csv') #reads data and saves in a variable
    categories = categories['categories'].str.split(';', expand = True) #split values in columns
    row = categories.iloc[0] #selects all strings in first row and stores in variable
    category_colnames = row.apply(lambda row: row.split("-")[0])  #splits selected values in row and selects the first string and store in variable
    categories.columns = category_colnames #replaces all column names
    
    for column in categories:
        #iterates through the values in a given column, splits strings and keep the second string value then converts to numerical value
        categories[column] = categories[column].str.split('-').str.get(1)
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop('categories', axis = 1) #drops column from dataframe
    df = pd.concat([df, categories], axis = 1) #concatenates two dataframes
    df.drop_duplicates(keep=False,inplace=True) #drops duplicates
    df = df.dropna(axis = 0, how = 'any') #drops all null(NaN) vales
    
    return df #returns clean data
    pass


def save_data(df, database_filename):
    '''
    This function saves the data in an sql database format
    
    INPUT: dataframe (data), database file path to save the data
    OUTPUT: Saves data in an sql data format
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename)) #creates home base for the database
    df.to_sql('DisasterResponse', engine, index=False) #converts dataframe to sql database
    return
    pass  


def main():
    '''
    Runs all functions above
    
    INPUT: None
    OUTPUT: Runs all functions
    '''
    #condition for length for lists of arguements
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath =  sys.argv[1:] #inputs for funtion

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath)) #prints strings
        df = load_data(messages_filepath, categories_filepath) #runs load_data function and saves in a variable df

        print('Cleaning data...') #prints strings
        df = clean_data(df) #runs the clean_data function and saves in a variable df
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath)) #prints strings
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!') #prints strings
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db') #prints strings

#executes code in the file
if __name__ == '__main__':
    main()