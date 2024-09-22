import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """The function loads the date from the csv files given in the arguments,
    merges both datasets and stres the merged data in a dataframe as output.  

    Arguments:
        messages_filepath -- filepath/filename for the csv-file  with the messages
        categories_filepath -- filepath/filename for the csv-file with the categories
    
    Output:
        Dataframe with the merged data from both csv-files
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    #merge both dataframes
    df = pd.merge(messages, categories, on = 'id')

    return df


def clean_data(df):
    """Cleans the data given in the Pandas dataframe df

    Arguments:
        df -- Pandas dataframe to clean
    OUTPUT:
        df -- Cleaned dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories
    # by applying a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x:x[:-2]).tolist()

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replace `categories` column in `df` with new category columns.

    # drop the original categories column from `df`
    df = df.drop('categories', axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis =1)

    # drop duplicates
    df.drop_duplicates(inplace = True)

    # Remove rows with value 2 in the column related from df
    df = df[df['related'] != 2]

    return df



def save_data(df, database_filename):
    """Saves the data from the dataframe in the table 'DisasterResponseTable' in a sqlite database file

    Arguments:
        df -- Pandas dataframe to save
        database_filename -- Filepath/filename of the sqlite database where the dataframe is saved
    """
    #create sqlite engine for the given databasefile
    engine = create_engine('sqlite:///'+ database_filename)
    #store the date from the dataframe in the databasefile
    df.to_sql('DisasterResponseTable', engine, index = False, if_exists = 'replace')


def main():
    if len(sys.argv) ==4:
        #messages_filepath = 'disaster_messages.csv'
        #categories_filepath = 'disaster_categories.csv'
        #database_filepath =  'DisasterResponse.db'

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