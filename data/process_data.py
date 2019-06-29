# Import the necessary libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

# Define function to load the data
def load_data(messages_filepath, categories_filepath):
    """
    Loads the merged dataset of two different files.

    Args:
        messages_filepath: file path of twitter messages
        categories_filepath: file path of the classification categories

    Output:
        Loads the merged data for pre-processing
    """

    # Load the messages csv file
    messages = pd.read_csv(messages_filepath)

    # Load the categories csv file
    categories = pd.read_csv(categories_filepath)

    # Merge the both the files
    df = pd.DataFrame(messages.merge(categories, how = 'outer', on = 'id'))

    return df


def clean_data(df):
    """
    Args:
        df: Merged dataset(the output) from the load_data function

    Output:
        Cleaned dataset
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)

    # select the first row of the categories dataframe
    row = categories.iloc[0, :]

    # Extract new column names for categories
    category_colnames = row.apply(lambda x: x.split('-')[0])

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)

    # drop duplicates
    df = df.drop_duplicates(subset = 'id', keep = 'first')

    return df


def save_data(df, database_filepath):
    """
    Saves the data in the sql database

    Args:
        df: cleaned dataset
        database_filename: filename of the databased to be saved
    """

    # Create a connection
    engine = create_engine('sqlite:///' + database_filepath)
    print(type(df))
    df.to_sql(name = 'messages_disaster', con = engine, index=False)
    #data.to_sql(name='sample_table2', con=engine, if_exists = 'append', index=False)




def main():
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
