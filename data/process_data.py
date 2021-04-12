import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """
    Reads in two .csv files (messages & categories) and outputs one merged pandas dataframe

    INPUTS:
        messages_filepath: filepath to .csv file containing messages
        categories_filepath: filepath to .csv file containing categories

    OUTPUT:
        df: merged pandas dataframe
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    """
    Reads in pandas dataframe and pre-processes category data to prepare it for saving in SQLite database

    INPUT:
        df: pandas dataframe

    OUTPUT:
        df: cleaned pandas dataframe without duplicates
    """
    # create a dataframe of the category columns
    categories = df.categories.str.split(pat=';', expand=True)

    # select first row of the categories dataframe
    row = categories.iloc[0, :]

    # extract list of new column names for categories
    category_colnames = row.apply(lambda x: x[:-2])

    # rename columns
    categories.columns = category_colnames

    # convert category values to 0 or 1
    for column in categories:
        # set value as last character
        categories[column] = categories[column].str[-1]
        # convert from string to integer
        categories[column] = categories[column].astype(np.int64)

    # drop original categories column from df
    df.drop('categories', axis=1, inplace=True)

    # concatenate original dataframe with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    # drop ["child_alone"] column as it has only zeros as values
    df = df.drop(["child_alone"], axis=1)

    # replace assumed wrong entries in ["related"] column with most common value, i.e. 1
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    return df


def save_data(df, database_filepath):
    """
    Saves dataframe to SQLite database

    INPUTS:
        df: pandas dataframe
        database_filepath: database filepath

    OUTPUT:
        None
    """
    # create SQLAlchemy engine
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    # save dataframe to SQLite database; replace if already exists
    df.to_sql('df', engine, index=False, if_exists='replace')


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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'Disaster.db')


if __name__ == '__main__':
    main()