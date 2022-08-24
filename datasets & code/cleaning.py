"""
CSE163 Group Project
Akshita, Bella, and Shey


This file merges and cleans the mutiple and messy datasets
(video_games, apps, demo, editor) into two big datasets that can be used to
solve the resesarch questions mentioned in our report.
"""
import pandas as pd
import numpy as np


def merging(video_games, apps, demo, editor):
    """
    Merges all the CSV files into one dataframe,
    and this dataframe can be used to solve the last two research questions
    mentioned in the report.
    """
    video_games = video_games[['name', 'price of game', 'date release',
                               'publisher of game', 'developer of game',
                               'genres of games', 'features of game']]
    apps = apps[['name', 'genres', 'features', 'price', 'developer',
                 'publisher', 'release_date']]
    demo = demo[['name', 'genres', 'features', 'price', 'Developer',
                 'publisher', 'release date']]
    editor = editor[['name', 'genres', 'features', 'price', 'Developer',
                     'publisher', 'release date']]
    merged_df1 = video_games.merge(apps, left_on='name',
                                   right_on='name', how='outer')
    merged_df2 = merged_df1.merge(demo, left_on='name',
                                  right_on='name', how='outer')
    merged_df = merged_df2.merge(editor, left_on='name',
                                 right_on='name', how='outer')

    filtered_df = merged_df[['name', 'price of game', 'date release',
                             'publisher of game', 'developer of game',
                             'genres of games', 'features of game']]
    df = filtered_df.dropna()
    df = df.copy()

    """
    Cleans the merged dataframe into one that is much clearer,
    and creates some new columns when necessary,
    This dataframe can be used to solve the last two research questions
    mentioned in the report.
    """
    df['price'] = df['price of game'].str[1:]
    df['price'] = df['price'].str.replace(',', '')
    df = df[df['price'] != 'ree'].copy()
    df['price'] = pd.to_numeric(df['price'])
    df['date release'] = df['date release'].str.replace('Release Date', '')
    df['date release'] = df['date release'].str.replace('AvailableQ2, 2022',
                                                        '')
    df = df.copy()

    df["Action Adventure"] = df['genres of games'].str.contains("Action")
    df["Action Adventure"] += df['genres of games'].str.contains("Adventure")
    df["Shooter"] = df['genres of games'].str.contains("Shooter")
    df["Racing"] = df['genres of games'].str.contains("Racing")
    df["Strategy-Survival"] = df['genres of games'].str.contains("Strategy")
    df["Strategy-Survival"] += df['genres of games'].str.contains("Survival")
    df["Horror"] = df['genres of games'].str.contains("Horror")
    df = df.copy()

    df['genre'] = 'Unknown Genre'
    df['genre'] = np.where((df['Action Adventure']).copy(),
                           'Action Adventure', df['genre'])
    df['genre'] = np.where((df['Strategy-Survival']).copy(),
                           'Strategy-Survival', df['genre'])
    df['genre'] = np.where((df['Shooter']).copy(),
                           'Shooter', df['genre'])
    df['genre'] = np.where((df['Racing']).copy(),
                           'Racing', df['genre'])
    df['genre'] = np.where((df['Horror']).copy(),
                           'Horror', df['genre'])

    df['year'] = df['date release'].str[-2:]
    df = df.copy()
    df['year'] = pd.to_numeric(df['year'])
    df = df[(df['year'] >= 18) & (df['year'] <= 22)]
    df['year'] = df['year'] + 2000
    df['year'] = df['year'].astype(int)
    df = df.copy()

    df['date release'] = df['date release'].str.replace('/', '').copy()
    df['dates'] = pd.to_datetime(df['date release'], format='%m%d%y')
    df['dates'].max(), df['dates'].min()
    df['publisher of game'] = df['publisher of game'].str.replace('Publisher',
                                                                  '')
    df['developer of game'] = df['developer of game'].str.replace('Developer',
                                                                  '')
    return df


def cleaning_for_ML(video_games):
    """
    Cleans and creatse new columns based on the dataset video_game,
    which is the one we will be using for the first research question
    (machine learning).
    """
    df = video_games
    df['price'] = df['price of game'].str[1:]
    df['price'] = df['price'].str.replace(',', '')
    df = df[df['price'] != 'ree'].copy()
    df['price'] = pd.to_numeric(df['price'])
    df['date release'] = df['date release'].str.replace('Release Date', '')
    df['date release'] = df['date release'].str.replace('AvailableQ2, 2022',
                                                        '')
    df["Action Adventure"] = df['genres of games'].str.contains("Action")
    df["Action Adventure"] += df['genres of games'].str.contains("Adventure")
    df["Shooter"] = df['genres of games'].str.contains("Shooter")
    df["Racing"] = df['genres of games'].str.contains("Racing")
    df["Strategy-Survival"] = df['genres of games'].str.contains("Strategy")
    df["Strategy-Survival"] += df['genres of games'].str.contains("Survival")
    df["Horror"] = df['genres of games'].str.contains("Horror")
    df['date release'] = df['date release'].str.replace('/', '').copy()
    df['publisher of game'] = df['publisher of game'].str.replace('Publisher',
                                                                  '')
    df['developer of game'] = df['developer of game'].str.replace('Developer',
                                                                  '')
    df['genre'] = 'Unknown Genre'
    df['genre'] = np.where((df['Action Adventure']).copy(),
                           'Action Adventure', df['genre'])
    df['genre'] = np.where((df['Strategy-Survival']).copy(),
                           'Strategy-Survival', df['genre'])
    df['genre'] = np.where((df['Shooter']).copy(),
                           'Shooter', df['genre'])
    df['genre'] = np.where((df['Racing']).copy(),
                           'Racing', df['genre'])
    df['genre'] = np.where((df['Horror']).copy(),
                           'Horror', df['genre'])
    df['year'] = df['date release'].str[-2:]
    df = df.copy()
    df['year'] = pd.to_numeric(df['year'])
    df = df[(df['year'] >= 18) & (df['year'] <= 22)]
    df['year'] = df['year'] + 2000
    df['year'] = df['year'].astype(int)
    df = df.copy()
    df['Affordable'] = np.where(df['price'] > df["price"].median(),
                                False, True)
    df = df.copy()
    df = df[['genre', 'year', 'Top Critic Average', 'Affordable',
            'features of game', 'Critics Recommend', 'price']]
    df['Critics Recommend'] = df['Critics Recommend'].str.replace('%', '')
    df['Critics Recommend'] = pd.to_numeric(df['Critics Recommend'])
    return df
