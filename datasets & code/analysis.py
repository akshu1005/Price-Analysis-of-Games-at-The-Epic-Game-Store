#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE163 Group Project
Akshita, Bella, and Shey


This file performs the summary analysis that can be usead to answer the
second research question. It also makes some graphs for the thrid
research question. Finally, it performs machine learning for the first
research question.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def top_3_game_publishers(df):
    """
    Find out the top three game publishers by the number of game they published
    """
    top_3 = df.groupby(['publisher of game']).size().nlargest(n=3)
    return top_3.index.values


def most_expensive_game_publishers(df):
    """
    Find out who are the top tree games publishers that publish the
    most expensive games
    """
    top_3 = df.groupby(['publisher of game'])['price'].max().nlargest(n=3)
    return top_3.index.values


def average_price_game_publishers(df):
    """
    Find out the average price of each game publisher
    """
    ave_price = df.groupby(['publisher of game'])['price'].mean()
    return ave_price


def genre_bar_plot(df):
    """
    A bar graph that shows how number of games for different genres changed
    from year 2018 to year 2022
    """
    df.groupby(['year'])[['Action Adventure', 'Shooter', 'Racing',
                          'Strategy-Survival', 'Horror']].sum().plot.bar()
    plt.title('How Number of Games for Different Genres Changed over Time')
    plt.xlabel('Year Release')
    plt.ylabel('Number of Game Released')
    plt.legend(title="Game Genres")
    plt.xticks(rotation=360)
    plt.savefig(
        'Bar Plot - How Number of Games for Different Genres Changedover Time')


def genre_line_plot(df):
    """
    A line graph that shows how number of games for different genres changed
    from year 2018 to year 2022
    """
    df.groupby(['year'])[['Action Adventure', 'Shooter',
                          'Racing',
                          'Strategy-Survival', 'Horror']].sum().plot()
    plt.title('How Number of Games for Different Genres Changed over Time')
    plt.xlabel('Year Release')
    plt.ylabel('Number of Game Released')
    plt.legend(title="Game Genres")
    plt.xticks(rotation=360)
    plt.savefig(
        'Line Plot:How Number of Games for Different Genres Changed over Time')


def genre_scatter_plot(df):
    """
    A scatter plot that shows how the price of games for each genre changed
    over year 2018 to year 2022
    """
    df_genre = df[df['genre'] != 'Unknown Genre']
    sns.relplot(kind='scatter', data=df_genre, x='dates',
                y='price', hue='genre')

    plt.title('Trends in Genres of Games over Year 2018 to Year 2022')
    plt.xlabel('Date Release')
    plt.ylabel('Price of Game (₹)')
    plt.legend(title="Game Genres")
    plt.xticks(rotation=-45)
    plt.savefig('Trends in Genres of Games over Year 2018 to Year 2022')


def predict_accord_each_feature(df):
    """
    Machine learning:
    (1) predict if a game is affordable according to each feature using
    LogisticRegression Model; there will be a comparison of accuracy
    between trainsing set and testing set for each feature prediction.
    (2) makes a plot to visualize the comparison of accuracy for different set
    """
    columns = ['genre', 'year', 'Top Critic Average',
               'features of game', 'Critics Recommend']
    scores_test = dict()
    scores_train = dict()
    for column_name in columns:
        features = df.loc[:, column_name]
        features = pd.get_dummies(features)
        labels = df['Affordable']
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.2)
        model = LogisticRegression()
        model.fit(features_train, labels_train)
        test_pred = model.predict(features_test)
        train_pred = model.predict(features_train)
        scores_test[column_name] = accuracy_score(labels_test, test_pred)
        scores_train[column_name] = accuracy_score(labels_train, train_pred)
    return scores_test, scores_train
    fig, ax = plt.subplots(1)
    ax.bar(scores_test.keys(), scores_test.values(), width=0.2,
           align='edge', label='testing')
    ax.bar(scores_train.keys(), scores_train.values(), width=-0.2,
           align='edge', label='training')
    ax.legend(title='Predict Type')
    plt.xticks(rotation=-45)
    plt.title('Accuracy of Predictions for Different Features')
    plt.xlabel('Features')
    plt.ylabel('Accuracy Score')
    plt.savefig('Accuracy of Predictions for Different Features')


def predict_accord_all_features(df):
    """
    Machine learning:
    (1) predict if a game is affordable using all the features using
    LogisticRegression Model; there will be a comparison of accuracy between
    trainsing set and testing set
    (2) makes a plot to visualize the comparison of accuracy for different set
    """
    df = df.dropna()
    features = df.loc[:, ['genre', 'year', 'Top Critic Average',
                          'features of game', 'Critics Recommend']]
    features = pd.get_dummies(features)
    labels = df['Affordable']
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.2)
    model = LogisticRegression()
    model.fit(features_train, labels_train)
    test_pred = model.predict(features_test)
    train_pred = model.predict(features_train)
    test_accuracy = accuracy_score(labels_test, test_pred)
    train_accuracy = accuracy_score(labels_train, train_pred)
    scores = dict()
    scores['test accuracy'] = test_accuracy
    scores['train accuracy'] = train_accuracy
    return scores
