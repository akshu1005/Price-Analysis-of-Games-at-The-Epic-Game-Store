#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE163 Group Project
Akshita, Bella, and Shey

This file uses the main pattern and calls the method to present the
results for each research question. It also calls the test functinos to check
if the result we get are reasonable.
"""
import pandas as pd
import cleaning
import analysis
from cse163_utils import assert_equals


def research_question_2(merged):
    """
    Return the answers of research question 2
    """
    top_3 = analysis.top_3_game_publishers(merged)
    expensive = analysis.most_expensive_game_publishers(merged)
    average = analysis.average_price_game_publishers(merged)
    return top_3, expensive, average


def research_question_3(merged):
    """
    Return the graphs that can be used to answer research question 3
    """
    analysis.genre_bar_plot(merged)
    analysis.genre_line_plot(merged)
    analysis.genre_scatter_plot(merged)


def research_question_1(ml_data):
    """
    Return the answers and graphs of research question 1
    """
    analysis.predict_accord_each_feature(ml_data)
    all_features_predict = analysis.predict_accord_all_features(ml_data)
    return all_features_predict


def test_question_2(merged):
    """
    test research question 2 by using less data, which means to
    get a part of the data (sample data of 10 rows) from the large dataframe
    we used and see if it still gets the correct answer using the code we
    provided
    """
    assert_equals(['11 Bit Studios', '2K', 'Bethesda Softworks'],
                  analysis.top_3_game_publishers(merged))
    assert_equals(['Bethesda Softworks', '2K', 'Texel Raptor'],
                  analysis.most_expensive_game_publishers(merged))
    x = analysis.average_price_game_publishers(merged)
    assert_equals(['11 Bit Studios', '2K', 'Bethesda Softworks',
                   'Big Fish Games', 'HandyGames',
                   'META Publishing', 'Mixtvision', 'Super.com', 'THQ Nordic',
                   'Texel Raptor'],
                  x.index.values)
    assert_equals([521.0, 750.0, 999.0, 349.0, 529.0, 239.0, 459.0, 259.0,
                   499.0, 699.0],
                  x)


def test_question_3(merged):
    """
    test research question 3 by using less data, which means to
    get a part of the data (sample data of 100 rows) from the large dataframe
    we used and see if it still gets the correct graph using the code we
    provided
    """
    analysis.genre_bar_plot(merged)
    analysis.genre_line_plot(merged)
    analysis.genre_scatter_plot(merged)


def test_question_1(ml_data):
    """
    test research question 1 by using less data, which means to
    get a part of the data (sample data) from the large dataframe we used
    and see if it still gets the correct result using the code we provided
    """
    analysis.predict_accord_each_feature(ml_data)


def main():
    """
    Read in the datasets and call functions
    """

    # original datasets
    video_games = pd.read_csv('epic games store-video games.csv')
    apps = pd.read_csv('epic-games-apps.csv')
    demo = pd.read_csv('epic-games-demo.csv')
    editor = pd.read_csv('epic_games_editor.csv')

    # merged datasets used for research questions
    merged = cleaning.merging(video_games, apps, demo, editor)
    ml_data = cleaning.cleaning_for_ML(video_games)

    # report result of research questions
    print(research_question_1(ml_data))
    print(research_question_2(merged))
    research_question_3(merged)

    # datasets used for test functions
    test_data = merged.sample(n=10, random_state=1)
    test_data_graph = merged.sample(n=100, random_state=1)
    test_data_ml = cleaning.cleaning_for_ML(video_games.sample(n=100,
                                                               random_state=1))

    # report result of test functinos
    test_question_2(test_data)
    test_question_3(test_data_graph)
    test_question_1(test_data_ml)


if __name__ == '__main__':
    main()
