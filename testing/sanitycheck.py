#!/usr/bin/env python

# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
#
# Usage:
#   python sanity_check.py --help
######################################################################
import inspect
import os
import sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from chatbot import Chatbot

import argparse
import numpy as np
import math


def assertNumpyArrayEquals(givenValue, correctValue, failureMessage):
    try:
        assert np.array_equal(givenValue, correctValue)
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def assertListEquals(givenValue, correctValue, failureMessage, orderMatters=True):
    try:
        if orderMatters:
            assert givenValue == correctValue
            return True
        givenValueSet = set(givenValue)
        correctValueSet = set(correctValue)
        assert givenValueSet == correctValueSet
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def assertEquals(givenValue, correctValue, failureMessage):
    try:
        assert givenValue == correctValue
        return True
    except AssertionError:
        print(failureMessage)
        print("Expected: {}".format(correctValue))
        print("Actual: {}".format(givenValue))
        return False


def test_similarity():
    print("Testing similarity() functionality...")
    chatbot = Chatbot(False)

    x = np.array([1, 1, -1, 0], dtype=float)
    y = np.array([1, 0, 1, -1], dtype=float)

    self_similarity = chatbot.similarity(x, x)
    if not math.isclose(self_similarity, 1.0):
        print('Unexpected cosine similarity between {} and itself'.format(x))
        print('Expected 1.0, calculated {}'.format(self_similarity))
        print()
        return False

    ortho_similarity = chatbot.similarity(x, y)
    if not math.isclose(ortho_similarity, 0.0):
        print('Unexpected cosine similarity between {} and {}'.format(x, y))
        print('Expected 0.0, calculated {}'.format(ortho_similarity))
        print()
        return False

    print('similarity() sanity check passed!')
    print()
    return True


def test_binarize():
    print("Testing binarize() functionality...")
    chatbot = Chatbot(False)
    if assertNumpyArrayEquals(
            chatbot.binarize(np.array([[1, 2.5, 5, 0]])),
            np.array([[-1., -1., 1., 0.]]),
            "Incorrect output for binarize(np.array([[1, 2.5, 5, 0]]))."
    ):
        print("binarize() sanity check passed!")
    print()


def test_extract_titles():
    print("Testing extract_titles() functionality...")
    chatbot = Chatbot(False)

    # add more test cases here!!!
    test_cases = [
        # ('I liked "The Notebook"', ["The Notebook"]),
        # ('You are a great bot!', []),
        # ('I enjoyed "Titanic (1997)" and "Scream 2 (1997)"', ["Titanic (1997)", "Scream 2 (1997)"]),
        # ('I thought the titanic was really good!', ['the titanic']),
        # # ('i liked scream and tHe Notebook', ['scream', 'the notebook']),
        # ('transformers was a great movie', ['transformers']),
        # ('my favorite movie is interstellar', ['interstellar']),
        # # ('i enjoyed movie 1, movie2, and movie3', ['movie 1', 'movie2', 'movie3']),
        # ('the actress in snow white and the seven dwarves is so talented', ['snow white and the seven dwarves']),
        # ('the beginning of transformers was stunning', ['transformers']),
        # ('i liked the movie transformers', ['transformers']),
        # ('chris hemsworth rocked it in thor', ['thor']),
        # ('liam hemsworth is so hot in the hunger games', ['the hunger games']),
        # # ('parasite was a really interesting movie and it deserved to win so many awards', ['parasite']),
        # # ('i was too scared to watch the movie get out', ['get out']),
        # # ('my favorite movie ever was avatar', ['avatar']),
        # # ('i just saw to all the boys i\'ve loved before 2 and it was worse than the first one', ['to all the boys i\'ve loved before 2']),
        # ('the best movie of all time is one day in rinconada', ['one day in rinconada']),
        # ('the worst movie i have ever seen is a day in the life of lyron', ['a day in the life of lyron']),
        # ('action movies like james bond are my favorite kind', ['james bond']),
        # ('i really identify with bobby in the great escape', ['the great escape']),
        # ('a christmas story made me feel warm and fuzzy inside', ['a christmas story']),
        # # ('home alone always makes me laugh', ['home alone']),
        # ('last weekend i watched knives out and a marriage story', ['knieves out', 'a marriage story']),
        # ('avenger\'s end game is playing in theaters now and i really want to watch it', ['avenger\'s end game']),
        # ('music in frozen is sooooo good', ['frozen']),
        # ('i want to see ad astra', ['ad astra']),
        # # ('joker and jo jo rabbit really hit the spot, but the movie booksmart missed the mark', ['joker', 'jo jo rabbit', 'booksmart']),
        # ('in my opinion, the tom brady documentary was the goat', ['tom brady documentary']),
        # ('i thought the notebook was fantastic', ['the notebook']),
        # ('i loved jungle book', ['jungle book']),
        # ('I liked The NoTeBoOk!', ["the notebook"]),
        # ('I thought 10 things i hate about you was great', ['10 things i hate about you']),
        # ('i really liked the movie the half-blood prince', ['the half-blood prince']),
        # ('the goblet of fire', ['the goblet of fire'])
        ('I liked "Transformers: The Movie (1986)"', ['Transformers: The Movie (1986)'])
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assertListEquals(
            chatbot.extract_titles(chatbot.preprocess(input_text)),
            expected_output,
            "Incorrect output for extract_titles(chatbot.preprocess('{}')).".format(input_text),
            orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('extract_titles() sanity check passed!')
    print()


def test_find_movies_by_title():
    print("Testing find_movies_by_title() functionality...")
    chatbot = Chatbot(False)

    # add more test cases here!!!
    test_cases = [
        ('The American President', [10]),
        ('Titanic', [1359, 2716]),
        ('Titanic (1997)', [1359]),
        ('An American in Paris (1951)', [721]),
        ('The Notebook (1220)', []),
        ('Scream', [1142]),
        ("10 things i HATE about you", [2063]),
        ('chamber Of secrets', [4325]),
        ("La Guerre du feu", [2439]),
        ('harRy pOTter and the half-blood prince', [7274]),
        ('TwiN DraGonS', [2071]),
        ('eXiStenZ',[2083]),
        ('XIu XIu: thE SeNt-down girl', [2097]),
        ('EternIty and A Day', [2140]),
        ('Mia aoniotita kai mia mera', [2140]),
        ('Vie des anges',[1998]),
        ('Los amantes polar',[2073]),
        ('Wild WiLD wesT',[2161]),
        ('conte d\'automne', [2166]),
        ('The Blair Witch Project', 2168),
        ('Transformers: The Movie (1986)', 3208)
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assertListEquals(
            chatbot.find_movies_by_title(input_text),
            expected_output,
            "Incorrect output for find_movies_by_title('{}').".format(input_text),
            orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('find_movies_by_title() sanity check passed!')
    print()


def test_extract_sentiment():
    print("Testing extract_sentiment() functionality...")
    chatbot = Chatbot(False)

    # add more test cases here!!!
    test_cases = [
        # ('I like "Titanic (1997)".', 1),
        # ('I saw "Titanic (1997)".', 0),
        # ('I didn\'t enjoy "Titanic (1997)".', -1),
        # ('I didn\'t really like "Titanic (1997)".', -1),
        # ('I never liked "Titanic (1997)".', -1),
        # ('I really enjoyed "Titanic (1997)".', 1),
        # ('"Titanic (1997)" started out terrible, but the ending was totally great and I loved it!', 1),
        # ('I loved "10 Things I Hate About You"', 1),
        ('I loved "Zootopia"', 2),
        ('"Zootopia" was terrible', -2),
        ('I really reeally liked "Zootopia"', 2),
        ('"Zootopia" was good', 1),
        ('I really didn\'t like "Zootopia"', -2),
        ('I hated Zootopia"', -2),
        ('"Zootopia" is a movie', 0),
        ('"Zootopia" was the worst movie ever made', -2),
        ('"Zootopia" wasn\'t great', -1),
        ('"Zootopia" is fine', 0),
        ('"Zootopia" is okay', 0),
        ('"Zootopia" was not bad', 1),
        ('"Zootopia" was fantastic', 2),
        ('The movie "Zootopia" is a cinematic masterpiece', 2),
        ('"Summer of Sam" is my favorite movie', 2),
        ('I can not think of a better movie than "Ghostbusters"', 2),
        ('"Eyes wide shut" is a decent movie', 1),
        ('"Velocity of Gary" is worth a watch', 1),
        ('"Lake Placid" is a fine movie', 0),
        ('"Zootopia" is the worst movie ever', -2),
        ('I did not like "The killing"', -1),
        ('"Mystery Men" was decent', -1)
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assertEquals(
            chatbot.extract_sentiment(chatbot.preprocess(input_text)),
            expected_output,
            "Incorrect output for extract_sentiment(chatbot.preprocess('{}')).".format(input_text)
        ):
            tests_passed = False
    if tests_passed:
        print('extract_sentiment() sanity check passed!')
    print()


def test_extract_sentiment_for_movies():
    print("Testing test_extract_sentiment_for_movies() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('I liked both "I, Robot" and "Ex Machina".', [("I, Robot", 1), ("Ex Machina", 1)]),
        ('I liked "I, Robot" but not "Ex Machina".', [("I, Robot", 1), ("Ex Machina", -1)]),
        ('I didn\'t like either "I, Robot" or "Ex Machina".', [("I, Robot", -1), ("Ex Machina", -1)]),
        ('I liked "Titanic (1997)", but "Ex Machina" was not good.', [("Titanic (1997)", 1), ("Ex Machina", -1)]),
    ]

    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assertListEquals(
            chatbot.extract_sentiment_for_movies(chatbot.preprocess(input_text)),
            expected_output,
            "Incorrect output for extract_sentiment_for_movies(chatbot.preprocess('{}')).".format(input_text),
            orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('extract_sentiment_for_movies() sanity check passed!')
    print()


def test_find_movies_closest_to_title():
    print("Testing find_movies_closest_to_title() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('Sleeping Beaty', [1656]),
        ('Te', [8082, 4511, 1664]),
        ('BAT-MAAAN', [524, 5743]),
        ('Blargdeblargh', []),
        ('Twin Falls Idjho', [2182]),
        ('Toy stry',[0]),
        ('Sense and Sisnesbility', []),
        ('Alien Secape', [1344])
    ]
 
    tests_passed = True
    for input_text, expected_output in test_cases:
        if not assertListEquals(
            chatbot.find_movies_closest_to_title(input_text),
            expected_output,
            "Incorrect output for find_movies_closest_to_title(chatbot.preprocess('{}')).".format(input_text),
            orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('find_movies_closest_to_title() sanity check passed!')
    print()
    return True


def test_disambiguate():
    print("Testing disambiguate() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('1997', [1359, 2716], [1359]),
        ('2', [1142, 1357, 2629, 546], [1357]),
        ('Sorcerer\'s Stone', [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842], [3812]),
    ]

    tests_passed = True
    for clarification, candidates, expected_output in test_cases:
        if not assertListEquals(
            chatbot.disambiguate(clarification, candidates),
            expected_output,
            "Incorrect output for disambiguate('{}', {})".format(clarification, candidates),
            orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('disambiguate() sanity check passed!')
    print()
    return True


def test_disambiguate_complex():
    print("Testing complex disambiguate() functionality...")
    chatbot = Chatbot(True)

    # add more test cases here!!!
    test_cases = [
        ('2', [8082, 4511, 1664], [4511]),
        ('most recent', [524, 5743], [524]),
        ('the Goblet of Fire one', [3812, 4325, 5399, 6294, 6735, 7274, 7670, 7842], [6294]),
        ('the second one', [3812, 6294, 4325, 5399, 6735, 7274, 7670, 7842], [6294]),
        ('the original', [1359, 2716],[2716]),
        ('the first', [3460, 4614, 6458, 7203, 7277, 7799, 8325], [3460]),
        ('the third', [3460, 4614, 6458, 7203, 7277, 7799, 8325], [6458]),
        ('the latest', [3460, 4614, 6458, 7203, 7277, 7799, 8325], [8325]),

    ]

    tests_passed = True
    for clarification, candidates, expected_output in test_cases:
        if not assertListEquals(
            chatbot.disambiguate(clarification, candidates),
            expected_output,
            "Incorrect output for complex disambiguate('{}', {})".format(clarification, candidates),
            orderMatters=False
        ):
            tests_passed = False
    if tests_passed:
        print('complex disambiguate() sanity check passed!')
    print()
    return True


def test_recommend():
    print("Testing recommend() functionality...")
    chatbot = Chatbot(False)

    user_ratings = np.array([1, -1, 0, 0, 0, 0])
    all_ratings = np.array([
        [1, 1, 1, 0],
        [1, -1, 0, -1],
        [1, 1, 1, 0],
        [0, 1, 1, -1],
        [0, -1, 1, -1],
        [-1, -1, -1, 0],
    ])
    small_recommendations = chatbot.recommend(user_ratings, all_ratings, 2)
    user_ratings = np.zeros(9125)
    user_ratings[[8514, 7953, 6979, 7890]] = 1
    user_ratings[[7369, 8726]] = -1
    recommendations = chatbot.recommend(user_ratings, chatbot.ratings, k=5)

    test_cases = [
        (small_recommendations, [2, 3]),
        (recommendations, [8582, 8596, 8786, 8309, 8637]),
    ]

    tests_passed = True
    for i, (recs, expected_output) in enumerate(test_cases):
        if not assertListEquals(
            recs,
            expected_output,
            "Test case #{} for recommender tests failed".format(i),
        ):
            tests_passed = False
    if tests_passed:
        print('recommend() sanity check passed!')
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Sanity checks the chatbot. If no arguments are passed, all checks for starter mode are run; you can use the '
                    'arguments below to test specific parts of the functionality.')

    parser.add_argument('-a', '--all', help='Tests all of the functions', action='store_true')
    parser.add_argument('-c', '--creative', help='Tests all of the creative function', action='store_true')
    parser.add_argument('--extract-titles', help='Tests only the extract_titles function', action='store_true')
    parser.add_argument('--find-movies', help='Tests only the find_movies_by_title function', action='store_true')
    parser.add_argument('--extract-sentiment', help='Tests only the extract_sentiment function', action='store_true')
    parser.add_argument('--recommend', help='Tests only the recommend function', action='store_true')
    parser.add_argument('--binarize', help='Tests only the binarize function', action='store_true')
    parser.add_argument('--similarity', help='Tests only the similarity function', action='store_true')
    parser.add_argument('--find-closest', help='Tests only the find_movies_closest_to_title function', action='store_true')
    parser.add_argument('--extract-sentiment-multiple', help='Tests only the extract_sentiment_for_movies function', action='store_true')
    parser.add_argument('--disambiguate', help='Tests only the disambiguate functions (for part 2 and 3)', action='store_true')

    args = parser.parse_args()
    if args.extract_titles:
        test_extract_titles()
        return
    if args.find_movies:
        test_find_movies_by_title()
        return
    if args.extract_sentiment:
        test_extract_sentiment()
        return
    if args.recommend:
        test_recommend()
        return
    if args.binarize:
        test_binarize()
        return
    if args.similarity:
        test_similarity()
        return
    if args.find_closest:
        test_find_movies_closest_to_title()
        return
    if args.extract_sentiment_multiple:
        test_extract_sentiment_for_movies()
        return
    if args.disambiguate:
        test_disambiguate()
        test_disambiguate_complex()
        return

    testing_creative = args.creative
    testing_all = args.all

    if not testing_creative or testing_all:
        test_extract_titles()
        test_find_movies_by_title()
        test_extract_sentiment()
        # comment out test_recommend() if it's taking too long!
        test_recommend()
        test_binarize()
        test_similarity()

    if testing_creative or testing_all:
        test_find_movies_closest_to_title()
        test_extract_sentiment_for_movies()
        test_disambiguate()
        test_disambiguate_complex()


if __name__ == '__main__':
    main()