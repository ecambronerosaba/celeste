# PA6, CS124, Stanford, Winter 2019
# v.1.0.3
# Original Python code by Ignacio Cases (@cases)
######################################################################

import movielens
import numpy as np
import re
import random

import nltk
from nltk.metrics import edit_distance
from nltk.tokenize import word_tokenize

from PorterStemmer import PorterStemmer

POS_RESPONSES = [('Wow! I can see that you liked ', '. I thought the actors/actresses were superb! What\'re some other movies'), 
                 ('I absolutely loved ',' too! Tell me more!'),
                 ('I would definitely recommend ',' as well! Tell me about other movies you\'ve seen'),
                 ('Oh I thought the plot was super interesting in ',' too. Any other recommendations?'),
                 ('Agreed! I thought that ',' did a great job conveying an important message. What other movies stuck out to you?'),
                 ('Well... I personally didn\'t like ','. Glad You enjoyed it though! What do you think about other movies?'),
                 ('Wow! ',' is/are on my list of favorites for sure! What other movies have you seen?')
                ]

NEG_RESPONSES = [('Hmmm I see you didn\'t like ', '. We\'ll make sure to find you better watching material! What about some other movies you\'ve seen?'),
                 ('It seems like you didn\'t like ', '. Well that\'s okay. Let\'s talk about more movies.'),
                 ('', ' is kind of my favorite movie... but let\'s just agree to disagree! Let\'s keep going!'),
                 ('Yeah I agree that ', ' just missed the mark. How about we talk about some other movie?'),
                 ('You don\'t seem very enthusiastic about ', '. Honestly, I can\'t blame you. What about a different movie you\'ve seen.'), 
                 ]

ARBITRARY_RESPONSES = ["I'm not quite sure I understand. Could you tell me about another movie?",
                       "That was unclear to me. Let's talk about another movie.",
                       "It seems like you want to talk about something else. Why don't you tell me about a different movie?",
                       "We should talk about some other movie!"
                      ]

CAN_DO = "My specialty is movie recommendations so feel free to talk to me about any movie and we'll see what I think you'll like. Tell me about a movie you've seen."

RECOMMEND_RESPONSES = ["I think you'd like ", 'It seems to me that you would enjoy ', 
                        "Based on what you've told me, I think you would be pleased watching ",
                        "I think you would love ",
                        "If you haven't seen it, you should definitely watch ",
                        "In my expert opinion you would thoroughly enjoy ",
                        "You seem like the kind of person that would love ",
                        "Maybe this is out of the blue, but I think you'd really have a good time watching ",
                        "You should really watch ",
                        "I think you'd really like ",
                        "Definitely see ",
                        "Take the time to watch ",
                        "From what we've talked about I can't believe you haven't seen ",
                        "If you want to have a life changing experience, you need to watch ",
                        "You're the kind of person that would get a kick out of "
                        ]

NEG_WORDS = ['no', 'cannot', 'not', 'never', 'nothing', 'neither', 'nobody', 
             'none', 'nor', 'nowhere', 'rarely', 'hardly', 'cannot', 'seldom',
             'despite', 'without']
STOP_PUNC = ['.', ';', '?', '!']

AMPLIFY_WORDS = ['very', 'absolutely', 'amazingly', 'really', 'awfully', 
                 'completely', 'considerably', 'considerable', 'enormously',
                 'deeply', 'decidedly', 'especially', 'exceptionally', 
                 'extremeley', 'entirely', 'greatly', 'highly', 'incredibly',
                 'incredible', 'intensely', 'particularly', 'remarkably', 
                 'so', 'substantially', 'totally', 'thoroughly', 'tremendously',
                 'tremendous', 'utterly', 'utter', 'unbelievably', 'exceedingly',
                 'excessively', 'extraordinarily', 'acutely']

title_regexes = [
    r"([\w\s\d\-:&]*)",
    r"watch\s([\w\s\d\-:&]*)(?:\.|\?|!|\n|\sand|\sor|\sbut)",
    r"\s(?:like|hate|love|dislike|adore|enjoy)\s([\w\s\d\-:&]*)(?:\.|\?|!|\n|\sand|\sor|\sbut|\sis)",
    r"\s(?:thought|think|that|in|with|believe|suppose|of)\s([\w\s\d&]*)(?:\swas|\sis)",
    r"([\w\s\d\-:&]*)\s(?:was|is)",
    r"(?:[\w\s]+ed)\s([\w\s\d\-:&]+)(?:\.|\?|!|\n|\sand|\sor|\sbut)",
    r"movie is\s([\w\s\d\-:&]*)[\w\s]*(?:\.|\?|!|\n|\sand|\sor|\sbut)",
    r"movie\s([\w\s\d\-:&]*)[\w\s]*(?:\.|\?|!|\n|\sand|\sor|\sbut|$)",
    r"see ([\w\s\d\-:&]+)(?:\.|\?|!|\n|\sand|\sor|\sbut|$)",
    r"in ([\w\s\d\-:&]+)(?:\.|\?|!|\n|\sand|\sor|\sbut|$|\swas)",
]


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`. Give your chatbot a new name.
        self.name = 'Celeste'

        self.creative = creative
        # self.creative = True

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = movielens.ratings()
        self.sentiment = movielens.sentiment()

        self.vader_lexicon = self.load_vader_lexicon('deps/vader_lexicon_trimmed.txt')

        #############################################################################
        # TODO: Binarize the movie ratings matrix.                                  #
        #############################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = ratings
        # Track how many movies the user has gave opinions on
        self.count = 0
        self.i = 0
        self.user_ratings = np.zeros(9125)
        self.reachedFive = False
        self.recList = []
        self.goodToGo = False
        self.finalIndexes = []
        self.reference = ["it", "the movie", "that movie", "that film"]
        self.currTitle = ""
        self.reference2 = ["but", "and"]
        self.currSentiment = ""
        self.disambiguateMode = False
        self.common = ["No", "no", "Yes", "yes", "Of course", "Sure", "sure", "y", "Y", "n", "N", "Cool"]
        self.spellFlag = False

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    #############################################################################
    # 1. WARM UP REPL                                                           #
    #############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        #############################################################################
        # TODO: Write a short greeting message                                      #
        #############################################################################

        greeting_message = "Hi my name's Celeste! If you're ready to find your next favorite movie, tell me about some movie you've seen."

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return greeting_message

    def goodbye(self):
        """Return a message that the chatbot uses to bid farewell to the user."""
        #############################################################################
        # TODO: Write a short farewell message                                      #
        #############################################################################

        goodbye_message = "Until we meet again ;)"

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return goodbye_message

    ###############################################################################
    # 2. Modules 2 and 3: extraction and transformation                           #
    ###############################################################################

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        #############################################################################
        # TODO: Implement the extraction and transformation in this method,         #
        # possibly calling other functions. Although modular code is not graded,    #
        # it is highly recommended.                                                 #
        #############################################################################
        if self.creative:
            if line.count("\"") % 2 != 0:
                response = "Looks like there's a weird number of quotes. Could you go back and fix that for me?"
            else:
                if self.reachedFive == False:
                    if self.disambiguateMode == True:
                        disambiguateList = []
                        for i in self.find_movies_by_title(self.currTitle):
                            disambiguateList.append(i)
                        if len(self.disambiguate(line, disambiguateList)) >= 1:
                            title = self.titles[self.disambiguate(line, disambiguateList)[0]][0]
                            self.currTitle = title
                            if self.currSentiment == 1:
                                line = "I liked " + '"' + title + '"'
                            elif self.currSentiment == -1:
                                line = "I didn't like " + '"' + title + '"'
                            elif self.currSentiment == 0:
                                response = "I'm not quite sure about whether you liked or disliked the movie(s). Could you be a bit more explicit?"
                            self.goodToGo = True
                            self.disambiguateMode = False
                        else:
                            response = "Sorry I didn't quite get that. Why don't we talk about another movie."
                            self.disambiguateMode = False
                    else:
                        response = "What are some of your opinions on movies?"
                        titles = self.extract_titles(line, use_starter=True)
                        if line in self.common and self.spellFlag != True:
                            rand = random.choice(ARBITRARY_RESPONSES)
                            response = rand
                        elif self.spellFlag == True:
                            if line == "Yes" or line == 'yes':
                                if self.currSentiment == -1: 
                                        line = "I didn't like " + '"' + self.currTitle + '"'
                                if self.currSentiment == 1:
                                    line = "I liked " + '"' + self.currTitle + '"'
                                self.goodToGo = True
                                self.spellFlag = False
                            else:
                                rand = random.choice(ARBITRARY_RESPONSES)
                                response = "Ok, let's try again. " + rand
                                self.spellFlag = False
                        elif len(titles) < 1:
                            check = False
                            for x in self.reference:
                                if x in line: 
                                    line = line.replace(x, '"' + self.currTitle + '"', 1)
                                    check = True
                                    self.goodToGo = True
                            if check == False:
                                rand = random.choice(ARBITRARY_RESPONSES)
                                response = rand
                        elif len(titles) == 1:
                            title = titles[0]
                            self.currTitle = title
                            if len(self.find_movies_by_title(title)) == 0:
                                spellCheckedTitles = self.find_movies_closest_to_title(title, max_distance=3)
                                if len(spellCheckedTitles) == 1:
                                    self.spellFlag = True
                                    response = "Did you mean " + self.titles[spellCheckedTitles[0]][0]
                                    self.currTitle = self.titles[spellCheckedTitles[0]][0]
                                    self.currSentiment = self.extract_sentiment(line, use_starter=True)
                                else:
                                    response = "I don't recognize one movie you mentioned, could you try again"
                            elif len(self.find_movies_by_title(title)) > 1:
                                titleList = ""
                                for i in (self.find_movies_by_title(title)):
                                    titleList += self.titles[i][0] + " | "
                                response = "Which one did you mean? Specify one from the following:" + titleList
                                self.currSentiment = self.extract_sentiment(line, use_starter=True)
                                self.disambiguateMode = True
                            elif self.extract_sentiment(line, use_starter=True) == 0:
                                if "But" in line or "but" in line:
                                    if self.currSentiment == 1: 
                                        line = "I didn't like " + '"' + title + '"'
                                    if self.currSentiment == -1:
                                        line = "I liked " + '"' + title + '"'
                                    self.goodToGo = True
                                elif "and" in line or "And" in line:
                                    if self.currSentiment == 1: 
                                        line = "I liked " + '"' + title + '"'
                                    if self.currSentiment == -1:
                                        line = "I didn't like " + '"' + title + '"'
                                    self.goodToGo = True
                                else:
                                    response = "I'm not quite sure about whether you liked or disliked the movie(s). Could you be a bit more explicit?"
                            else:
                                self.goodToGo = True
                        elif len(titles) > 1:
                            for title in titles:
                                self.currTitle = title
                                if len(self.find_movies_by_title(title)) == 0:
                                    spellCheckedTitles = self.find_movies_closest_to_title(title, max_distance=3)
                                    if len(spellCheckedTitles) == 1:
                                        self.spellFlag = True
                                        response = "Did you mean " + self.titles[spellCheckedTitles[0]][0]
                                        self.currTitle = self.titles[spellCheckedTitles[0]][0]
                                        self.currSentiment = self.extract_sentiment(line, use_starter=True)
                                    else:
                                        response = "I don't recognize the movie you mentioned. Could you try a different one for me?"
                                elif len(self.find_movies_by_title(title)) > 1:
                                    titleList = ""
                                    for i in (self.find_movies_by_title(title)):
                                        print(self.titles[i])
                                        titleList += self.titles[i][0] + " | "
                                    response = "Which one did you mean? Specify one from the folowing:" + titleList
                                    self.currSentiment = self.extract_sentiment(line, use_starter=True)
                                    self.disambiguateMode = True
                                elif self.extract_sentiment(line, use_starter=True) == 0:
                                    response = "I'm not quite sure about whether you liked or disliked the movie(s). Could you be a bit more explicit?"
                                else:
                                    self.goodToGo = True
                    if self.goodToGo == True:
                        self.disambiguateMode = False
                        titleSentiments = self.extract_sentiment_for_movies(line)
                        posTitle = ""
                        negTitle = ""
                        posCount, negCount = 0, 0
                        for titleSentimentPair in titleSentiments:
                            title = titleSentimentPair[0]
                            index = self.find_movies_by_title(title)
                            score = titleSentimentPair[1]
                            self.currSentiment = score
                            self.user_ratings[index] = score
                            self.count += 1
                            if score == 1:
                                posCount += 1
                                if posCount <= 1:
                                    posTitle = title
                                else:
                                    posTitle += (" and " + title)
                            if score == -1:
                                negCount += 1
                                if negCount <= 1:
                                    negTitle = title
                                else:
                                    negTitle += (" and " + title)
                        if self.count <= 5 or self.count%5 != 0:
                            self.reachedFive = self.check_five_data_points()
                            if negTitle != "":
                                rand = random.choice(NEG_RESPONSES)
                                response = rand[0] + negTitle + rand[1]
                            if posTitle != "": 
                                rand = random.choice(POS_RESPONSES)
                                response = rand[0] + posTitle + rand[1]
                            if negTitle != "" and posTitle != "":
                                response = "Got it, so you did not like " + negTitle + " but you liked " + posTitle + ". Can you share more?" 
                        if self.count != 0 and self.count%5 == 0:
                            self.reachedFive = self.check_five_data_points()
                            self.recList = self.recommend(self.user_ratings, self.ratings, k=10, creative=False)
                        self.goodToGo = False
                if self.reachedFive == True:
                    if self.i < 10:
                        if line == "No" or line == "no":
                            response = "If you want to continue, share your opinion on another movie(s). If not, enter :quit."
                            self.reachedFive = False
                        else:
                            recTitle = self.titles[self.recList[self.i]][0]
                            rand = random.choice(RECOMMEND_RESPONSES)
                            response = rand + recTitle +". Would you like to hear another one? Enter Yes or No"
                            self.i += 1
                    else:
                        response = "That's all I've got for you:) If you want to continue, share your opinion on another movie. If not, enter quit."   
                        self.i = 0
                        self.reachedFive = self.check_five_data_points()
        else:
            if line.count("\"") % 2 != 0:
                response = "Looks like there's a weird number of quotes."
            else:
                if self.reachedFive == False:
                    response = "Could you tell me about your opinions of five movies, one at a time?"
                    if len(self.extract_titles(line, use_starter=True)) > 1:
                        response = "Could you tell me about your opinions of five movies, one at a time?"
                    if len(self.extract_titles(line, use_starter=True)) == 1:
                        title = self.extract_titles(line, use_starter=True)[0]
                        if len(self.find_movies_by_title(title)) == 0:
                            response = "I don't recognize the movie you mentioned, could you share your opinion on another movie"
                        elif len(self.find_movies_by_title(title)) > 1:
                            response = "There are multiple versions of this movie, could you specify?"
                        elif self.extract_sentiment(line, use_starter=True) == 0:
                            response = " I'm sorry, I'm not quite sure if you liked " + title + " or not. Could you tell me more about it?"
                        else:
                            score = self.extract_sentiment(line, use_starter=True)
                            index = self.find_movies_by_title(title)
                            self.user_ratings[index] = score
                            self.count += 1
                            if self.count <= 5 or self.count%5 != 0:
                                self.reachedFive = self.check_five_data_points()
                                if score == -1: 
                                    rand = random.choice(NEG_RESPONSES)
                                    response = rand[0] + title + rand[1]
                                if score == 1: 
                                    rand = random.choice(POS_RESPONSES)
                                    response = rand[0] + title + rand[1]
                            if self.count != 0 and self.count%5 == 0:
                                self.reachedFive = self.check_five_data_points()
                                self.recList = self.recommend(self.user_ratings, self.ratings, k=10, creative=False)
                if self.reachedFive == True:
                    if self.i < 10:
                        if line == "No" or line == 'no':
                            response = "If you want to continue, share your opinion on another movie. If not, enter :quit"
                            self.reachedFive = False
                        else:
                            recTitle = self.titles[self.recList[self.i]][0]
                            rand = random.choice(RECOMMEND_RESPONSES)
                            response = rand + recTitle + "! Would you like to hear another one? Enter Yes or No"
                            self.i += 1
                    else:
                        response = "That's all I've got for you:) If you want to continue, share your opinion on another movie. If not, enter :quit"
                        self.i = 0
                        self.reachedFive = self.check_five_data_points()                   

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return response                                         

    def check_five_data_points(self):
        non_zero_ratings = np.flatnonzero(self.user_ratings)
        return len(non_zero_ratings) != 0 and len(non_zero_ratings)%5 == 0

    def load_vader_lexicon(self, filename):
        v_lexicon = dict()
        for line in open(filename, 'r'):
            vals = line.split(',')
            v_lexicon[vals[0]] = float(vals[1])
        return v_lexicon


    def extract_sentiment_for_movies(self, text):
        polar_diff_clauses = text.split(' but ')

        sentiment = []
        primary = self.extract_sentiment(polar_diff_clauses[0], use_starter=True)

        for t in self.extract_titles(polar_diff_clauses[0]):
            sentiment.append((t, primary))

        if len(polar_diff_clauses) == 2:
            for t in self.extract_titles(polar_diff_clauses[1]):
                sentiment.append((t, primary * -1))

        return sentiment


    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information from a line of text.

        Given an input line of text, this method should do any general pre-processing and return the
        pre-processed string. The outputs of this method will be used as inputs (instead of the original
        raw text) for the extract_titles, extract_sentiment, and extract_sentiment_for_movies methods.

        Note that this method is intentially made static, as you shouldn't need to use any
        attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        #############################################################################
        # TODO: Preprocess the text into a desired format.                          #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to your    #
        # implementation to do any generic preprocessing, feel free to leave this   #
        # method unmodified.                                                        #
        #############################################################################


        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

        return text

    def extract_titles(self, preprocessed_input, use_starter = False):
        """Extract potential movie titles from a line of pre-processed text.
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially in the text.
        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.
        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess('I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]
        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = [] 
        if self.creative and '"' not in preprocessed_input and not use_starter:
            candidate_titles = []
            preprocessed_input = preprocessed_input.lower()
            if 'and' not in preprocessed_input:
                preprocessed_input += '.'
            if 'ed the movie' in preprocessed_input:
                preprocessed_input = preprocessed_input.split('ed the movie')[0] + 'ed' + preprocessed_input.split('ed the movie')[1]
            for r in title_regexes:
                matches = re.findall(r, preprocessed_input)
                if len(matches):
                    candidate_titles.extend(matches)
            for title in candidate_titles:
                if title != 'I' or title != 'i':
                    if len(self.find_movies_by_title(title)) != 0:
                        titles.append(title)
        else:
            split = preprocessed_input.split("\"")
            for i in range(len(split)):
                if i%2 == 1:
                    titles.append(split[i])
        for i, t in enumerate(titles):
            titles[i] = t.strip()
        titles = set(titles)
        titles = list(titles)
        return titles


    def standalone_in(self, movie, title):
        if title not in movie:
            return False

        index_before = movie.index(title) - 1
        index_after  = movie.index(title) + len(title)

        if index_before >= 0 and movie[index_before].isalnum():
            return False
        if index_after < len(movie) and movie[index_after].isalnum():
            return False
        return True


    def generate_all_forms(self, title):
        forms = [title]

        year = ''
        if '(' in title and ')' in title:
            if title.find('(')+1 < title.find(')'):
                year = title[title.find("(")+1:title.find(")")]
        words = self.rm_year(title).split(' ')

        if year != '':
            forms.append(' '.join(words[1:]) + ', ' + words[0] + ' (' + year + ')')
        else:
            forms.append(' '.join(words[1:]) + ', ' + words[0])
        return forms


    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.
        - If no movies are found that match the given title, return an empty list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a list
        that contains the index of that matching movie.
        Example:
        ids = chatbot.find_movies_by_title('Titanic')
        print(ids) // prints [1359, 1953]
        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """

        articles = ['A ', 'An ', 'The ']
        if self.creative:
            title = title.lower()
            indexes = []
            titleWords = title.split(" ")
            if title == '':
                return indexes
            for i, j in enumerate(self.titles):
                movie = j[0].lower()
                for form in self.generate_all_forms(title):
                    if self.standalone_in(movie, form):
                        indexes.append(i)
                        break
            return indexes
        else:
            indexes = []
            new_title = title
            for article in articles: 
                if article in title and title.index(article) == 0:
                    if '(' in title:
                        split_title = title.split('(')
                        split_title[0] = split_title[0][len(article):len(split_title[0])-1] + ', ' + article
                        new_title = '('.join(split_title)
                    else:
                        new_title = title[len(article):len(title)] + ', ' + article[:len(article)-1]
            for i, t in enumerate(self.titles):
                possible_title = t[0]
                if new_title == possible_title or title == possible_title:
                    indexes.append(i)
                else: 
                    if '(' in possible_title:
                        possible_title = possible_title[:possible_title.index('(')-1]
                        if new_title == possible_title or title == possible_title:
                            indexes.append(i)
            return indexes




    def mark_negation(self, tokenized):
        result = []
        neg = False
        for word in tokenized:
            if (word in NEG_WORDS or 
                word[-3:] == "n't"):
                neg = not neg
            elif word in STOP_PUNC:
                neg = False
            elif neg:
                word += '_NEG'
            result.append(word)
        return result


    def extract_sentiment(self, preprocessed_input, use_starter = False):
        """Extract a sentiment rating from a line of pre-processed text.
        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.
        As an optional creative extension, return -2 if the sentiment of the text
        is super negative and +2 if the sentiment of the text is super positive.
        Example:
        sentiment = chatbot.extract_sentiment(chatbot.preprocess('I liked "The Titanic"'))
        print(sentiment) // prints 1
        :param preprocessed_input: a user-supplied line of text that has been pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        preprocessed_input = ' '.join(preprocessed_input.split('"')[::2])

        p = PorterStemmer()
        tokenized = word_tokenize(preprocessed_input)
        tokenized = self.mark_negation(tokenized)
        total = 0

        # print("============")
        # print(preprocessed_input)

        amplifier = 1
        for w in tokenized:
           sen, flip = 0, 1
           if w[-4:] == '_NEG':
               w = w[:-4]
               flip = -1

           if self.creative and not use_starter:
               if w in self.vader_lexicon:
                   sen = self.vader_lexicon[w]
               elif p.stem(w) in self.vader_lexicon:
                   sen = self.vader_lexicon[p.stem(w)]
               elif w in STOP_PUNC:
                   amplifier = 1
               elif w in AMPLIFY_WORDS:
                   amplifier = 5

               if "didn't really" in preprocessed_input:
                   amplifier = 1

               if flip == -1:
                   flip = -0.5
           else:
               if w in self.sentiment:
                   sen = 1 if (self.sentiment[w] == 'pos') else -1
               if p.stem(w) in self.sentiment:
                   sen = 1 if (self.sentiment[p.stem(w)] == 'pos') else -1

           total += sen * flip * amplifier

        if self.creative and not use_starter:
            # print("total: " + str(total))
            if total > 2:
                return 2
            elif total < -2:
                return -2
            elif total > 0.5:
                return 1
            elif total < -0.5:
                return -1
            return 0
        else:
            if total > 0:
               return 1
            elif total < 0:
               return -1
            return 0


    def rm_year(self, movie):
        if '(' in movie:
            return movie[:movie.index('(')-1]
        return movie


    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least edit distance
        from the provided title, and with edit distance at most max_distance.

        - If no movies have titles within max_distance of the provided title, return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given title
          than all other movies, return a 1-element list containing its index.
        - If there is a tie for closest movie, return a list with the indices of all movies
          tying for minimum edit distance to the given movie.

        Example:
          chatbot.find_movies_closest_to_title("Sleeping Beaty") # should return [1656]

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title and within edit distance max_distance
        """

        candidates = [[] for i in range(max_distance + 1)]

        title = title.lower()
        for i, movie in enumerate(self.titles):
            movie_name = movie[0].lower()
            for form in self.generate_all_forms(title):
                ed_w_year = edit_distance(movie_name, form, 
                                substitution_cost = 2)
                ed_wo_year = edit_distance(self.rm_year(movie_name), form, 
                                substitution_cost = 2)
                ed = min(ed_w_year, ed_wo_year)

                if ed <= max_distance:
                    candidates[ed].append(i)

        for c in candidates:
            if len(c):
                return c
        return [] 


    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be talking about
        (represented as indices), and a string given by the user as clarification
        (eg. in response to your bot saying "Which movie did you mean: Titanic (1953)
        or Titanic (1997)?"), use the clarification to narrow down the list and return
        a smaller list of candidates (hopefully just 1!)

        - If the clarification uniquely identifies one of the movies, this should return a 1-element
        list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it should return a list
        with the indices it could be referring to (to continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by the clarification
        """
        cardinal_numbers = {'first':0,
                            'second':1,
                            'third':2,
                            'fourth':3,
                            'fifth':4,
                            'sixth':5,
                            'seventh':6,
                            'eighth':7,
                            'ninth':8,
                            'tenth':9,
                            'eleventh':10,
                            'twelth':11,
                            '1st': 0,
                            '2nd': 1,
                            '3rd': 2,
                            '4th': 3, 
                            '5th': 4,
                            '6th': 5,
                            '7th': 6,
                            '8th': 7,
                            '9th': 8,
                            '10th': 9, 
                            '11th': 10,
                            '12th': 11}
        time_new = ['recent', 'newest', 'modern', 'latest', 'last']
        time_old = ['oldest', 'original', 'classic', 'old', 'first']
        final_candidates = []
        dates = {}
        if ' one' in clarification:
            clarification = clarification[:clarification.index(' one')]
        for cardinal_number in cardinal_numbers:
            if cardinal_number in clarification:
                final_candidates.append(candidates[cardinal_numbers[cardinal_number]])
                return final_candidates

        for candidate in candidates:
            movie = self.titles[candidate][0]
            if '(' in movie:
                date = movie[movie.index('(')+1:movie.index(')')]   
                dates[candidate] = date

        if not clarification.isnumeric():
            for time in time_new:
                if time in clarification:
                    final_candidates.append(max(dates, key=dates.get))
                    return final_candidates
            for time in time_old:
                if time in clarification:
                    final_candidates.append(min(dates, key=dates.get))
                    return final_candidates

        elif clarification.isnumeric() and int(clarification) <= len(candidates):
            final_candidates.append(candidates[int(clarification)-1])
            return final_candidates

        for candidate in candidates:
            if clarification.lower() in self.titles[candidate][0].lower():
                final_candidates.append(candidate)
            elif clarification == self.titles[candidate][0]:
                final_candidates.append(candidate)
        return final_candidates

    #############################################################################
    # 3. Movie Recommendation helper functions                                  #
    #############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use any
        attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from 0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered positive

        :returns: a binarized version of the movie-rating matrix
        """
        #############################################################################
        # TODO: Binarize the supplied ratings matrix. Do not use the self.ratings   #
        # matrix directly in this function.                                         #
        #############################################################################

        # The starter code returns a new matrix shaped like ratings but full of zeros.
        binarized_ratings = np.copy(ratings)
        binarized_ratings[(binarized_ratings <= 2.5) & (binarized_ratings > 0)] = -1
        binarized_ratings[binarized_ratings > 2.5] = 1
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        #############################################################################
        # TODO: Compute cosine similarity between the two vectors.
        #############################################################################

        norm_prod = (np.linalg.norm(u) * np.linalg.norm(v))

        if norm_prod != 0:
            similarity = np.dot(u, v) / norm_prod
        else:
            similarity = 0

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return similarity

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative filtering.
        You should return a collection of `k` indices of movies recommendations.
        As a precondition, user_ratings and ratings_matrix are both binarized.
        Remember to exclude movies the user has already rated!
        Please do not use self.ratings directly in this method.
        :param user_ratings: a binarized 1D numpy array of the user's movie ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode
        :returns: a list of k movie indices corresponding to movies in ratings_matrix,
          in descending order of recommendation
        """
        #######################################################################################
        # TODO: Implement a recommendation function that takes a vector user_ratings          #
        # and matrix ratings_matrix and outputs a list of movies recommended by the chatbot.  #
        # Do not use the self.ratings matrix directly in this function.                       #
        #                                                                                     #
        # For starter mode, you should use item-item collaborative filtering                  #
        # with cosine similarity, no mean-centering, and no normalization of scores.          #
        #######################################################################################
        # Populate this list with k movie indices to recommend to the user.
        recommendations = []
        new_ratings = []
        zero_ratings = np.flatnonzero(user_ratings == 0)
        non_zero_ratings = np.flatnonzero(user_ratings)
        for i in zero_ratings:
            weighted_sum = 0
            for j in non_zero_ratings:
                cos_sim = self.similarity(ratings_matrix[i], ratings_matrix[j])
                weighted_sum += user_ratings[j] * cos_sim
            new_ratings.append((weighted_sum, i))
        recommendations = [r[1] for r in sorted(new_ratings, reverse = True)][:k]
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return recommendations

    #############################################################################
    # 4. Debug info                                                             #
    #############################################################################

    def debug(self, line):
        """Return debug information as a string for the line string from the REPL"""
        # Pass the debug information that you may think is important for your
        # evaluators
        debug_info = 'debug info'
        return debug_info

    #############################################################################
    # 5. Write a description for your chatbot here!                             #
    #############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your chatbot
        can do and how the user can interact with it.
        """
        return """
         _____  _____  __     _____  _____  _____  _____ 
        |     ||   __||  |   |   __||   __||_   _||   __|
        |   --||   __||  |__ |   __||__   |  | |  |   __|
        |_____||_____||_____||_____||_____|  |_|  |_____|
        \n
        \n
        Welcome to Celeste. The perfect companion for finding incredible movies 
        or just talking about cinematic masterpieces. 
        \n
        When talking to Celeste make sure you put quotation marks around any movie titles.
        \n        We hope that you enjoy your experience and feel
        that you learned something about yourself. Enjoy!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, run:')
    print('    python3 repl.py')
