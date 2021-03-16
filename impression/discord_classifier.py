from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from operator import itemgetter

from impression.message import DiscordMetaChannel
import json


def get_classifier(path, train_test_ratio):
    dmc = DiscordMetaChannel('info.json')
    text_clf = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultinomialNB()),
    ])
    dataset = [(x["content"], x["author"]["name"]) for x in dmc if x["type"] == "Default" and
               x["author"]["name"] in set(json.load(open(".env", encoding='utf8'))["whitelist"])]

    train_n, test_n = int(len(dataset) * train_test_ratio), int(len(dataset) * (1 - train_test_ratio))
    X_train, y_train, X_test, y_test = tuple(map(itemgetter(0), dataset[:train_n])), \
                                       tuple(map(itemgetter(1), dataset[:train_n])), \
                                       tuple(map(itemgetter(0), dataset[test_n:])), \
                                       tuple(map(itemgetter(1), dataset[test_n:]))

    text_clf.fit(X_train, y_train)

    return text_clf
