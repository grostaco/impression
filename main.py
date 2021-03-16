from impression.message import DiscordMetaChannel
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from operator import itemgetter

dmc = DiscordMetaChannel('info.json')
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

dataset = [(x["content"], x["author"]["name"]) for x in dmc if x["type"] == "Default" and
           x["author"]["name"] in {"~Chansey~", "(traits*)sbrk(sizeof *traits)", "RoaR",
                                   "🍔🥤MrChacocha🥤🍔"}]

X_train, y_train, X_test, y_test = tuple(map(itemgetter(0), dataset[:50000])), tuple(map(itemgetter(1), dataset[:50000])),\
                                   tuple(map(itemgetter(0), dataset[50000:])), tuple(map(itemgetter(1), dataset[50000:]))

text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

while True :
    x = input("Text to predict: ")
    print(text_clf.predict([x])[0])