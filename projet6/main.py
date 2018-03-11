
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.linear_model import SGDClassifier

def data():
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']
    twenty_train = fetch_20newsgroups(subset='train',
                      categories=categories, shuffle=True, random_state=42)
    twenty_test = fetch_20newsgroups(subset='test',
        categories=categories, shuffle=True, random_state=42)
    return twenty_train, twenty_test
twenty_train, twenty_test = data()
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(twenty_test.data)

print(metrics.classification_report(twenty_test.target, predicted,
    target_names=twenty_test.target_names))
print(metrics.confusion_matrix(twenty_test.target, predicted))
