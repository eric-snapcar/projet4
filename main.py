
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
    train = fetch_20newsgroups(subset='train',
                      categories=categories, shuffle=True, random_state=42)
    test = fetch_20newsgroups(subset='test',
        categories=categories, shuffle=True, random_state=42)
    return train, test

# SELECT * FROM posts WHERE Id < 50000 AND Tags IS NOT NULL;

train, test = data()
count_vect = CountVectorizer()
train_counts = count_vect.fit_transform(train.data)
tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(train_counts)
clf = MultinomialNB().fit(train_tfidf, train.target)



testBis = ['God is love', 'OpenGL on the GPU is fast']
testBis_count = count_vect.transform(testBis)
testBis_tfidf = tfidf_transformer.transform(testBis_count)
testBis_predicted = clf.predict(testBis_tfidf)

for doc, category in zip(testBis, testBis_predicted):
    print('%r => %s' % (doc, train.target_names[category]))

"""
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)),
])
text_clf.fit(train.data, train.target)
predicted = text_clf.predict(test.data)
print(metrics.classification_report(test.target, predicted,
    target_names=test.target_names))
print(metrics.confusion_matrix(test.target, predicted))
"""
