
from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups

dataSize = 2000
vocabularySize = 1000
numberOfTopics = 10
numberOfTopWords = 5

def display(model, feature_names, numberOfTopWords):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic %d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-numberOfTopWords - 1:-1]])
        print(message)
    print()

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
data_samples = dataset.data[:dataSize]
tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=vocabularySize,
                                   stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(data_samples)
"""
nmf = NMF(n_components=numberOfTopics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()
display(nmf, tfidf_feature_names, numberOfTopWords)
"""

nmf = NMF(n_components=numberOfTopics, random_state=1,
          beta_loss='kullback-leibler', solver='mu', max_iter=1000, alpha=.1,
          l1_ratio=.5).fit(tfidf)


display(nmf, tfidf_vectorizer.get_feature_names(), numberOfTopWords)
