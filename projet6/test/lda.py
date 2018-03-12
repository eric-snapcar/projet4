from __future__ import print_function
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups


#http://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html#sphx-glr-auto-examples-applications-plot-topics-extraction-with-nmf-lda-py

def display(model, feature_names, n_top_words):
    for i, topic in enumerate(model.components_):
        message = "Topic %d: " % i
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)


dataSize = 2000
vocabularySize = 1000
numberOfTopics = 5
numberOfTopWords = 3

dataset = fetch_20newsgroups(shuffle=True, random_state=1,
                             remove=('headers', 'footers', 'quotes'))
train_data = dataset.data[:dataSize]

count_vect = CountVectorizer(max_df=0.95, min_df=2,max_features=vocabularySize,stop_words='english')
train_count = count_vect.fit_transform(train_data)

lda = LatentDirichletAllocation(n_components=numberOfTopics, max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
lda.fit(train_count)

names = count_vect.get_feature_names()


display(lda, names, numberOfTopWords)
