#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain
from os.path import basename, join, splitext
import re

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


from room007.data import info


TAG_STATS_IDX = pd.Index(('present', 'total'))




def norm_tag(tag):
    return re.sub('-', ' ', tag)

def predict_per_category(data):
    # split data, 0.2 arbitrary chosen
    train, test = train_test_split(data, test_size=0.2)
    # get features 
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    train_counts = count_vect.fit_transform(train['titlecontent'])
    train_tfidf_features = tfidf_transformer.fit_transform(train_counts)
    test_counts = count_vect.transform(test['titlecontent'])
    test_tfidf_features = tfidf_transformer.transform(test_counts)
    # just to use whatever classifier
    #tag_count_vect = CountVectorizer()
    #tag_train_counts = count_vect.fit_transform(train['tags'])
    #tag_test_counts = count_vect.transform(test['tags'])
    train_tfidf_features = tfidf_transformer.fit_transform(train_counts)
    test_counts = count_vect.transform(test['titlecontent'])
    classifier = MultinomialNB().fit(train_tfidf_features, train['tags'])
    predicted_tags = classifier.predict(test_tfidf_features)
    print(numpy.mean(predicted_tags == test['tags']))


def main():
    data_info = info.CleanedData()
    dataframes = info.get_train_dataframes(data_info)
    for fname, data in dataframes.items():
        #data['tags'] = data['tags'].str.split()
        data['titlecontent'] = data['title'] + data['content']
        predict_per_category(data)

if __name__ == "__main__":
    main()
