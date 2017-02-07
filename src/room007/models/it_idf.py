#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from sklearn.feature_extraction.text import TfidfVectorizer

import nltk
import string
stop_words = nltk.corpus.stopwords.words('english') + [x for x in string.printable]


class Predictor(object):
    def __init__(self, *args):
        self.vectorizer = None
        self.feature_names = None
        self.functional_test = 'functional-test' in args

    def fit(self, train_data):
        print('start fitting')
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)
        train_features = self.vectorizer.fit_transform(train_data['titlecontent'])
        self.feature_names = self.vectorizer.get_feature_names()

    def predict(self, test_dataframe):
        print('start predicting')
        number_of_tags = 3
        tf_idf_data = self.vectorizer.transform(test_dataframe['titlecontent'])
        test_dataframe['index'] = range(tf_idf_data.shape[0])
        test_dataframe['nonzeros'] = test_dataframe['index'].apply(
                lambda x: tf_idf_data[x, :].nonzero()[1].tolist())
        predictions = test_dataframe.apply(
            lambda row: [self.feature_names[i] for i, value in 
                         sorted(
                             zip(row['nonzeros'],[ tf_idf_data[row['index'], x] for x in row['nonzeros']]),
                             key=lambda t: t[1] * -1)[:number_of_tags]
                         ],
            axis=1)
        return predictions
