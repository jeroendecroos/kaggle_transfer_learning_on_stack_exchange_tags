#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from room007.data import info


class Predictor(object):
    def __init__(self, functional_test=False):
        self.vectorize = None
        self.feature_names = None
        self.functional_test = functional_test

    def fit(self, train_data):
        print('start fitting')
        apply_preprocessing(train_data)
        if self.functional_test:
            train_data, throw_away = train_test_split(train_data, test_size=0.99)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        train_features = self.vectorizer.fit_transform(train_data['titlecontent'])
        self.feature_names = self.vectorizer.get_feature_names()

    def predict(self, test_dataframe):
        print('start predicting')
        apply_preprocessing(test_dataframe)
        predictions = []
        times = 0
        for entry in test_dataframe.to_dict(orient='records'):
            ## TODO: probably be made faster by using panda tricks and mass transform iso one transform per entry
            times += 1
            number_of_tags = 3
            if self.functional_test and times > 10:
                prediction = []
            else:
                test_features = self.vectorizer.transform([entry['titlecontent']])
                entry = test_features.toarray()[0]
                prediction = [self.feature_names[i] for i, score in sorted(enumerate(entry), key=lambda t: t[1] * -1)[:number_of_tags]]
            if 'tags' in entry:
                diff_length = len(prediction) - len(entry['tags'])
                if diff_length > 0:
                    entry['tags'].extend(['']*diff_length)
                if diff_length < 0:
                    prediction.extend(['']*(-1*diff_length))
            predictions.append(prediction)
        return predictions


def remove_numbers(text):
    return re.sub('[0-9]', '', text)


def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)


def apply_preprocessing(data):
    if 'tags' in data:
        data['tags'] = data['tags'].str.split()
    data['titlecontent'] = data['title'] + data['content']
    do_extra_cleaning(data)
