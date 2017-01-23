#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer



class Predictor(object):
    def __init__(self, functional_test=False):
        self.functional_test = functional_test
        self.ti_idf_vectorizer = None

    def fit(self, train_data):
        print('start fitting')
        if self.functional_test:
            train_data, throw_away = train_test_split(train_data, test_size=0.99)
        apply_preprocessing(train_data)
        self._fit(train_data)

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
                prediction = self._predict_for_one_entry(entry)
            else:
                prediction = []
            self._align_prediction(prediction, entry)
            predictions.append(prediction)
        return predictions


    def _fit(self, train_data):
        self._train_ti_idf_vectorizer(train_data)
        features = self._get_features_per_word(train_data)
        truths = self._get_truths_per_word(train_data)
        self._learn(features, truth)

    def _train_ti_idf_vectorizer(self, train_data):
        self.ti_idf_vectorizer = TfidfVectorizer(stop_words='english')
        train_features = self.ti_idf_vectorizer.fit(train_data['titlecontent'])

    def _get_features_per_word(self, train_data):
        import pdb; pdb.set_trace()
        vec = self.ti_idf_vectorizer
        features = self.ti_idf_vectorizer.transform(train_data['titlecontent'])
        features2 = [self.ti_idf_vectorizer.transform([x]) for x in train_data.titlecontent.values[0].split()]




    def _predict_for_one_entry(self, entry):
        prediction = set()
        for word in entry['titlecontent']:
            features = self._get_features_for_word(word)
            if self._predict_if_tag(word):
                prediction.add(word)
        return list(prediction)

    def _align_prediction(self, prediction, entry):
        if 'tags' in entry:
            diff_length = len(prediction) - len(entry['tags'])
            if diff_length > 0:
                entry['tags'].extend(['']*diff_length)
            if diff_length < 0:
                prediction.extend(['']*(-1*diff_length))


def remove_numbers(text):
    return re.sub('[0-9]', '', text)


def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)


def apply_preprocessing(data):
    if 'tags' in data:
        data['tags'] = data['tags'].str.split()
    data['titlecontent'] = data['title'] + data['content']
    do_extra_cleaning(data)
