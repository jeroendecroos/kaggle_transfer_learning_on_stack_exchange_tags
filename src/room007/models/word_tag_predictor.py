#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re
import collections

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def remove_numbers(text):
    return re.sub('[0-9]', '', text)


def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)


def apply_preprocessing(data):
    if 'tags' in data:
        data['tags'] = data['tags'].str.split()
    data['titlecontent'] = data['title'] + data['content']
    do_extra_cleaning(data)


class Predictor(object):
    def __init__(self, functional_test=False):
        self.functional_test = functional_test
        self.ti_idf_vectorizer = None

    def fit(self, train_data):
        print('start fitting')
        if self.functional_test:
            train_data, throw_away = train_test_split(train_data, test_size=0.9999)
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
        self.ti_idf_vectorizer.fit(train_data['titlecontent'])
        if self.functional_test:
            self._write_example_it_idf_features(train_data)

    def _get_features_per_word(self, train_data):
        features = collections.OrderedDict()
        ti_idf = self._get_ti_idf_features_per_word(train_data)
        features['ti_idf'] = ti_idf
        self._write_some_features(features)
        return features

    def _get_ti_idf_features_per_word(self, train_data):
        ti_idf_data = self.ti_idf_vectorizer.transform(train_data['titlecontent'])
        feature_names = self.ti_idf_vectorizer.get_feature_names()
        features = []
        for i, titlecontent in enumerate(train_data.titlecontent.values):
            words = titlecontent.split()
            ti_idf_values = ti_idf_data[i].toarray()[0]
            pf = [ti_idf_values[feature_names.index(word)] if word in feature_names else 0 for word in words]
            features.extend(pf)
        return features


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

    def _write_example_it_idf_features(self, train_data):
        words = train_data.titlecontent.values[0].split()
        features_one = [self.ti_idf_vectorizer.transform([x]).toarray() for x in words]
        features_two = self.ti_idf_vectorizer.transform([' '.join(words)]).toarray()
        self.feature_names = self.ti_idf_vectorizer.get_feature_names()
        with open('debug_files/it_idf_feats_per_word', 'wt') as outstream:
            outstream.write(' '.join(train_data.tags.values[0]) + '\n')
            outstream.write(' '.join(words) + '\n')
            for feat in features_one:
                feat = feat[0]
                outstream.write('A: ')
                for i, value in enumerate(feat):
                    if value != 0:
                        name = self.feature_names[i]
                        outstream.write('{}:{}:{}:{} , '.format(i, features_two[0][i], name, value))
                outstream.write('\n')

    def _write_some_features(self, features):
        with open('debug_files/feats_per_word', 'wt') as outstream:
            outstream.write(','.join(features.keys()))
            for i in range(10):
                outstream.write(','.join(str(f[i]) for f in features.values()) + '\n')

