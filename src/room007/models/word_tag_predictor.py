#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import collections
import itertools

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

from pandas import DataFrame


class Features(object):
    def __init__(self, functional_test=False):
        self.functional_test = functional_test
        self.tf_idf_vectorizer = None

    def fit(self, train_data):
        self._train_tf_idf_vectorizer(train_data)

    def transform(self, train_data):
        features = self._get_features_per_word(train_data)
        features = [ [f[i] for f in features.values()] for i in range(len(features['tf_idf']))]
        return features

    def _train_tf_idf_vectorizer(self, train_data):
        self.tf_idf_vectorizer = TfidfVectorizer(stop_words='english')
        self.tf_idf_vectorizer.fit(train_data['titlecontent'])
        feature_names = self.tf_idf_vectorizer.get_feature_names()
        self.feature_names = {word: i for i, word in enumerate(feature_names)}
        if self.functional_test:
            self._write_example_it_idf_features(train_data)

    def _get_features_per_word(self, train_data):
        features = collections.OrderedDict()
        tf_idf = self._get_tf_idf_features_per_word(train_data)
        features['tf_idf'] = tf_idf
        #self._write_some_features(features)
        return features

    def _get_tf_idf_features_per_word(self, train_data):
        tf_idf_data = self.tf_idf_vectorizer.transform(train_data['titlecontent']).toarray()
        train_data['index']= range(len(tf_idf_data))
        voc = self.tf_idf_vectorizer.vocabulary_
        features = list(itertools.chain(*train_data.apply(
                lambda row: [tf_idf_data[row['index']][voc.get(word)]
                             if voc.get(word) else 0
                             for word in row['titlecontent'].split()], axis=1)
                ))
        return features

    def _write_example_it_idf_features(self, train_data):
        features_two = self.tf_idf_vectorizer.transform([' '.join(words)]).toarray()
        self.feature_names = self.tf_idf_vectorizer.get_feature_names()
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
            for i in range(2):
                outstream.write(','.join(str(f[i]) for f in features.values()) + '\n')


class Predictor(object):
    def __init__(self, functional_test=False):
        self.functional_test = functional_test

    def fit(self, train_data):
        print('start fitting')
        self._fit(train_data)

    def predict(self, test_dataframe):
        print('start predicting')
        predictions = []
        for entry in test_dataframe.to_dict(orient='records'):
            ## TODO: probably be made faster by using panda tricks and mass transform iso one transform per entry
            prediction = self._predict_for_one_entry(entry)
            self._align_prediction(prediction, entry)
            predictions.append(prediction)
        return predictions


    def _fit(self, train_data):
        self.feature_creator = Features()
        self.feature_creator.fit(train_data)
        features = self.feature_creator.transform(train_data)
        truths = self._get_truths_per_word(train_data)
        self._learn(features, truths)

    def _learn(self, features, truths):
        self.logreg = linear_model.LogisticRegression(C=1e5)
        self.logreg.fit(features, truths)

    def _get_truths_per_word(self, train_data):
        truths = []
        for i, titlecontent in enumerate(train_data.titlecontent.values):
            words = titlecontent.split()
            tags = train_data.tags.values[i]
            truths.extend(w in tags for w in words)
        return truths

    def _predict_for_one_entry(self, entry):
        prediction = set()
        data = DataFrame.from_dict(entry)
        features = self.feature_creator.transform(data)
        predictions = self.logreg.predict(features)
        words = data.titlecontent.values[0].split()
        for pred, word in zip(predictions, words):
            if pred:
                prediction.add(word)
        return list(prediction)

    def _align_prediction(self, prediction, entry):
        if 'tags' in entry:
            diff_length = len(prediction) - len(entry['tags'])
            if diff_length > 0:
                entry['tags'].extend(['']*diff_length)
            if diff_length < 0:
                prediction.extend(['']*(-1*diff_length))

