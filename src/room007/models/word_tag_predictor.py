#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import collections

import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn import linear_model

from pandas import DataFrame
import nltk
import string
stop_words = nltk.corpus.stopwords.words('english') + [x for x in string.printable]


class Features(object):
    def __init__(self, functional_test=False, changes=False):
        self.functional_test = functional_test
        self.tf_idf_vectorizer = None
        self.changes=changes

    def fit(self, train_data):
        self._train_tf_idf_vectorizer(train_data)

    def transform(self, train_data):
        tf_idf_features = self._get_tf_idf_features_per_word(train_data)
        in_title_features = self._times_word_in('title', train_data)
        in_content_feature = self._times_word_in('content', train_data)
        if self.changes:
            in_question_feature = self._is_in_question(train_data)
            feats = tuple(zip(tf_idf_features, in_title_features, in_content_feature, in_question_feature))
        else:
            feats = tuple(zip(tf_idf_features, in_title_features, in_content_feature))
        return feats

    def _is_in_question(self, train_data):
        def is_in_question(row):
            features = []
            question = 0
            for word in row.split()[::-1]:
                if word in '.:;!?':
                    question = int(word == '?')
                if word not in stop_words:
                    features.append(question)
            import pdb; pdb.set_trace()
            return features[::-1]
        return list(itertools.chain(*train_data['titlecontent'].apply(
            is_in_question
            )))

    def _add_number_of_non_stop_words(self, data):
        data['title_non_stop_words'] = [x for x in row['title'] if x not in stop_words]
        data['content_non_stop_words'] = [x for x in row['content'] if x not in stop_words]

    def _times_word_in(self, column, data):
        return list(itertools.chain(*data.apply(
            lambda row: [row[column].split().count(word)
                         for word in row['titlecontent'].split()
                         if word not in stop_words], axis=1)
            ))


    def _train_tf_idf_vectorizer(self, train_data):
        self.tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tf_idf_vectorizer.fit(train_data['titlecontent'])
        #if self.functional_test:
        #    self._write_example_it_idf_features(train_data)


    def _get_tf_idf_features_per_word(self, train_data):
        tf_idf_data = self.tf_idf_vectorizer.transform(train_data['titlecontent'])
        train_data['index'] = range(tf_idf_data.shape[0])
        voc = self.tf_idf_vectorizer.vocabulary_
        features = list(itertools.chain(*train_data.apply(
            lambda row: [tf_idf_data[row['index'], voc.get(word)]
                         if voc.get(word) else 0
                         for word in row['titlecontent'].split()
                         if word not in stop_words], axis=1)
            ))
        return features


    def _write_some_features(self, features, keys):
        with open('debug_files/feats_per_word', 'wt') as outstream:
            outstream.write(','.join(keys))
            for feat in features:
                outstream.write(','.join(f for f in feat) + '\n')


class Predictor(object):
    def __init__(self, functional_test=False, classifier=None, changes=False):
        if classifier == None:
            classifier = linear_model.LogisticRegression(class_weight='balanced')
        self.classifier = classifier
        self.functional_test = functional_test
        self.changes = changes

    def fit(self, train_data):
        print('start fitting')
        self._fit(train_data)

    def predict(self, test_dataframe):
        print('start predicting')
        predictions = []
        tag_predictions = self.classifier.predict(
                self.feature_creator.transform(test_dataframe)
                )
        line = 0
        for i in range(len(test_dataframe)):
            entry = test_dataframe[i:i+1]
            words = [w for w in entry.titlecontent.values[0].split() if w not in stop_words]
            n_words = len(words)
            tag_predictions[line:line+n_words]
            line += n_words
            prediction = set()
            for pred, word in zip(tag_predictions[line:line+n_words], words):
                if pred:
                    prediction.add(word)
            prediction = list(prediction)
            predictions.append(prediction)
        return predictions

    def _fit(self, train_data):
        print("get features")
        self.feature_creator = Features(changes=self.changes)
        self.feature_creator.fit(train_data)
        features = self.feature_creator.transform(train_data)
        truths = self._get_truths_per_word(train_data)
        print("learning")
        self._learn(features, truths)
        print("finished learning")

    def _learn(self, features, truths):
        self.classifier.fit(features, truths)

    def _get_truths_per_word(self, train_data):
        truths = []
        for i, titlecontent in enumerate(train_data.titlecontent.values):
            words = [w for w in titlecontent.split() if w not in stop_words]
            tags = train_data.tags.values[i]
            truths.extend(w in tags for w in words)
        return truths

    def _predict_for_one_entry(self, entry):
        features = self.feature_creator.transform(entry)
        tag_predictions = self.classifier.predict(features)
        words = [w for w in entry.titlecontent.values[0].split() if w not in stop_words]
        predictions = set()
        for pred, word in zip(tag_predictions, words):
            if pred:
                predictions.add(word)
        return list(predictions)

