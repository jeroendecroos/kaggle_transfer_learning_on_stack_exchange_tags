#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import collections

import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes
from sklearn import linear_model

import spacy
nlp = spacy.load('en')

from pandas import DataFrame
import nltk
import string
stop_words = nltk.corpus.stopwords.words('english') + [x for x in string.printable]




class Features(object):
    def __init__(self, functional_test=False, changes=False):
        self.functional_test = functional_test
        self.tf_idf_vectorizer = None
        self.changes = changes
        self.feature_dir = 'data/features'

    def load_or_create_feature(self, fun, feature_name, optional_save=False):
        def _fun(data, *args, **kwargs):
            feature_location = os.path.join(self.feature_dir, feature_name)
            if os.path.exist(feature_location):
                features = self._load_features(data['id'], feature_location)
            else:
                features = fun(data, *args, **kwargs)
                if optional_save:
                    self._save_features(data, features)
            return features
        return _fun

    def _load_features(self, ids, filepath):
        iter_csv = pandas.read_csv(filepath, iterator=True, chunksize=1000)
        dataframe = pandas.concat([chunk[chunk['id'] in ids] for chunk in iter_csv])

    def _save_features(self, data, features):
        features = deque(features)
        data['features'] = data.apply(lambda row:
                [features.popleft() for _ in range(len(row['title_non_stop_words'])+len(row['content_non_stop_words']))],
            axis=1)
        pass

    def fit(self, train_data):
        self._train_tf_idf_vectorizer(train_data)

    def transform(self, data):
        self._add_number_of_non_stop_words(data)
        features = [
                self._get_tf_idf_features_per_word(data),
                self._times_word_in(data, 'title'),
                self._times_word_in(data, 'content'),
                self._is_in_question(data),
                self._title_or_content(data),
        ]
        if self.changes:
            s_feats = self._spacy_features(data)
            features.extend(s_feats)
        feats = tuple(zip(*features))
        return feats

    def _spacy_features(self, data):
        return [[x for x in itertools.chain(*data.apply(
            lambda row: [tags.pos_ == "NOUN" for word, tags in zip(row['titlecontent'].split(), nlp(row['titlecontent'])) if word not in stop_words],
            axis=1))]]

    def _title_or_content(self, data):
        return list(itertools.chain(*data.apply(
            lambda row: [1] * len(row['title_non_stop_words']) + [0] * len(row['content_non_stop_words']),
            axis=1)))


    def _add_number_of_non_stop_words(self, data):
        split_row = lambda row: [x for x in row.split() if x not in stop_words]
        data['title_non_stop_words'] = data['title'].apply(split_row)
        data['content_non_stop_words'] = data['content'].apply(split_row)


    def _is_in_question(self, data):
        def is_in_question(row):
            features = []
            question = 0
            for word in row.split()[::-1]:
                if word in '.:;!?':
                    question = int(word == '?')
                if word not in stop_words:
                    features.append(question)
            return features[::-1]
        return list(itertools.chain(*data['titlecontent'].apply(
            is_in_question
            )))

    def _times_word_in(self, data, column):
        return list(itertools.chain(*data.apply(
            lambda row: [row[column].split().count(word)
                         for word in row['titlecontent'].split()
                         if word not in stop_words], axis=1)
            ))


    def _train_tf_idf_vectorizer(self, data):
        self.tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tf_idf_vectorizer.fit(data['titlecontent'])
        #if self.functional_test:
        #    self._write_example_it_idf_features(data)


    def _get_tf_idf_features_per_word(self, train_data):
        tf_idf_data = self.tf_idf_vectorizer.transform(train_data['titlecontent'])
        train_data['index'] = range(tf_idf_data.shape[0])
        voc = self.tf_idf_vectorizer.vocabulary_
        features = list(itertools.chain(*train_data.apply(
            lambda row: [tf_idf_data[row['index'], voc.get(word)]
                         if voc.get(word) else 1
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

