#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain
import re
import argparse

import numpy
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.decomposition import NMF

from room007.data import info

from sklearn.feature_extraction.text import TfidfVectorizer

TAG_STATS_IDX = pandas.Index(('present', 'total'))



class Predictor(object):
    def __init__(self):
        self.vectorize = None
        self.feature_names = None

    def fit(self, train_data):
        print('start fitting')
        apply_preprocessing(train_data)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        train_features = self.vectorizer.fit_transform(train_data['titlecontent'])
        self.feature_names = self.vectorizer.get_feature_names()

    def predict(self, test_dataframe):
        print('start predicting')
        apply_preprocessing(test_dataframe)
        predictions = []
        for entry in test_dataframe.to_dict(orient='records'):
            test_features = self.vectorizer.transform([entry['titlecontent']])
            entry = test_features.toarray()[0]
            number_of_tags = 3
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


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict with it-idf.')
    parser.add_argument('--eval', help='apply to the testdata', action='store_true')
    args = parser.parse_args()
    return args


def write_predictions(predictios):
    f = open('test.out.csv', 'w')
    f.write('"id","tags"\n')
    for i, prediction in predictions:
        f.write('"{}","{}"\n'.format(i, ' '.join(prediction)))


def get_scoring(predictions, all_tags):
    score = 0
    j = 0
    for i, prediction in enumerate(predictions):
        diff_length = len(prediction) - len(all_tags[i])
        if diff_length > 0:
            all_tags[i].extend(['']*diff_length)
        if diff_length < 0:
            prediction.extend(['']*(-1*diff_length))
        fscore = f1_score(all_tags[i], prediction, average='macro')
        score += fscore
        j += 1
    return(score/j*100)


def main():
    args = get_arguments()
    data_info = info.CleanedData()
    train_dataframes = info.get_train_dataframes(data_info)
    if args.eval:
        test_dataframe = list(info.get_test_dataframes(data_info))[0]
        predictor = Predictor()
        predictor.fit(train_dataframes)
        predictions = predictor.predict(test_dataframe)
        write_predictions(predictios)
    else:
        avg_score = 0
        for fname, test_data in sorted(train_dataframes.items()):

            train_data = pandas.concat([data for name, data in train_dataframes.items() if name!=fname], ignore_index=True)
            print('start learning for {} {} {}'.format(fname, len(test_data), len(train_data)))
            predictor = Predictor()
            predictor.fit(train_data)
            predictions = predictor.predict(test_data)
            all_tags = test_data['tags']
            avg_score += get_scoring(predictions, all_tags)
        print(avg_score/len(train_dataframes))
    

if __name__ == "__main__":
    main()
