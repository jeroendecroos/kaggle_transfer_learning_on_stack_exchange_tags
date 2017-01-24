#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re
import argparse
import importlib

import pandas as pandas
from sklearn.metrics import f1_score

from room007.data import info
from room007.eval import cross_validation


class Predictor(object):
    def __init__(self, functional_test=False):
        self.functional_test = functional_test

    def fit(self, train_data):
        print('start fitting {}'.format(len(train_data)))

    def predict(self, test_dataframe):
        print('start predicting')
        predictions = [[''] for _ in range(test_dataframe)]
        return predictions

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict with it-idf.')
    parser.add_argument('--eval', help='apply to the testdata', action='store_true')
    parser.add_argument('--test', help='run only on minimal part of data, to test functionality', action='store_true')
    parser.add_argument('--model', help='the name of the model to train and test, should be an importable module containing a Predictor class')
    args = parser.parse_args()
    return args


def write_predictions(test_name, test_dataframe):
    filename = '{}.out.csv'.format(test_name)
    test_dataframe.to_csv(filename, columns=['id','tags'], index=False)


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


def remove_numbers(text):
    return re.sub('[0-9]', '', text)


def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)


def apply_preprocessing(data):
    data['titlecontent'] = data['title'] + ' ' + data['content']
    do_extra_cleaning(data)

def sample_dataframes(dataframes):
    new_dataframes = {}
    for fname, data in sorted(dataframes.items()):
        new_dataframes[fname] = data.sample(n=1000)
    return new_dataframes

def main():
    args = get_arguments()
    data_info = info.CleanedData()
    train_dataframes = info.get_train_dataframes(data_info)
    if args.test:
        train_dataframes = sample_dataframes(train_dataframes)
    predictor_factory = importlib.import_module(args.model).Predictor
    for fname, data in sorted(train_dataframes.items()):
        apply_preprocessing(data)
    if args.eval:
        train_data = pandas.concat([data for name, data in train_dataframes.items()], ignore_index=True)
        predictor = predictor_factory(functional_test=args.eval)
        predictor.fit(train_data)
        test_dataframes = info.get_test_dataframes(data_info)
        if args.test:
            test_dataframes = sample_dataframes(test_dataframes)
        for fname, test_data in test_dataframes.items():
            print('start predicting for {} {} {}'.format(fname, len(test_data), len(train_data)))
            apply_preprocessing(test_data)
            predictions = predictor.predict(test_data)
            test_data['tags'] = predictions
            test_data['tags'] = test_data['tags'].apply(' '.join)
            write_predictions(fname, test_data)
    else:
        learner = predictor_factory(functional_test=args.test)
        result = cross_validation.cross_validate(learner, train_dataframes)
        print(result)


if __name__ == "__main__":
    main()
