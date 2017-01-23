#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import argparse
import importlib

import pandas as pandas
from sklearn.metrics import f1_score

from room007.data import info



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

def main():
    args = get_arguments()
    data_info = info.CleanedData()
    train_dataframes = info.get_train_dataframes(data_info)
    predictor_factory = importlib.import_module(args.model).Predictor
    if args.eval:
        train_data = pandas.concat([data for name, data in train_dataframes.items()], ignore_index=True)
        predictor = predictor_factory(functional_test=args.eval)
        predictor.fit(train_data)
        test_dataframes = info.get_test_dataframes(data_info)
        for fname, test_data in test_dataframes.items():
            print('start predicting for {} {} {}'.format(fname, len(test_data), len(train_data)))
            predictions = predictor.predict(test_data)
            test_data['tags'] = predictions
            test_data['tags'] = test_data['tags'].apply(' '.join)
            write_predictions(fname, test_data)
    else:
        avg_score = 0
        for fname, test_data in sorted(train_dataframes.items()):

            train_data = pandas.concat([data for name, data in train_dataframes.items() if name!=fname], ignore_index=True)
            print('start learning for {} {} {}'.format(fname, len(test_data), len(train_data)))
            predictor = predictor_factory(functional_test=args.test)
            predictor.fit(train_data)
            predictions = predictor.predict(test_data)
            all_tags = test_data['tags']
            score = get_scoring(predictions, all_tags)
            avg_score += score
            print(score)
        print(avg_score/len(train_dataframes))


if __name__ == "__main__":
    main()
