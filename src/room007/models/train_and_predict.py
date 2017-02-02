#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re
import argparse
import importlib
import time

import pandas as pandas
from sklearn.metrics import f1_score

from room007.data import info
from room007.eval import cross_validation

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

classifiers = {    ## would be better to add the names to this
    "Nearest Neighbors": KNeighborsClassifier(3),
#    "Linear SVM": SVC(kernel="linear", C=0.025),
#    "RBF SVM": SVC(gamma=2, C=1),
    #"Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), # too slow?
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#    "Neural Net": MLPClassifier(alpha=1),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "QDA": QuadraticDiscriminantAnalysis()
}


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
    parser.add_argument('--speedtest', help='run only on small part of data, but big enough for speed profiling', action='store_true')
    parser.add_argument('--reduce-train-data', help='run with same amount of train data as speedteest, but full testset', action='store_true')
    parser.add_argument('--model', help='the name of the model to train and test, should be an importable module containing a Predictor class')
    parser.add_argument('--classifier', help='what classifier the Predictor class uses')
    args = parser.parse_args()
    return args


def write_predictions(test_name, test_dataframe):
    filename = '{}.out.csv'.format(test_name)
    test_dataframe.to_csv(filename, columns=['id','tags'], index=False)


def remove_numbers(text):
    return re.sub('[0-9]', '', text)


def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)
    data['title'] = data['title'].map(remove_numbers)
    data['content'] = data['content'].map(remove_numbers)


def apply_preprocessing(data):
    data['titlecontent'] = data['title'] + ' ' + data['content']
    do_extra_cleaning(data)


def sample_dataframes(dataframes, size):
    new_dataframes = {}
    for fname, data in sorted(dataframes.items()):
        new_dataframes[fname] = data.sample(frac=size)
    return new_dataframes


def _get_sample_size(test):
    #return 2000 if speed else 10
    return 0.001 if test else 0.01


def _get_train_test_data(args):
    def _select_and_process(dataframes, reduce_data):
        if args.test or args.speedtest or reduce_data:
            size = _get_sample_size(args.test)
            dataframes = sample_dataframes(dataframes, size)
        for _, data in sorted(dataframes.items()):
            apply_preprocessing(data)
        return dataframes
    data_info = info.CleanedData()
    train_dataframes = _select_and_process(info.get_train_dataframes(data_info), True)
    if args.eval:
        test_dataframes = _select_and_process(info.get_test_dataframes(data_info), False)
    else:
        test_dataframes = {}
    return train_dataframes, test_dataframes

def main():
    args = get_arguments()
    train_dataframes, test_dataframes = _get_train_test_data(args)
    predictor_factory = importlib.import_module(args.model).Predictor
    if args.eval:
        train_data = pandas.concat([data for name, data in train_dataframes.items()], ignore_index=True)
        classifier = getattr('classifier', args, "Logistic Regression")
        predictor = predictor_factory(functional_test=args.eval, classifier=classifier)
        predictor.fit(train_data)
        for fname, test_data in test_dataframes.items():
            print('start predicting for {} {} {}'.format(fname, len(test_data), len(train_data)))
            predictions = predictor.predict(test_data)
            test_data['tags'] = predictions
            test_data['tags'] = test_data['tags'].apply(' '.join)
            write_predictions(fname, test_data)
    else:
        results = []
        if args.classifier:
            global classifiers
            classifiers = {args.classifier: classifiers[args.classifier]}
        for name, classifier in classifiers.items():
            t0 = time.time()
            print('started learning for {}'.format(name))
            learner = predictor_factory(functional_test=args.test, classifier=classifier)
            result = cross_validation.cross_validate(learner, train_dataframes)
            t1 = time.time()
            time_needed = t1-t0
            print("{} {} {}".format(result, name, time_needed))
            results.append((name, result, time_needed))
            if True: ##changes
                t0 = time.time()
                name = name+' changes'
                print('started learning for {}'.format(name))
                learner = predictor_factory(functional_test=args.test, classifier=classifier, changes=True)
                result = cross_validation.cross_validate(learner, train_dataframes)
                t1 = time.time()
                time_needed = t1-t0
                print("{} {} {}".format(result, name, time_needed))
                results.append((name, result, time_needed))
        print('############################################"')
        for name, result, time_needed in sorted(results, key=lambda x: x[1]*-1):
            print("{} {} {}".format(result, name, time_needed))



if __name__ == "__main__":
    main()
