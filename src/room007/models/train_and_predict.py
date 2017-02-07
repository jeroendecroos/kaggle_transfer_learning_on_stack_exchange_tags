#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import argparse
import importlib
import time
import logging

import pandas as pandas

from room007.eval import cross_validation
from room007.data import info

logger = logging.getLogger()


class Predictor(object):
    def __init__(self, functional_test=False):
        self.functional_test = functional_test

    def fit(self, train_data):
        print('start fitting {}'.format(len(train_data)))

    def predict(self, test_data_frame):
        print('start predicting')
        predictions = [[''] for _ in range(test_data_frame)]
        return predictions


class ArgumentParser(object):
    def __init__(self):
        parser = argparse.ArgumentParser(description='Predict with a model.')
        parser.add_argument('-e', '--eval', help='apply to the test data', action='store_true')
        parser.add_argument('-n', '--set-name', default='../interim',
                help='name of the pre-processed data set')
        parser.add_argument('-m', '--model', default='word_tag_predictor',
                help=('the name of the model to train and test, '
                      'should be an importable module containing a Predictor class'))
        parser.add_argument('-a', '--args', nargs='*', default=[],
                help='arguments passed to the constructor of Predictor, values with ":" are considered kwargs')
        self.parser = parser

    def parse_args(self):
        args = self.parser.parse_args()
        args.kwargs = dict(arg.split(':') for arg in args.args)
        args.args = [arg for arg in args.args if ':' not in arg]
        return args


def get_arguments():
    parser = ArgumentParser()
    return parser.parse_args()


def write_predictions(test_name, test_dataframe):
    filename = '{}.out.csv'.format(test_name)
    test_dataframe.to_csv(filename, columns=['id','tags'], index=False)


def get_data(set_name):
    logger.info('loading the data')
    processed_info = info.ProcessedData(set_name)
    train_data_frames = info.get_train_dataframes(processed_info, split_tags=True)
    test_data_frames = info.get_test_dataframes(processed_info)
    logger.info('data loaded')
    return train_data_frames, test_data_frames


def _create_predictor(model, args, kwargs):
    logger.info('creating predictor')
    predictor_factory = importlib.import_module(model).Predictor
    predictor = predictor_factory(*args, **kwargs)
    logger.info('predictor created')
    return predictor


def time_function(fun):
    def timed_fun(*args, **kwargs):
        logger.info('started at {}'.format(time.strftime('%H:%M:%S', time.gmtime())))
        start_time = time.time()
        returns = fun(*args, **kwargs)
        end_time = time.time()
        time_needed = end_time - start_time
        logger.info('finished at {}'.format(time.strftime('%H:%M:%S', time.gmtime())))
        logger.info("it took: {0:.0f} seconds".format(time_needed))
        return returns, time_needed
    return timed_fun


def evaluate_on_test_data(predictor, train_data_frames, test_data_frames):
    train_data = pandas.concat([data for _, data in train_data_frames.items()], ignore_index=True)
    predictor.fit(train_data)
    for frame_name, test_data in test_data_frames.items():
        logger.info('start predicting for {}, test size {}'.format(frame_name, len(test_data)))
        predictions = predictor.predict(test_data)
        logger.info('done predicting')
        test_data['tags'] = predictions
        test_data['tags'] = test_data['tags'].apply(' '.join)
        logger.info('writing result to file')
        write_predictions(frame_name, test_data)
        logger.info('result written')


def cross_validate(predictor, train_data_frames):
    logger.info('started cross-validation testing')
    result = cross_validation.cross_validate(predictor, train_data_frames)
    logger.info('finished cross-validation testing')
    logger.info("cross-validation result: {}".format(result))
    return result


@time_function
def main():
    args = get_arguments()
    train_data_frames, test_data_frames = get_data(args.set_name)
    predictor = _create_predictor(args.model, args.args, args.kwargs)
    if args.eval:
        evaluate_on_test_data(predictor, train_data_frames, test_data_frames)
    else:
        cross_validate(predictor, train_data_frames)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
