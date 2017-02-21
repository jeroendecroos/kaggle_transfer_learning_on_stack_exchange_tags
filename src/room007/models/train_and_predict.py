#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import argparse
import importlib
import time
import logging

import pandas as pandas

from room007.data import info
from room007.eval import cross_validation
from room007.models.model import create_predictor
from room007.util import time_function

logger = logging.getLogger()


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgumentParser, self).__init__(
            description='Predict with a model.')
        self.add_argument('-e', '--eval',
                          help='apply to the test data',
                          action='store_true')
        self.add_argument('-n', '--set-name',
                          default='../interim',
                          help='name of the pre-processed data set')
        self.add_argument('-m', '--model',
                          default='word_tag_predictor',
                          help='the name of the model to train and test, '
                               'should be an importable module containing a '
                               'Predictor class')
        self.add_argument('-a', '--args',
                          nargs='*',
                          default=[],
                          help='Arguments passed to the constructor of '
                               'Predictor. Values with ":" are considered '
                               'kwargs, others are ignored.')

    def parse_args(self):
        args = super(ArgumentParser, self).parse_args()
        args.kwargs = dict(arg.split(':', 1)
                           for arg in args.args
                           if ':' in arg)
        args.args = [arg for arg in args.args if ':' not in arg]
        return args


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


@time_function(logger)
def main():
    args = ArgumentParser().parse_args()
    train_data_frames, test_data_frames = get_data(args.set_name)
    _, predictor = create_predictor(args.model, args.args, args.kwargs)
    if args.eval:
        evaluate_on_test_data(predictor, train_data_frames, test_data_frames)
    else:
        cross_validate(predictor, train_data_frames)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
