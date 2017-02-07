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


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict with it-idf.')
    parser.add_argument('-e', '--eval', help='apply to the test data', action='store_true')
    parser.add_argument('-n', '--set-name', help='name of the pre-processed data set')
    parser.add_argument('-m', '--model', help=('the name of the model to train and test, '
                                         'should be an importable module containing a Predictor class'))
    parser.add_argument('-a', '--args', nargs='*', default=[],
                        help='arguments passed to the constructor of Predictor')
    args = parser.parse_args()
    return args


def write_predictions(test_name, test_dataframe):
    filename = '{}.out.csv'.format(test_name)
    test_dataframe.to_csv(filename, columns=['id','tags'], index=False)


def main():
    args = get_arguments()
    start_time = time.time()
    logger.info('started at {}'.format(time.strftime('%H:%M:%S', time.gmtime())))
    logger.info('loading the data')
    processed_info = info.ProcessedData(args.set_name)
    train_data_frames = info.get_train_dataframes(processed_info, split_tags=False)
    test_data_frames = info.get_test_dataframes(processed_info)
    logger.info('data loaded')

    logger.info('creating predictor')
    predictor_factory = importlib.import_module(args.model).Predictor
    predictor = predictor_factory(*args.args)
    logger.info('predictor created')

    if args.eval:
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
    else:
        logger.info('started cross-validation testing')
        result = cross_validation.cross_validate(predictor, train_data_frames)
        logger.info('finished cross-validation testing')
        logger.info("cross-validation result: {0:.0f}".format(result))

    end_time = time.time()
    time_needed = end_time - start_time
    logger.info('finished at {}'.format(time.strftime('%H:%M:%S', time.gmtime())))
    logger.info("it took: {0:.0f} seconds".format(time_needed))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
