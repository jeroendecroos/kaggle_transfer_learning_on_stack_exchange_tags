#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import re
import importlib
import logging

from room007.data import info
from room007.logging import loggingmgr
from room007.models import model, train_and_predict
from room007.preprocessing import preprocess
from room007.util import time_function

loggingmgr.set_up()
logger = logging.getLogger(__name__)


class ArgumentParser(train_and_predict.ArgumentParser):
    def __init__(self):
        train_and_predict.ArgumentParser.__init__(self)
        self.parser.add_argument('--test', help='run only on minimal part of data, to test functionality', action='store_true')
        self.parser.add_argument('--speedtest', help='run only on small part of data, but big enough for speed profiling', action='store_true')
        self.parser.add_argument('--reduce-train-data', help='run with same amount of train data as speedteest, but full testset', action='store_true')


def write_predictions(test_name, test_dataframe):
    filename = '{}.out.csv'.format(test_name)
    test_dataframe.to_csv(filename, columns=['id','tags'], index=False)


def sample_dataframes(dataframes, size):
    new_dataframes = {}
    for fname, data in sorted(dataframes.items()):
        new_dataframes[fname] = data.sample(frac=size)
    return new_dataframes


def _get_sample_size(test):
    return 0.001 if test else 0.01


def _get_data(args):
    def _select_and_process(dataframes, reduce_data):
        if args.test or args.speedtest or reduce_data:
            size = _get_sample_size(args.test)
            dataframes = sample_dataframes(dataframes, size)
        for _, data in sorted(dataframes.items()):
            preprocess(data)
        return dataframes
    data_info = info.CleanedData()
    train_data_frames, test_data_frames = train_and_predict.get_data(args.set_name)
    train_data_frames = _select_and_process(train_data_frames, False) ## this false should be adjustable
    if args.eval:
        test_data_frames = _select_and_process(test_data_frames, False)
    else:
        test_data_frames = {}
    return train_data_frames, test_data_frames


def log_results(results):
    logger.info('#################################################')
    logger.info('###   score   ###   parameters   ###   time   ###')
    logger.info('#################################################')
    for name, result, time_needed in sorted(results, key=lambda x: x[1]*-1):
        logger.info("{:.4f} {} {:.2f}".format(result, name, time_needed))
    logger.info('#################################################')


@time_function(logger)
def main():
    args = ArgumentParser().parse_args()
    train_data_frames, test_data_frames = _get_data(args)
    model_module = importlib.import_module(args.model)
    predictor_factory = model_module.Predictor
    if args.eval:
        predictor = predictor_factory(*args.args, **args.kwargs)
        train_and_predict.evaluate_on_test_data(predictor, train_data_frames, test_data_frames)
    else:
        results = []
        options_setter = getattr(model_module, "OptionsSetter", model.OptionsSetter)()
        for name, options in options_setter.combinations(args.kwargs):
            logger.info('started cross_validation for {}'.format(name))
            predictor = predictor_factory(*args.args, **options)
            result, time_needed = time_function(logger)(
                train_and_predict.cross_validate)(
                predictor, train_data_frames)
            results.append((name, result, time_needed))
        log_results(results)


if __name__ == "__main__":
    main()
