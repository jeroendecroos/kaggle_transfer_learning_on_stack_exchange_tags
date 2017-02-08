#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from room007.data import info
import logging
import argparse
import importlib

logger = logging.getLogger()


def main(args):
    processor = importlib.import_module(args.processor).Processor(*args.processor_args)
    processed_info = info.ProcessedData(args.set_name)

    cleaned_info = info.CleanedData()
    data = info.get_train_dataframes(cleaned_info, split_tags=False)
    data = processor.process(data)
    info.save_training_data(processed_info, data)

    data = info.get_test_dataframes(cleaned_info)
    data = processor.process(data)
    info.save_test_data(processed_info, data)


def get_arguments():
    parser = argparse.ArgumentParser(description='Pre-process data.')
    parser.add_argument('-n', '--set-name', help='name of the resulting data set')
    parser.add_argument('-p', '--processor', help='importable module name with Processor class in it')
    parser.add_argument('-a', '--processor-args', nargs='*', default=[],
                        help='arguments to the processor constructor')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(get_arguments())
