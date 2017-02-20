# -*- coding: utf-8 -*-
import os
from os.path import dirname, join, realpath

PROJECT_DIR = dirname(dirname(dirname(dirname(realpath(__file__)))))
import pandas


class Data(object):
    def __init__(self):
        self.training_sets = [
                'biology',
                'cooking',
                'crypto',
                'diy',
                'robotics',
                'travel',
                ]
        self.test_sets = ['test']
        self.extension = '.csv'
        self.features = ['id', 'title', 'content']
        self.labels = ['tags']
        self.data_dir = ''

    @property
    def training_files(self):
        for f in self._iterate_files(self.training_sets):
            yield f

    @property
    def test_files(self):
        for f in self._iterate_files(self.test_sets):
            yield f

    def _iterate_files(self, data_sets):
        for filebase in data_sets:
            filename = filebase + self.extension
            filepath = join(self.data_dir, filename)
            yield filepath


class RawData(Data):
    def __init__(self):
        super().__init__()
        self.data_dir = join(PROJECT_DIR, 'data', 'raw')


class CleanedData(Data):
    def __init__(self):
        super().__init__()
        self.data_dir = join(PROJECT_DIR, 'data', 'interim')


class ProcessedData(Data):
    def __init__(self, name):
        super().__init__()
        self.data_dir = join(PROJECT_DIR, 'data', 'processed', name)


class FeaturedData(Data):
    def __init__(self, name):
        super().__init__()
        self.data_dir = join(PROJECT_DIR, 'data', 'features', name)


def load_dataset(filepath, split_tags=True):
    frame = pandas.read_csv(filepath)
    if split_tags:
        frame['tags'] = frame['tags'].str.split()
    return frame


def get_train_dataframes(data_info, split_tags=True):
    dataframes = {dataname: load_dataset(filepath, split_tags)
                  for dataname, filepath in
                  zip(data_info.training_sets, data_info.training_files)
                  }
    return dataframes

def get_test_dataframes(data_info):
    dataframes = {dataname: pandas.read_csv(filepath)
                  for dataname, filepath in
                  zip(data_info.test_sets, data_info.test_files)
                  }
    return dataframes


def save_training_data(data_info, data):
    for data_set, data_filepath in zip(data_info.training_sets, data_info.training_files):
        print(data_filepath)
        directory = dirname(data_filepath)
        if not os.path.exists(directory):
            os.makedirs(directory)
        data[data_set].to_csv(data_filepath, index=False)


def save_test_data(data_info, data):
    for data_set, data_filepath in zip(data_info.test_sets, data_info.test_files):
        print(data_filepath)
        data[data_set].to_csv(data_filepath, index=False)
