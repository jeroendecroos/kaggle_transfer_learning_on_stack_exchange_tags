# -*- coding: utf-8 -*-
import os
from os.path import dirname, join, realpath

PROJECT_DIR = dirname(dirname(dirname(dirname(realpath(__file__)))))


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
        self.header = ['id', 'title', 'content', 'tags']
        self.data_dir = ''

    def training_files(self):
        for f in self._iterate_files(self.training_sets):
            yield f

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
