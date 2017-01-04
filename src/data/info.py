# -*- coding: utf-8 -*-
import os


def project_dir():
    return os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)


class Data(object):
    def __init__(self):
        self.training_sets = []
        self.test_sets = ['']
        self.data_dir = os.path.join(project_dir(), 'data', '')
        self.extension = ''
        self.header = []

    def training_files(self):
        for f in self._iterate_files(self.training_sets):
            yield f

    def test_files(self):
        for f in self._iterate_files(self.test_sets):
            yield f

    def _iterate_files(self, data_sets):
        for filebase in data_sets:
            filename = filebase + self.extension
            filepath = os.path.join(self.data_dir, filename)
            yield filepath


class RawData(Data):
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
        self.data_dir = os.path.join(project_dir(), 'data', 'raw')
        self.extension = '.csv'
        self.header = ['id', 'title', 'content', 'tags']


class CleanedData(Data):
    def __init__(self):
        self.training_sets = [
                'merged',
                ]
        self.test_sets = ['test']
        self.data_dir = os.path.join(project_dir(), 'data', 'raw')
        self.extension = '.csv'
        self.header = ['id', 'title', 'content', 'tags']
