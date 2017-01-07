# -*- coding: utf-8 -*-
import unittest
import info

class TestData(unittest.TestCase):
    pass

class TestTrainingFiles(TestData):
    def setUp(self):
        self.data = info.Data()
        self.data.extension = '.ext'
        self.data.data_dir = '/none/'
        self.prop = self.data.training_files

    def test_no_trainingfiles(self):
        self.data.training_sets = []
        self.assertItemsEqual(self.prop, [])

    def test_one_trainingfile(self):
        self.data.training_sets = ['1']
        self.assertItemsEqual(self.prop, ['/none/1.ext'])

    def test_multiple_trainingfiles(self):
        self.data.training_sets = ['1', '2', '3']
        expected = ['/none/1.ext',
                    '/none/2.ext',
                    '/none/3.ext',
                    ]
        self.assertItemsEqual(self.prop, expected)


class TestTestFiles(TestData):
    def setUp(self):
        self.data = info.Data()
        self.data.extension = '.ext'
        self.data.data_dir = '/none/'
        self.prop = self.data.test_files

    def test_no_testfiles(self):
        self.data.test_sets = []
        self.assertItemsEqual(self.prop, [])

    def test_one_testfile(self):
        self.data.test_sets = ['1']
        self.assertItemsEqual(self.prop, ['/none/1.ext'])

    def test_multiple_testfiles(self):
        self.data.test_sets = ['1', '2', '3']
        expected = ['/none/1.ext',
                    '/none/2.ext',
                    '/none/3.ext',
                    ]
        self.assertItemsEqual(self.prop, expected)
