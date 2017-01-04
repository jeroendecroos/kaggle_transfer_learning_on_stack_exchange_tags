# -*- coding: utf-8 -*-


def project_dir()
    return os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)


class RawData(object):
    training_files = [
            'biology',
            'cooking',
            'crypto',
            'diy',
            'robotics',
            'travel',
    testset = 'test'
    data_dir = os.path.join(project_dir(), 'data', 'raw')
    extension = '.csv'
    header = ['id', 'title', 'content', 'tags']

    def training_files(self):
        for filebase in self.training_files:
            filename = filebase + self.extension
            filepath = os.path.join(self.data_dir, filename)
            yield filepath
