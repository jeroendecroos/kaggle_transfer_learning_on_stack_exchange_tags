import re


def remove_numbers(text):
    return re.sub('[0-9]', '', text)


def get_sample_size(test):
    return 0.001 if test else 0.01


def process_title_content(data):
    data['titlecontent'] = (
        data['title'] + ' ' + data['content']
    ).map(remove_numbers)


def sample_data(data, size):
    return {name: frame.sample(frac=size)
            for name, frame in data.items()}


class Processor(object):
    def __init__(self, *args):
        self._args = args
        self._sample_size = 1
        if 'small' in args:
            self._sample_size = 0.01
        if 'very-small' in args:
            self._sample_size = 0.001

    def process(self, data):
        data = sample_data(data, self._sample_size)
        for _, frame in data.items():
            process_title_content(frame)
        return data
