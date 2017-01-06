#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from itertools import chain


def word_freq(series):
    return Counter(chain.from_iterable(series.str.split(' ')))


def num_words(series):
    wc_series = series.str.split(' ').apply(len)
    return Counter(wc_series)


def tags_stat(frame, stat=num_words):
    return stat(frame['tags'])
