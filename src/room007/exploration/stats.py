#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain
from os.path import basename, join, splitext
import re

import pandas as pd

from room007.data import info


TAG_STATS_IDX = pd.Index(('present', 'total'))


def word_freq(series):
    return Counter(chain.from_iterable(series.str.split(' ')))


def num_words(series):
    wc_series = series.str.split(' ').apply(len)
    return Counter(wc_series)


def tags_stat(frame, stat=num_words):
    return stat(frame['tags'])


def norm_tag(tag):
    return tag.replace('-', ' ')


def tag_in_text(tag, text):
    tag_re = '\\b{tag}\\b'.format(tag=re.escape(tag))
    return re.search(tag_re, text) is not None


def question_mentions_tag(qrec, tag):
    normed = norm_tag(tag)
    return (tag_in_text(normed, qrec['title']) or
            tag_in_text(normed, qrec['content']))


def rec_tag_stats(rec):
    num_tags_in_text = sum(map(partial(question_mentions_tag, rec),
                               rec['tags']))
    # XXX Would be cleaner to pull out the definition for this data type.
    return pd.Series((num_tags_in_text, len(rec['tags'])),
                     index=TAG_STATS_IDX,
                     dtype='i4')


def frame_tag_stats(frame):
    # TODO Do this splitting earlier.
    frame['tags'] = frame['tags'].str.split()
    return frame.apply(rec_tag_stats, 1).sum()



if __name__ == "__main__":
    overall_stats = pd.Series((0, 0),
                              index=TAG_STATS_IDX,
                              dtype='i4')
    for relpath in ('data/interim/biology.csv',
                    'data/interim/cooking.csv',
                    'data/interim/crypto.csv',
                    'data/interim/diy.csv',
                    'data/interim/robotics.csv',
                    'data/interim/travel.csv',
                    ):
        path = join(info.PROJECT_DIR, relpath)
        fname = splitext(basename(path))[0]
        print(fname)
        stats = frame_tag_stats(pd.read_csv(path))
        overall_stats += stats
        print(stats)
        print(stats['present'] / stats['total'])
        print('')
    print('overall')
    print(overall_stats)
    print(overall_stats['present'] / overall_stats['total'])

    # Prints:
    #
    # biology
    # present     7444
    # total      33129
    # dtype: int64
    # 0.224697395032
    #
    # cooking
    # present    19960
    # total      35542
    # dtype: int64
    # 0.561589105847
    #
    # crypto
    # present    12615
    # total      25484
    # dtype: int64
    # 0.495016480929
    #
    # diy
    # present    31299
    # total      59129
    # dtype: int64
    # 0.529334167667
    #
    # robotics
    # present    3317
    # total      6520
    # dtype: int64
    # 0.508742331288
    #
    # travel
    # present    25906
    # total      65334
    # dtype: int64
    # 0.396516362078
    #
    # overall
    # present    100541
    # total      225138
    # dtype: int64
    # 0.446574989562
