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
    return frame.apply(rec_tag_stats, 1).sum()

def tag_available(frame):
    frame['scontent'] = frame['content'].str.split()
    frame['stitle'] = frame['title'].str.split()
    all_tags = set(chain(*frame['tags']))
    all_words = chain(chain(*frame['stitle']), chain(*frame['scontent']))
    all_words = set(all_words)
    return pd.Series((len(all_tags & all_words), len(all_tags)),
                     index=TAG_STATS_IDX,
                     dtype='i4')



if __name__ == "__main__":
    overall_stats = pd.Series((0, 0),
                              index=TAG_STATS_IDX,
                              dtype='i4')
    overall_stats2 = pd.Series((0, 0),
                              index=TAG_STATS_IDX,
                              dtype='i4')
    data_info = info.CleanedData()
    dataframes = info.get_train_dataframes(data_info)
    for fname, data in dataframes.items():
        print(fname)
        data['tags'] = data['tags'].str.split()
        stats = tag_available(data)
        stats2 = frame_tag_stats(data)
        overall_stats += stats
        overall_stats2 += stats2
        print('together')
        print(stats)
        print(stats['present'] / stats['total'])
        print('seperate')
        print(stats2)
        print(stats2['present'] / stats2['total'])
        print('')
    print('overall')
    print('together')
    print(overall_stats)
    print(overall_stats['present'] / overall_stats['total'])
    print('seperate')
    print(overall_stats2)
    print(overall_stats2['present'] / overall_stats2['total'])

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
