#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain, filterfalse, islice

import pandas as pd

from room007.data import info
from room007.exploration.common import *


def extract_tag_features(tag, suf_len=3):
    """Transforms a tag to a string consisting only of its suffices of length
    `suf_len` (or less if the word is shorter)."""
    return '-'.join(word[-suf_len:] for word in tag.split('-'))


# TODO Move to data.
def concat_frames(n=6):
    data_info = info.CleanedData()
    dataframes = dict(
        islice(info.get_train_dataframes(data_info).items(), 0, n))
    frame = pd.concat(dataframes.values())
    return frame


def drop_mentioned_tags(qrec):
    """destructive!"""
    qrec['tags'] = list(filterfalse(partial(question_mentions_tag, qrec),
                                    qrec['tags']))
    return qrec


def list_suffices(frame, suf_len=3, n=42):
    """Prints a table with frequencies of the most common tag suffices."""
    tag_characteristics = Counter(map(partial(extract_tag_features,
                                              suf_len=suf_len),
                                      chain.from_iterable(frame['tags'])))
    print('\n'.join('{: 6d} {}'.format(count, feat)
                    for feat, count in tag_characteristics.most_common(n)))


# nonmen = non-mentioned tags'
def list_nonmen_suffices(frame, *args, **kwargs):
    frame.apply(drop_mentioned_tags, 1)
    return list_suffices(frame, *args, **kwargs)


if __name__ == "__main__":
    frame = concat_frames(2)
    # list_suffices(frame)
    list_nonmen_suffices(frame)
