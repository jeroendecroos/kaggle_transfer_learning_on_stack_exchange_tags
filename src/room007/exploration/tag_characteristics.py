#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from itertools import chain, islice

import pandas as pd

from room007.data import info


def extract_tag_features(tag, suf_len=3):
    """Transforms a tag to a string consisting only of its suffices of length
    `suf_len` (or less if the word is shorter)."""
    return '-'.join(word[-suf_len:] for word in tag.split('-'))


def preview():
    """Prints a table with frequencies of the most common tag suffices."""
    data_info = info.CleanedData()
    dataframes = dict(
        islice(info.get_train_dataframes(data_info).items(), 0, 6))
    frame = pd.concat(dataframes.values())
    tag_characteristics = Counter(map(extract_tag_features,
                                      chain.from_iterable(frame['tags'])))
    print('\n'.join('{: 6d} {}'.format(count, feat)
                    for feat, count in tag_characteristics.most_common(42)))


if __name__ == "__main__":
    preview()
