#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain, filterfalse, islice
from operator import itemgetter

import editdistance as ed
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
    orig_tags = list(qrec['tags'])
    qrec['tags'] = list(filterfalse(partial(question_mentions_tag, qrec),
                                    qrec['tags']))
    # if qrec['tags'] != orig_tags:
    #     import ipdb
    #     ipdb.set_trace()
    return qrec


def list_suffices(frame, suf_len=3, n=42):
    """Prints a table with frequencies of the most common tag suffices."""
    tag_characteristics = Counter(map(partial(extract_tag_features,
                                              suf_len=suf_len),
                                      chain.from_iterable(frame['tags'])))
    print('\n'.join('{: 6d} {}'.format(count, feat)
                    for feat, count in tag_characteristics.most_common(n)))


def iter_recs(frame):
    # DataFrame.iterrows returns tuples (index, Series), so extract only the
    # Series part.
    return map(itemgetter(1), frame.iterrows())


def min_ed_and_word(sentence, tag):
    # FIXME: Split the sentence earlier and only once.
    try:
        return min(((ed.eval(word, tag), word) for word in sentence.split()),
                   key=itemgetter(0))
    except ValueError:
        return None, None


def find_nearest_phrase(qrec, tag):
    # TODO Implement for multi-word tags.
    if '-' in tag:
        return None
    title_ed, title_word = min_ed_and_word(qrec['title'], tag)
    content_ed, content_word = min_ed_and_word(qrec['content'], tag)
    if title_ed is None and content_ed is None:
        return None
    if title_ed is None:
        return content_word, tag
    if content_ed is None:
        return title_word, tag
    nearest_word = title_word if title_ed <= content_ed else content_word
    # TODO Don't just return the two words, instead, extract their relationship
    # somehow.
    return nearest_word, tag


def find_nearest_phrases(qrec):
    return filter(None, map(partial(find_nearest_phrase, qrec), qrec['tags']))


def list_nearest_phrases(frame, n=42):
    nearest_phrases_ctr = Counter(chain.from_iterable(
                    map(find_nearest_phrases, iter_recs(frame))))
    print('\n'.join('{: 6d} {}'.format(count, feat)
                    for feat, count in nearest_phrases_ctr.most_common(n)))


if __name__ == "__main__":
    frame = concat_frames(2)
    frame = frame.apply(drop_mentioned_tags, 1)
    list_nearest_phrases(frame)

    # list_nearest_phrases after dropping mentioned tags outputs:
    # 3247 ('visa', 'visas')
    #  939 ('us', 'usa')
    #  550 ('passport', 'passports')
    #  407 ('train', 'trains')
    #  341 ('substitute', 'substitutions')
    #  314 ('airport', 'airports')
    #  235 ('ticket', 'tickets')
    #  221 ('layover', 'layovers')
    #  187 ('but', 'budget')
    #  161 ('a', 'usa')
    #  146 ('bus', 'buses')
    #  146 ('making', 'baking')
    #  145 ('border', 'borders')
    #  139 ('get', 'budget')
    #  134 ('safe', 'safety')
    #  133 ('booking', 'bookings')
    #  121 ('baggage', 'luggage')
    #  112 ('phone', 'cellphones')
    #  112 ('hotel', 'hotels')
    #  112 ('passport', 'paperwork')
    #  104 ('egg', 'eggs')
    #  102 ('freeze', 'freezing')
    #   96 ('is', 'usa')
    #   89 ('airline', 'airlines')
    #   85 ('bake', 'baking')
    #   76 ('use', 'usa')
    #   75 ('grill', 'grilling')
    #   73 ('for', 'flavor')
    #   73 ('freezer', 'freezing')
    #   73 ('clean', 'cleaning')
    #   71 ('substitution', 'substitutions')
    #   67 ('the', 'lhr')
    #   65 ('ferry', 'ferries')
    #   65 ('cruise', 'cruising')
    #   65 ('taxi', 'taxis')
    #   61 ('drive', 'driving')
    #   61 ('fat', 'fats')
    #   61 ('substituting', 'substitutions')
    #   57 ('that', 'transit')
    #   57 ('book', 'bookings')
    #   56 ('indian', 'india')
    #   55 ('in', 'india')
    # ...So the most common transformations, and perhaps those that make sense
    # to reproduce, are the plural and changing to the gerund.
