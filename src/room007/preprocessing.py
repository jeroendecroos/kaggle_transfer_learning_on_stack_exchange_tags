# vim: set fileencoding=utf-8 :

import logging
import re
import string

import nltk

from room007.logging import loggingmgr

loggingmgr.set_up()
logger = logging.getLogger(__name__)

###############################################################################
#                                  Splitting                                  #
###############################################################################

STOP_WORDS = (set(nltk.corpus.stopwords.words('english'))
              .union(string.printable))


def tokenize(sentence):
    return [token for token in sentence.split() if token not in STOP_WORDS]

###############################################################################
#                                  Cleaning                                   #
###############################################################################

def remove_numbers(text):
    return re.sub('[0-9]+([.,][0-9]+)*', '', text)


def clean(data):
    for cmn in 'title', 'content', 'titlecontent':
        if cmn in data:
            data[cmn] = data[cmn].map(remove_numbers)
    return data

###############################################################################
#                                  Combining                                  #
###############################################################################

def add_tokenized(data):
    for cmn in 'title', 'content', 'titlecontent':
        if cmn in data:
            data[cmn + '_non_stop_words'] = data[cmn].apply(tokenize)
    return data


def expand(data):
    add_tokenized(data)
    data['titlecontent'] = data['title'] + ' ' + data['content']
    data['titlecontent_non_stop_words'] = (
        data['title_non_stop_words'] + data['content_non_stop_words'])
    return data

###############################################################################
#                            Convenience function                             #
###############################################################################

def preprocess(data):
    return expand(clean(data))
