# vim: set fileencoding=utf-8 :

import re
import logging

from room007.logging import loggingmgr

loggingmgr.set_up()
logger = logging.getLogger(__name__)


def remove_numbers(text):
    return re.sub('[0-9]+([.,][0-9]+)*', '', text)


def clean(data):
    for cmn in 'title', 'content', 'titlecontent':
        if cmn in data:
            data[cmn] = data[cmn].map(remove_numbers)
    return data


def expand(data):
    data['titlecontent'] = data['title'] + ' ' + data['content']
    return data


def preprocess(data):
    return expand(clean(data))
