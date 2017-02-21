#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :
import logging

from room007.logging import loggingmgr

loggingmgr.set_up()
logger = logging.getLogger(__name__)


class FlectWordTagifier(object):

    def label_words(self, words, tags):
        # TODO
        return ((word in tags) for word in words)

    def fit(self, data):
        # TODO
        pass

    def predict(self, word_sets):
        # TODO
        return word_sets
