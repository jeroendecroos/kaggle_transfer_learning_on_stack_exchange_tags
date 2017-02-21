#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import logging

from room007.logging import loggingmgr
from room007.models import model, advanced_train_and_predict
from room007.models.flect_word_tagifier import FlectWordTagifier
from room007.util import time_function

loggingmgr.set_up()
logger = logging.getLogger(__name__)

_ADVANCED_ARGER = advanced_train_and_predict.ArgumentParser()


class Predictor(model.Predictor):

    def __init__(self, options):
        # Create the tagifier.
        self.tagifier = FlectWordTagifier()
        # Create the extractor.
        extractor_class_name = (
            options.pop('model', _ADVANCED_ARGER.get_default('model')))
        _, self.extractor = model.create_predictor(extractor_class_name)
        self.set_options(options)

    def get_options(self):
        # TODO Factor the 'model' option in to self.extractor's options.
        return self.extractor.get_options()

    def set_options(self, options):
        # There are no options to set on the tagifier (as yet).
        # Set options on the extractor.
        adjusted = dict(options)
        adjusted['label-fun'] = self.tagifier.label_words
        self.extractor.set_options(adjusted)

    @time_function(logger, True)
    def fit(self, train_data):
        self.extractor.fit(train_data)
        with_words = train_data.assign(
            extracted_words=self.extractor.predict(train_data))
        self.tagifier.fit(with_words)

    def predict(self, test_dataframe):
        word_sets = self.extractor.predict(test_dataframe)
        tag_sets = self.tagifier.predict(word_sets)
        return tag_sets
