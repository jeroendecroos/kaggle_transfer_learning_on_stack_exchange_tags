#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter, namedtuple
from functools import partial
from itertools import chain
from os.path import basename, join, splitext
import re

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from room007.data import info


def evaluate(expected, predicted):
    # TODO Separate the tags into lists or so...
    return f1_score(expected, predicted, average='weighted')


def cross_validate(learner, dataframes):
    """Runs the provided learner (compliant to the Scikit protocol, fitting to
    a dataset using the fit(ds) method and predicting for a dataset using the
    predict(ds) method) on all sets of training data except for one, in sequel
    dropping the first, second... dataset, and capturing predictions of the
    model trained this way on the remaining dataset.

    Returns the average over the runs of cross-validation of scores
    achieved in them on the held-out dataset.

    """

    predictions = dict()
    # Run the cross-validation rounds.
    # XXX Could be easily parallelized.
    for eval_name, eval_dataset in dataframes.items():
        train_dataset = pd.concat(frame for name, frame in dataframes.items()
                                  if name != eval_name)
        model = learner.fit(train_dataset)
        predictions[eval_name] = learner.predict(eval_dataset)
    # Compute the scores.
    return np.mean(evaluate(dataframes[name]['tags'], preds)
                   for name, preds in predictions.items())),


if __name__ == "__main__":
    data_info = info.CleanedData()
    dataframes = info.get_train_dataframes(data_info)
