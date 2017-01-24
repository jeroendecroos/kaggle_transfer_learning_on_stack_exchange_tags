#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from functools import reduce
from itertools import islice

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from room007.data import info


def evaluate(expected, predicted):
    all_tags = (reduce(set.union, expected, set()) |
                reduce(set.union, predicted, set()))
    coder = MultiLabelBinarizer()
    coder.fit([all_tags])
    return f1_score(coder.transform(expected),
                    coder.transform(predicted),
                    average='weighted')


def cross_validate(learner, frames):
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
    for eval_name, eval_dataset in frames.items():
        train_dataset = pd.concat(frame for name, frame in frames.items()
                                  if name != eval_name)
        learner.fit(train_dataset)
        predictions[eval_name] = learner.predict(eval_dataset)
    # Compute the scores.
    weights, scores = zip(*((len(frames[name]),
                             evaluate(frames[name]['tags'], preds))
                            for name, preds in predictions.items()))
    return np.average(scores, 0, weights)


def test_oracle():
    data_info = info.CleanedData()
    dataframes = dict(
        islice(info.get_train_dataframes(data_info).items(), 0, 3))

    class Oracle(object):
        def fit(self, dataset):
            pass

        def predict(self, dataset):
            return dataset['tags']

    score = cross_validate(Oracle(), dataframes)
    print(score)
    assert score == 1.0


if __name__ == "__main__":
    test_oracle()
