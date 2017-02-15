#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from functools import reduce
from itertools import islice, chain

import numpy as np
import pandas as pd
import logging
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from room007.data import info

logger = logging.getLogger()


def evaluate(expected, predicted):
    all_tags = (reduce(set.union, expected, set()) |
                reduce(set.union, predicted, set()))
    coder = MultiLabelBinarizer(sparse_output=True)
    coder.fit([all_tags])
    predicted_enc = coder.transform(predicted)
    expected_enc = coder.transform(expected)

    # this seems to co-relate with the official evaluation the best
    return (f1_score(predicted_enc, expected_enc, average='weighted') +
            f1_score(expected_enc, predicted_enc, average='weighted')) / 2


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
        logger.info('evaluating on {}'.format(eval_name))
        logger.info('creating train set')
        train_dataset = pd.concat(frame for name, frame in frames.items()
                                  if name != eval_name)
        logger.info('train set created')
        logger.info('learning on the train set')
        learner.fit(train_dataset)
        logger.info('lerning complete')
        logger.info('predicting on {}, test size is {}'.format(eval_name, len(eval_dataset)))
        predictions[eval_name] = learner.predict(eval_dataset)
        logger.info('done predicting')


    logger.info('calculating the f-scores')
    # Compute the scores.
    weights, scores, names = zip(*((len(frames[name]),
                            evaluate(frames[name]['tags'], prediction), name)
                            for name, prediction in predictions.items()))
    for score, name in zip(scores, names):
        logger.info('result for {} is {}'.format(name, score))
    return np.average(scores, 0, weights)


def test_oracle():
    data_info = info.CleanedData()
    data_frames = dict(
        islice(info.get_train_dataframes(data_info).items(), 0, 3))

    class Oracle(object):
        def fit(self, dataset):
            pass

        def predict(self, dataset):
            return dataset['tags']

    score = cross_validate(Oracle(), data_frames)
    print(score)
    assert score == 1.0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_oracle()
