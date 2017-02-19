# vim: set fileencoding=utf-8 :

from itertools import chain
import logging
import string

logger = logging.getLogger(__name__)

import nltk
from sklearn import linear_model
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from room007.models import model

from room007.models.train_and_predict import time_function


stop_words = (set(nltk.corpus.stopwords.words('english'))
              .union(string.printable))

def split_row(row):
    return [x for x in row.split() if x not in stop_words]


class Features(object):
    def __init__(self, functional_test=False, changes=False):
        self.functional_test = functional_test
        self.tf_idf_vectorizer = None
        self.changes = changes
        self._changes_int = 1 if changes else 0

    def fit(self, train_data):
        self._train_tf_idf_vectorizer(train_data)

    def transform(self, data):
        self._add_texts_wo_stop_words(data)
        features = [
                self._get_tf_idf_features_per_word(data),
                self._times_word_in(data, 'title'),
                self._times_word_in(data, 'content'),
                self._is_in_question(data),
        ]
        #   if self.changes:
        features += [self._title_or_content(data)]
        feats = tuple(zip(*features))
        return feats

    def _title_or_content(self, data):
        return list(chain(*data.apply(
            lambda row: [1] * len(row['title_non_stop_words']) +
                        [0] * len(row['content_non_stop_words']),
            axis=1)))

    def _add_texts_wo_stop_words(self, data):
        for cmn in ('content', 'title', 'titlecontent'):
            data[cmn + '_non_stop_words'] = data[cmn].apply(split_row)

    @time_function(True)
    def _is_in_question(self, data):
        def is_in_question(row):
            features = []
            question = 0
            for word in row.split()[::-1]:
                if word in '.:;!?':
                    question = int(word == '?')
                if word not in stop_words:
                    features.append(question)
            return features[::-1]
        return list(chain(*data['titlecontent'].apply(
            is_in_question
            )))

    @time_function(True)
    def _times_word_in(self, data, column):
        return list(chain(*data.apply(
            lambda row: [row[column].split().count(word)
                         for word in row['titlecontent'].split()
                         if word not in stop_words], axis=1)
            ))

    def _train_tf_idf_vectorizer(self, data):
        self.tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tf_idf_vectorizer.fit(data['titlecontent'])
        #if self.functional_test:
        #    self._write_example_it_idf_features(data)

    @time_function(True)
    def _get_tf_idf_features_per_word(self, train_data):
        logger.debug("vectorizing")
        tf_idf_data = self.tf_idf_vectorizer.transform(
            train_data['titlecontent'])
        logger.debug("done vectorizing")
        # XXX Why is this needed? I see pandas' Index is weirdly shuffled...
        # why?
        train_data['index'] = range(tf_idf_data.shape[0])
        voc = self.tf_idf_vectorizer.vocabulary_

        logger.debug("listing features")
        train_recs = train_data.itertuples()
        feats_per_q = (((tf_idf_data[rec.index, voc[word]]
                         if word in voc else self._changes_int)
                        for word in rec.titlecontent_non_stop_words)
                       for rec in train_recs)
        features = list(chain.from_iterable(feats_per_q))
        # A alternative, less performant notation:
        # features = list(chain.from_iterable(train_data.apply(
        #     lambda row: [(tf_idf_data[row['index'], voc[word]]
        #                   if word in voc else self._changes_int)
        #                  for word in row['titlecontent_non_stop_words']],
        #     axis=1)))
        logger.debug("done listing features")

        return features

    def _write_some_features(self, features, keys):
        with open('debug_files/feats_per_word', 'wt') as outstream:
            outstream.write(','.join(keys))
            for feat in features:
                outstream.write(','.join(f for f in feat) + '\n')


class Option(object):
    def __init__(self, options, default):
        self.choices = options
        self.default = default


class OptionsSetter(model.OptionsSetter):
    def __init__(self):
        self.options = {}
        self.options['classifier'] = Option(
            {"Nearest Neighbors": KNeighborsClassifier(3),
             # "Linear SVM": SVC(kernel="linear", C=0.025),
             # "RBF SVM": SVC(gamma=2, C=1),
             # "Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), # too slow?
             "Decision Tree": DecisionTreeClassifier(max_depth=5),
             "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
             # "Neural Net": MLPClassifier(alpha=1),
             "AdaBoost": AdaBoostClassifier(),
             "Naive Bayes": GaussianNB(),
             "Logistic Regression": LogisticRegression(class_weight='balanced'),
             "QDA": QuadraticDiscriminantAnalysis()
             }, "Logistic Regression")
        self.options['changes'] = Option(
            {'True': True,
             'False': False,
             }, 'True')


def label_words(words, tags):
    return ((word in tags) for word in words)


class Predictor(model.Predictor):
    OptionsSetter = OptionsSetter

    def __init__(self, *args, **kwargs):
        self.functional_test = kwargs.get('functional-test', False)
        self.classifier = None
        self.changes = None
        self.set_options(kwargs)

    def fit(self, train_data):
        logger.info('start fitting')
        self._fit(train_data)

    def predict(self, test_dataframe):
        logger.info('start predicting')
        predictions = []
        tag_predictions = self.classifier.predict(
                self.feature_creator.transform(test_dataframe)
                )
        line = 0
        for i in range(len(test_dataframe)):
            entry = test_dataframe[i:i+1]
            words = [w for w in entry.titlecontent.values[0].split()
                     if w not in stop_words]
            n_words = len(words)
            tag_predictions[line:line+n_words]
            line += n_words
            prediction = set()
            for pred, word in zip(tag_predictions[line:line+n_words], words):
                if pred:
                    prediction.add(word)
            prediction = list(prediction)
            predictions.append(prediction)
        return predictions

    def _fit(self, train_data):
        self.feature_creator = Features(changes=self.changes)
        logger.info("fitting features")
        self.feature_creator.fit(train_data)
        logger.info("transforming features")
        features = self.feature_creator.transform(train_data)
        logger.info("getting truths")
        truths = self._get_truths_per_word(train_data)
        logger.info("learning")
        self._learn(features, truths)
        logger.info("finished learning")

    def _learn(self, features, truths):
        self.classifier.fit(features, truths)

    def _get_truths_per_word(self, train_data):
        labels_per_q = map(label_words,
                           train_data.titlecontent_non_stop_words.values,
                           train_data.tags.values)
        truths = chain.from_iterable(labels_per_q)
        return list(truths)

    def _predict_for_one_entry(self, entry):
        features = self.feature_creator.transform(entry)
        tag_predictions = self.classifier.predict(features)
        words = [w for w in entry.titlecontent.values[0].split()
                 if w not in stop_words]
        predictions = {word for (pred, word) in zip(tag_predictions, words)
                       if pred}
        return list(predictions)
