# vim: set fileencoding=utf-8 :

import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model


from pandas import DataFrame
import nltk
import string

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

from room007.models import model
from room007.data import info
from room007.data import feature_data


stop_words = nltk.corpus.stopwords.words('english') + [x for x in string.printable]

@feature_data.FeatureManager
def _is_in_question(data):
    def is_in_question(row):
        features = []
        question = 0
        for word in row.split()[::-1]:
            if word in '.:;!?':
                question = int(word == '?')
            if word not in stop_words:
                features.append(question)
        return features[::-1]
    return data['titlecontent'].apply(
        is_in_question
        )

@feature_data.FeatureManager
def _title_or_content(data):
    return data.apply(
        lambda row: [1] * len(row['title_non_stop_words']) + [0] * len(row['content_non_stop_words']),
        axis=1)


def _add_number_of_non_stop_words(data):
    split_row = lambda row: [x for x in row.split() if x not in stop_words]
    data['title_non_stop_words'] = data['title'].apply(split_row)
    data['content_non_stop_words'] = data['content'].apply(split_row)

@feature_data.FeatureManager
def _times_word_in(data, column):
    return data.apply(
        lambda row: [row[column].split().count(word)
                     for word in row['titlecontent'].split()
                     if word not in stop_words],
        axis=1)

class Features(model.Features):
    def __init__(self, functional_test=False, changes=False):
        self.functional_test = functional_test
        self.tf_idf_vectorizer = None
        self.changes = changes
        self.save = False

    def fit(self, train_data):
        self._train_tf_idf_vectorizer(train_data)

    def transform(self, data):
        features = self._get_features_per_row(data)
        for i, feat in enumerate(features):
            features[i] = list(itertools.chain.from_iterable(feat))
        feats = tuple(zip(*features))
        return feats

    def _get_features_per_row(self, data):
        features = self._get_data_independent_features(data)
        features.append(self._get_tf_idf_features_per_word(data))
        if self.changes:
            #  add features here you want to see impact of
            pass
        return features

    def _get_data_independent_features(self, data):
        _add_number_of_non_stop_words(data)
        return [
                _times_word_in(data, 'title'),
                _times_word_in(data, 'content'),
                _is_in_question(data),
                _title_or_content(data),
        ]


    def _train_tf_idf_vectorizer(self, data):
        self.tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tf_idf_vectorizer.fit(data['titlecontent'])

    def _get_tf_idf_features_per_word(self, train_data):
        tf_idf_data = self.tf_idf_vectorizer.transform(train_data['titlecontent'])
        train_data['index'] = range(tf_idf_data.shape[0])
        voc = self.tf_idf_vectorizer.vocabulary_
        features = train_data.apply(
            lambda row: [tf_idf_data[row['index'], voc.get(word)]
                         if voc.get(word) else 1
                         for word in row['titlecontent'].split()
                         if word not in stop_words],
            axis=1)
        return features

    def _write_some_features(self, features, keys):
        with open('debug_files/feats_per_word', 'wt') as outstream:
            outstream.write(','.join(keys))
            for feat in features:
                outstream.write(','.join(f for f in feat) + '\n')


class OptionsSetter(model.OptionsSetter):
    def __init__(self):
        self.options = {}
        self.options['classifier'] = model.Option(
            {#"Nearest Neighbors": KNeighborsClassifier(3),
             # "Linear SVM": SVC(kernel="linear", C=0.025),
             # "RBF SVM": SVC(gamma=2, C=1),
             # "Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), # too slow?
             # "Decision Tree": DecisionTreeClassifier(max_depth=5),
             # "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
             # "Neural Net": MLPClassifier(alpha=1),
             # "AdaBoost": AdaBoostClassifier(),
             "Naive Bayes": GaussianNB(),
             "Logistic Regression": LogisticRegression(class_weight='balanced'),
             "QDA": QuadraticDiscriminantAnalysis()
             }, "Logistic Regression")
        self.options['changes'] = model.Option(
            {'True': True,
             'False': False,
             }, 'True')


class Predictor(model.Predictor):
    OptionsSetter = OptionsSetter

    def __init__(self, *args, **kwargs):
        self.functional_test = kwargs.get('functional-test', False)
        self.classifier = None
        self.changes = None
        self.set_options(kwargs)

    def fit(self, train_data):
        print('start fitting')
        self._fit(train_data)

    def predict(self, test_dataframe):
        print('start predicting')
        predictions = []
        tag_predictions = self.classifier.predict(
                self.feature_creator.transform(test_dataframe)
                )
        line = 0
        for i in range(len(test_dataframe)):
            entry = test_dataframe[i:i+1]
            words = [w for w in entry.titlecontent.values[0].split() if w not in stop_words]
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
        print("get features")
        self.feature_creator = Features(changes=self.changes)
        self.feature_creator.fit(train_data)
        features = self.feature_creator.transform(train_data)
        truths = self._get_truths_per_word(train_data)
        print("learning")
        self._learn(features, truths)
        print("finished learning")

    def _learn(self, features, truths):
        self.classifier.fit(features, truths)

    def _get_truths_per_word(self, train_data):
        truths = []
        for i, titlecontent in enumerate(train_data.titlecontent.values):
            words = [w for w in titlecontent.split() if w not in stop_words]
            tags = train_data.tags.values[i]
            truths.extend(w in tags for w in words)
        return truths

    def _predict_for_one_entry(self, entry):
        features = self.feature_creator.transform(entry)
        tag_predictions = self.classifier.predict(features)
        words = [w for w in entry.titlecontent.values[0].split() if w not in stop_words]
        predictions = set()
        for pred, word in zip(tag_predictions, words):
            if pred:
                predictions.add(word)
        return list(predictions)

