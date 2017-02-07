# vim: set fileencoding=utf-8 :

import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import linear_model

import nltk
import string

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression



stop_words = nltk.corpus.stopwords.words('english') + [x for x in string.printable]


class Features(object):
    def __init__(self, functional_test=False, changes=False):
        self.functional_test = functional_test
        self.tf_idf_vectorizer = None
        self.changes=changes

    def fit(self, train_data):
        self._train_tf_idf_vectorizer(train_data)

    def transform(self, data):
        self._add_number_of_non_stop_words(data)
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
        return list(itertools.chain(*data.apply(
            lambda row: [1] * len(row['title_non_stop_words']) + [0] * len(row['content_non_stop_words']),
            axis=1)))


    def _add_number_of_non_stop_words(self, data):
        split_row = lambda row: [x for x in row.split() if x not in stop_words]
        data['title_non_stop_words'] = data['title'].apply(split_row)
        data['content_non_stop_words'] = data['content'].apply(split_row)


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
        return list(itertools.chain(*data['titlecontent'].apply(
            is_in_question
            )))

    def _times_word_in(self, data, column):
        return list(itertools.chain(*data.apply(
            lambda row: [row[column].split().count(word)
                         for word in row['titlecontent'].split()
                         if word not in stop_words], axis=1)
            ))


    def _train_tf_idf_vectorizer(self, data):
        self.tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        self.tf_idf_vectorizer.fit(data['titlecontent'])
        #if self.functional_test:
        #    self._write_example_it_idf_features(data)


    def _get_tf_idf_features_per_word(self, train_data):
        tf_idf_data = self.tf_idf_vectorizer.transform(train_data['titlecontent'])
        train_data['index'] = range(tf_idf_data.shape[0])
        voc = self.tf_idf_vectorizer.vocabulary_
        features = list(itertools.chain(*train_data.apply(
            lambda row: [tf_idf_data[row['index'], voc.get(word)]
                         if voc.get(word) else (0 if not self.changes else 1)
                         for word in row['titlecontent'].split()
                         if word not in stop_words], axis=1)
            ))
        return features


    def _write_some_features(self, features, keys):
        with open('debug_files/feats_per_word', 'wt') as outstream:
            outstream.write(','.join(keys))
            for feat in features:
                outstream.write(','.join(f for f in feat) + '\n')


class Predictor(object):
    classifiers = {    ## would be better to add the names to this
        "Nearest Neighbors": KNeighborsClassifier(3),
    #    "Linear SVM": SVC(kernel="linear", C=0.025),
    #    "RBF SVM": SVC(gamma=2, C=1),
        #"Gaussian Process":  GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True), # too slow?
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    #    "Neural Net": MLPClassifier(alpha=1),
        "AdaBoost": AdaBoostClassifier(),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(class_weight='balanced'),
        "QDA": QuadraticDiscriminantAnalysis()
    }
    def __init__(self, *args, **kwargs):
        classifier_name = kwargs.get('classifier_name', 'Logistic Regression')
        self.classifier = self.classifiers[classifier_name]
        self.functional_test = kwargs.get('functional-test', False)
        self.changes = kwargs.get('changes', False)

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

