#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain
from os.path import basename, join, splitext
import re

import numpy
import pandas as pandas
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


from room007.data import info


TAG_STATS_IDX = pandas.Index(('present', 'total'))




def norm_tag(tag):
    return re.sub('-', ' ', tag)

def get_vectorizer(train):
    vectorizer = TfidfVectorizer(stop_words='english')
    train_features = vectorizer.fit_transform(train['titlecontent'])
    return vectorizer

def predict_per_category(dataframes):
    total_score = 0
    for fname, data in sorted(dataframes.items()):
        # split data, 0.2 arbitrary chosen
        train, test = train_test_split(data, test_size=0.2)
        score = learn_and_predict(train, test)
        print(fname)
        print(score)
        total_score += score
    avg_score = total_score/len(dataframes)
    print('total: {}'.format(avg_score))


def predict_cross_category(dataframes):
    total_score = 0
    for fname, test_data in sorted(dataframes.items()):
        train_data = pandas.concat([data for name, data in dataframes.items() if name!=fname], ignore_index=True)
        train_data, throw_away = train_test_split(train_data, test_size=0.50)
        print('start learning for {} {} {}'.format(fname, len(test_data), len(train_data)))
        score = learn_and_predict(train_data, test_data)
        print(score)
        total_score += score
    avg_score = total_score/len(dataframes)
    print('total: {}'.format(avg_score))

def learn_and_predict(train_data, test_data):
    vectorizer = get_vectorizer(train_data)
    score = predict_on_test_data(vectorizer, test_data)
    return score

def predict_on_test_data(vectorizer, test):
    feature_names = vectorizer.get_feature_names()
    all_content = [x for x in test['titlecontent']]
    all_tags  = [x for x in test['tags']]
    score  = 0
    for i, entry in enumerate(all_content):
        test_features = vectorizer.transform([entry])
        entry = test_features.toarray()[0]
        content = all_content[i]
        number_of_tags = len(all_tags[i])
        number_of_tags = 5
        prediction = [feature_names[i] for i, score in sorted(enumerate(entry), key=lambda t: t[1] * -1)[:number_of_tags]]
        diff_length =  len(prediction) - len(all_tags[i])
        if diff_length > 0:
            all_tags[i].extend(['']*diff_length)
        if diff_length < 0:
            prediction.extend(['']*(-1*diff_length))
        fscore = f1_score(all_tags[i], prediction, average='macro')
        score += fscore
    return(score/len(all_tags)*100)

def remove_numbers(text):
    return re.sub('[0-9]', '', text)

def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)

def main():
    data_info = info.CleanedData()
    dataframes = info.get_train_dataframes(data_info)
    for fname, data in sorted(dataframes.items()):
        data['tags'] = data['tags'].str.split()
        data['titlecontent'] = data['title'] + data['content']
        do_extra_cleaning(data)

    predict_cross_category(dataframes)
    #predict_per_category(dataframes)

if __name__ == "__main__":
    main()
