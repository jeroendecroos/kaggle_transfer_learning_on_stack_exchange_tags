#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain
import re
import argparse

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
       # train_data, throw_away = train_test_split(train_data, test_size=0.50)
        print('start learning for {} {} {}'.format(fname, len(test_data), len(train_data)))
        score = learn_and_predict(train_data, test_data)
        print(score)
        total_score += score
    avg_score = total_score/len(dataframes)
    print('total: {}'.format(avg_score))

def predict_cross_category_save_result(dataframes, test_data):
    train_data = pandas.concat([data for name, data in dataframes.items()], ignore_index=True)
    #train_data, throw_away = train_test_split(train_data, test_size=0.99)
    print('start learning for {} {} {}'.format('physics', len(test_data), len(train_data)))
    learn_and_predict(train_data, test_data)

def learn_and_predict(train_data, test_data):
    vectorizer = get_vectorizer(train_data)
    score = predict_on_test_data(vectorizer, test_data)
    return score


def predict_on_test_data(vectorizer, test):
    feature_names = vectorizer.get_feature_names()
    number_of_tags = 2
    all_content = [(i-1, x) for (x, i) in zip(test['titlecontent'], test['id'])]
    if 'tags' in test:
        all_tags = [x for x in test['tags']]
    else:
        all_tags = [['blabla'] for x in test['id']]
    score = 0
    f = open('test.out.csv', 'w')
    f.write('"id","tags"\n')
    for i, entry in enumerate(all_content):
        j, entry = entry
        test_features = vectorizer.transform([entry])
        entry = test_features.toarray()[0]
        prediction = [feature_names[i] for i, score in sorted(enumerate(entry), key=lambda t: t[1] * -1)[:number_of_tags]]
        f.write('"{}","{}"\n'.format(i, ' '.join(prediction)))
        diff_length = len(prediction) - len(all_tags[i])
        if diff_length > 0:
            all_tags[i].extend(['']*diff_length)
        if diff_length < 0:
            prediction.extend(['']*(-1*diff_length))
        fscore = f1_score(all_tags[i], prediction, average='macro')
        score += fscore
    return(score/len(all_tags)*100)
    f.close()

def remove_numbers(text):
    return re.sub('[0-9]', '', text)

def do_extra_cleaning(data):
    data['titlecontent'] = data['titlecontent'].map(remove_numbers)

def apply_preprocessing(dataframes):
    for fname, data in sorted(dataframes.items()):
        if 'tags' in data:
            data['tags'] = data['tags'].str.split()
        data['titlecontent'] = data['title'] + data['content']
        do_extra_cleaning(data)

def get_arguments():
    parser = argparse.ArgumentParser(description='Predict with it-idf.')
    parser.add_argument('--eval', help='apply to the testdata', action='store_true')
    args = parser.parse_args()
    return args

def main():
    args = get_arguments()
    data_info = info.CleanedData()
    dataframes = info.get_train_dataframes(data_info)
    apply_preprocessing(dataframes)
    if args.eval:
        test_dataframes = info.get_test_dataframes(data_info)
        apply_preprocessing(test_dataframes)
        predict_cross_category_save_result(dataframes, [x for x in test_dataframes.values()][0])
    else:
        predict_cross_category(dataframes)

      #  predict_per_category(dataframes)

if __name__ == "__main__":
    main()
