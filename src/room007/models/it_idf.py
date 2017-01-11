#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

from collections import Counter
from functools import partial
from itertools import chain
from os.path import basename, join, splitext
import re

import numpy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score


from room007.data import info


TAG_STATS_IDX = pd.Index(('present', 'total'))




def norm_tag(tag):
    return re.sub('-', ' ', tag)

def get_vectorizer(train):
    vectorizer = TfidfVectorizer()
    train_features = vectorizer.fit_transform(train['titlecontent'])
    return vectorizer

def predict_per_category(data):
    # split data, 0.2 arbitrary chosen
    train, test = train_test_split(data, test_size=0.2)

    # get features 
    vectorizer = get_vectorizer(train)

    # lets just see if the top entries can predict
    feature_names = vectorizer.get_feature_names()
    all_content = [x for x in test['titlecontent']]
    all_tags  = [x for x in test['tags']]

    score  = 0
    for i, entry in enumerate(all_content):
        test_features = vectorizer.transform([entry])
        entry = test_features.toarray()[0]
        content = all_content[i]
        #print(content)
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
        #print(fscore)
    return(score/len(all_tags)*100)


def main():
    data_info = info.CleanedData()
    dataframes = info.get_train_dataframes(data_info)
    total_score = 0
    for fname, data in dataframes.items():
        data['tags'] = data['tags'].str.split()
        data['titlecontent'] = data['title'] + data['content']
        score = predict_per_category(data)
        print(fname)
        print(score)
        total_score += score
    avg_score = total_score/len(dataframes)
    print('total: {}'.format(avg_score))

if __name__ == "__main__":
    main()
