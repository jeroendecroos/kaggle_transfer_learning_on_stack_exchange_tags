#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

import collections
import logging
import re
import string

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info('Logging works')

from bs4 import BeautifulSoup
import pandas

from room007.data import info


def get_dataframes():
    data_info = info.RawData()
    dataframes = {dataname: pandas.read_csv(filepath) for dataname, filepath in zip(data_info.training_sets, data_info.training_files())
                  }
    return dataframes

def strip_tags_and_uris(x):
    uri_re = r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))'

    if x:
        # BeautifulSoup on content
        soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        if soup.code:
            soup.code.decompose()
        # Get all the text out of the html
        text =  soup.get_text()
        # Returning text stripping out all uris
        return re.sub(uri_re, "", text)
    else:
        return ""

def remove_punctuation(x):
    # Lowercase all words.
    x = x.lower()
    # Remove non ASCII chars.
    # XXX There are better ways to normalize (e.g. nlu-norm's character map).
    # By doing this, we lose words like "fiancèe".
    x = re.sub(r'[^\x00-\x7f]', r' ', x)
    # Remove (replace with empty spaces actually) all punctuation.
    # XXX By doing this, we also discard apostrophes, transforming words like
    # "don't" or "we'll" into non-words. We probably don't lose much important
    # information by doing this.
    return re.sub("[" + string.punctuation + "]", " ", x)

def clean_data(dataframes):
    # This could take a while
    for df in dataframes.values():
        df["content"] = df["content"].map(strip_tags_and_uris)
        df["title"] = df["title"].map(remove_punctuation)
        df["content"] = df["content"].map(remove_punctuation)
        df["tags"] = df["tags"].map(lambda x: x.split())

def save_data(data):
    data_info = info.CleanedData()
    for data_set, data_filepath in zip(data_info.training_sets, data_info.training_files()):
        print(data_filepath)
        data[data_set].to_csv(data_filepath, index=False)

def main():
    data = get_dataframes()
    clean_data(data)
    save_data(data)

if __name__ == '__main__':
    main()
