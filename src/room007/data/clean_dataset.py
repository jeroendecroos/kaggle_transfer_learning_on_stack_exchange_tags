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

from room007.data import info


def strip_tags_and_uris(x):
    if x:
        cleantext = BeautifulSoup(x, "html.parser").text
        # BeautifulSoup on content
        #soup = BeautifulSoup(x, "html.parser")
        # Stripping all <code> tags with their content if any
        #if soup.code:
        #    soup.code.decompose()
        # Get all the text out of the html
        #text =  soup.get_text()
        # Returning text stripping out all uris
        return cleantext #re.sub(uri_re, "", text)
    else:
        return ""

def strip_latex_code(text):
    #if 'lecturer poses the question' in text:
    #    import pdb; pdb.set_trace()
    return re.sub(r'\\[a-zA-Z/\-_0-9{}]{2,}', '', 
                re.sub(r'(\$)?\$.*?\$(\$)?','', text))

def remove_punctuation(x):
    # Lowercase all words.
    x = x.lower()
    x = ' '.join(re.findall(r"\w+|[^\w\s]", x, re.UNICODE))
    # Remove non ASCII chars.
    # XXX There are better ways to normalize (e.g. nlu-norm's character map).
    # By doing this, we lose words like "fiancèe".
    #x = re.sub(r'[^\x00-\x7f]', r' ', x)
    # Remove (replace with empty spaces actually) all punctuation.
    # XXX By doing this, we also discard apostrophes, transforming words like
    # "don't" or "we'll" into non-words. We probably don't lose much important
    # information by doing this.
    return x 
    # TODO Normalize whitespace, e.g. newlines should be replaced with spaces
    # and whitespace then squeezed.


def clean_data(dataframes):
    # This could take a while
    for df in dataframes.values():
        df["content"] = df["content"].map(strip_tags_and_uris)
        df["content"] = df["content"].map(strip_latex_code)
# We can also keep punctuation because it can give NLP information
#        df["title"] = df["title"].map(remove_punctuation)
#        df["content"] = df["content"].map(remove_punctuation)



def save_data(data):
    data_info = info.CleanedData()
    for data_set, data_filepath in zip(data_info.training_sets, data_info.training_files):
        print(data_filepath)
        data[data_set].to_csv(data_filepath, index=False)


def main():
    data_info = info.RawData()
    # XXX Do not split tags because this would only result in storing what was
    # a space-separated list using a Python list syntax, thereby requiring
    # re-parsing it as Python on next load.
    data = info.get_train_dataframes(data_info, split_tags=False)
    clean_data(data)
    data_info = info.CleanedData()
    info.save_training_data(data_info, data)
    data = info.get_test_dataframes(data_info)
    clean_data(data)
    data_info = info.CleanedData()
    info.save_test_data(data_info, data)



if __name__ == '__main__':
    main()
