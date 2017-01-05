import re
import pandas
from bs4 import BeautifulSoup

import info

def get_dataframes():
    data_info = info.RawData()
    dataframes = {dataname, panda.read_csv(filepath)
                  for zip(info.training_sets, info.training_files)
                  }
    return dataframes

def stripTagsAndUris(x):
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

def removePunctuation(x):
    # Lowercasing all words
    x = x.lower()
    # Removing non ASCII chars
    x = re.sub(r'[^\x00-\x7f]',r' ',x)
    # Removing (replacing with empty spaces actually) all the punctuations
    return re.sub("["+string.punctuation+"]", " ", x)

def clean_data(dataframes):
    # This could take a while
    for df in dataframes.values():
        df["content"] = df["content"].map(stripTagsAndUris)
        df["title"] = df["title"].map(removePunctuation)
        df["content"] = df["content"].map(removePunctuation)
        df["tags"] = df["tags"].map(lambda x: x.split())

def set_tag_in_data(data):
    pass

def main():
    data = get_dataframes()
    clean_data(data)
		set_tag_in_data(data)
		


if __name__ == '__main__':
    main()
