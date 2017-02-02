import pandas
import logging
import operator
import nltk
import string

from room007.data import info
from room007.models.word_tag_predictor import apply_preprocessing

logger = logging.getLogger()


def log(message):
    logger.info(message)


class RakePredictor(object):

    def __init__(self, rake, word_scores):
        self.word_scores = word_scores
        self.rake = rake

    @staticmethod
    def fit(sentences):
        rake = RakeKeywordExtractor()
        phrase_list = [x for x in rake._generate_candidate_keywords(sentences) if len(x) < 3]
        word_scores = rake._calculate_word_scores(phrase_list)
        return RakePredictor(rake, word_scores)

    def predict(self, sentences):
        return [self.predict_one(x) for x in sentences]

    def predict_one(self, sentence):
        phrase_list = [x for x in self.rake._generate_candidate_keywords([sentence]) if len(x) < 3]
        phrase_scores = self.rake._calculate_phrase_scores(phrase_list, self.word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(),
                                      key=operator.itemgetter(1), reverse=True)
        if len(sorted_phrase_scores) > 0:
            subresult = [x[0].replace(' ', '-') for x in sorted_phrase_scores[0:2]]
            return list(set(subresult + [y for x in subresult for y in x.split('-')]))
        else:
            return []


def train(train_data):
    log("Starting training")
    model = RakePredictor.fit(train_data)
    log("Training complete")
    return model


def predict(model, test_data):
    log("Starting to predict")
    predictions = model.predict(test_data['title'])
    test_data['tags'] = predictions
    test_data['tags'] = test_data['tags'].apply(' '.join)
    log("Prediction complete")
    return predictions


def save(test_data):
    log("Saving the results")
    filename = 'rake.out.csv'
    test_data.to_csv(filename, columns=['id','tags'], index=False)
    log("Results saved")


def run_rake_on_test(train_data, test_data):
    sentences = train_data['title']
    model = train(sentences)
    predict(model, test_data)
    save(test_data)


def rake_main(train_data, test_data):
    run_rake_on_test(train_data, test_data)


# most of the code below was borrowed from
# http://sujitpal.blogspot.de/2013/03/implementing-rake-algorithm-with-nltk.html


def isPunct(word):
    return len(word) == 1 and word in string.punctuation


def isNumeric(word):
    try:
        float(word) if '.' in word else int(word)
        return True
    except ValueError:
        return False


class RakeKeywordExtractor:
    def __init__(self):
        self.stopwords = set(nltk.corpus.stopwords.words())
        self.top_fraction = 3 # consider top third candidate keywords by score

    def _generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            words = map(lambda x: "|" if x in self.stopwords else x,
                        nltk.word_tokenize(sentence.lower()))
            phrase = []
            for word in words:
                if word == "|" or isPunct(word):
                    if len(phrase) > 0:
                        phrase_list.append(phrase)
                        phrase = []
                else:
                    phrase.append(word)
        return phrase_list


    def _calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(list(filter(lambda x: not isNumeric(x), phrase))) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree # other words
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word] # itself
            #  word score = deg(w) / freq(w)
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores

    def _calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores.get(word, 0)
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores

    def extract(self, text, incl_scores=False):
        sentences = nltk.sent_tokenize(text)
        phrase_list = self._generate_candidate_keywords(sentences)
        word_scores = self._calculate_word_scores(phrase_list)
        phrase_scores = self._calculate_phrase_scores(phrase_list, word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(),
                                      key=operator.itemgetter(1), reverse=True)
        n_phrases = len(sorted_phrase_scores)
        if incl_scores:
            return sorted_phrase_scores[0:int(n_phrases/self.top_fraction)]
        else:
            return map(lambda x: x[0],
                       sorted_phrase_scores[0:int(n_phrases/self.top_fraction)])


def test():
    rake = RakeKeywordExtractor()
    keywords = rake.extract(
        """Compatibility of systems of linear constraints over the set of natural
        numbers. Criteria of compatibility of a system of linear Diophantine
        equations, strict inequations, and nonstrict inequations are considered.
        Upper bounds for components of a minimal set of solutions and algorithms
        of construction of minimal generating sets of solutions for all types of
        systems are given. These criteria and the corresponding algorithms for
        constructing a minimal supporting set of solutions can be used in solving
        all the considered types of systems and systems of mixed types.""",
        incl_scores=True)
    print(keywords)


def test2():
    text = """Compatibility of systems of linear constraints over the set of natural
        numbers. Criteria of compatibility of a system of linear Diophantine
        equations, strict inequations, and nonstrict inequations are considered.
        Upper bounds for components of a minimal set of solutions and algorithms
        of construction of minimal generating sets of solutions for all types of
        systems are given. These criteria and the corresponding algorithms for
        constructing a minimal supporting set of solutions can be used in solving
        all the considered types of systems and systems of mixed types."""
    rake = RakePredictor.fit(text)
    print(rake.predict(text))




def main():
    log("Started preparing the data")
    data_info = info.CleanedData()
    train_data_frames = info.get_train_dataframes(data_info)
    test_data_frames = info.get_test_dataframes(data_info)

    train_data = pandas.concat([data for name, data in train_data_frames.items()],
                               ignore_index=True)
    test_data = [x for x in test_data_frames.values()][0]
    log("Prepared the data")

    apply_preprocessing(train_data)
    apply_preprocessing(test_data)
    rake_main(train_data, test_data)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
    #test2()
