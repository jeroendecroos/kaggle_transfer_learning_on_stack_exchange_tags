import operator
import nltk
import string


class Predictor(object):

    def __init__(self, *args):
        self._key_in_data = 'titlecontent' if 'titlecontent' in args else 'title'
        self.stopwords = set(nltk.corpus.stopwords.words())
        self._word_scores = None

    def fit(self, train_data_frame):
        sentences = train_data_frame[self._key_in_data]
        phrase_list = [x for x in self.generate_candidate_keywords(sentences) if len(x) < 3]
        word_scores = self.calculate_word_scores(phrase_list)
        self._word_scores = word_scores

    def predict(self, test_data_frame):
        sentences = test_data_frame[self._key_in_data]
        return [self.predict_one(x) for x in sentences]

    def predict_one(self, sentence):
        phrase_list = [x for x in self.generate_candidate_keywords([sentence]) if len(x) < 3]
        phrase_scores = self.calculate_phrase_scores(phrase_list, self._word_scores)
        sorted_phrase_scores = sorted(phrase_scores.items(),
                                      key=operator.itemgetter(1), reverse=True)
        if len(sorted_phrase_scores) > 0:
            sub_result = [x[0].replace(' ', '-') for x in sorted_phrase_scores[0:2]]
            return list(set(sub_result + [y for x in sub_result for y in x.split('-')]))
        else:
            return []

    def generate_candidate_keywords(self, sentences):
        phrase_list = []
        for sentence in sentences:
            words = map(lambda x: "|" if x in self.stopwords else x,
                        nltk.word_tokenize(sentence.lower()))
            phrase = []
            for word in words:
                if word == "|" or self.is_punctuation(word):
                    if len(phrase) > 0:
                        phrase_list.append(phrase)
                        phrase = []
                else:
                    phrase.append(word)
        return phrase_list

    def is_punctuation(self, word):
        return len(word) == 1 and word in string.punctuation

    def is_numeric(self, word):
        try:
            float(word) if '.' in word else int(word)
            return True
        except ValueError:
            return False

    def calculate_word_scores(self, phrase_list):
        word_freq = nltk.FreqDist()
        word_degree = nltk.FreqDist()
        for phrase in phrase_list:
            degree = len(list(filter(lambda x: not self.is_numeric(x), phrase))) - 1
            for word in phrase:
                word_freq[word] += 1
                word_degree[word] += degree
        for word in word_freq.keys():
            word_degree[word] = word_degree[word] + word_freq[word]
        word_scores = {}
        for word in word_freq.keys():
            word_scores[word] = word_degree[word] / word_freq[word]
        return word_scores

    def calculate_phrase_scores(self, phrase_list, word_scores):
        phrase_scores = {}
        for phrase in phrase_list:
            phrase_score = 0
            for word in phrase:
                phrase_score += word_scores.get(word, 0)
            phrase_scores[" ".join(phrase)] = phrase_score
        return phrase_scores









