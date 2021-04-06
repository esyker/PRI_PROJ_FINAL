import nltk
from stop_words import get_stop_words
import string


class Index:

    def __init__(self, name):
        self.name = name 
        self.document_count = 0
        self.dictionary = {}
        
    def tokenizeRemoveStopWords(self,_list):
            stopWords = nltk.corpus.stopwords.words('english') + get_stop_words('en')
            punctuation = list(string.punctuation)
            _list = list(filter(lambda word: word not in set(stopWords+punctuation) and word != "''", nltk.word_tokenize(_list)))
            return _list

    def addToIndex(self):
        pass