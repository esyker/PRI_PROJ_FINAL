from Index import Index
import nltk
import sys
import math
from html_topics_parser import TopicsParser
from Preprocess import Preprocess



class TopicsIndex(Index):
    def __init__(self,name,documentIndex,topics_filename,relevance_filename, preprocessed):
        super().__init__(name)
        self.parser=TopicsParser()
        self.documentIndex = documentIndex
        self.dictionary=dict()
        self.relevance_dictionary=dict()
        self.topics = self.parser.get_data(topics_filename)
        self.preProcess=Preprocess()
        self.preprocessed = preprocessed
        self.addDocsToIndex()
        self.caclRelevantDocsForTopic(relevance_filename)

    def addToIndex(self,topic):
            self.dictionary[topic['num']]=list()
            if(self.preprocessed):
                title_words = self.preProcess.preprocess(topic['title'])
                desc_words = self.preProcess.preprocess(topic['desc'])
                narr_words = self.preProcess.preprocess(topic['narr'])
            else:
                title_words = nltk.word_tokenize(topic['title'].lower())
                desc_words = nltk.word_tokenize(topic['desc'].lower())
                narr_words = nltk.word_tokenize(topic['narr'].lower())

            title_data_analysis = nltk.FreqDist(title_words)
            desc_data_analysis = nltk.FreqDist(desc_words)
            narr_data_analysis = nltk.FreqDist(narr_words)
            
            topicTerms = list(title_data_analysis.keys()) + list(desc_data_analysis.keys()) + list(narr_data_analysis.keys())
            
            for term in topicTerms:
                total_count=int(title_data_analysis[term])+int(desc_data_analysis[term])+int(narr_data_analysis[term])
                self.dictionary[topic['num']].append((term,total_count))#pair saved is ('term', tf)  
                
                
    def addSentenceToIndex(self, topic):
        self.dictionary[topic['num']]=list()
        title_words = nltk.sent_tokenize(topic['title'].lower())
        desc_words = nltk.sent_tokenize(topic['desc'].lower())
        narr_words = nltk.sent_tokenize(topic['narr'].lower())
        title_data_analysis = nltk.FreqDist(title_words)
        desc_data_analysis = nltk.FreqDist(desc_words)
        narr_data_analysis = nltk.FreqDist(narr_words)
        
        topicTerms = list(title_data_analysis.keys()) + list(desc_data_analysis.keys()) + list(narr_data_analysis.keys())

        for term in topicTerms:
            total_count=int(title_data_analysis[term])+int(desc_data_analysis[term])+int(narr_data_analysis[term])
            self.dictionary[topic['num']].append((term,total_count))#pair saved is ('term', tf)  

    def caclRelevantDocsForTopic(self, relevanceDoc):
        qrels = open(relevanceDoc, 'r')
        for topic in self.topics:
            self.relevance_dictionary[topic['num'].lower()] =  { }
            self.relevance_dictionary[topic['num'].lower()]['relevant'] = set()
            self.relevance_dictionary[topic['num'].lower()]['non-relevant'] = set()

        for line in qrels:
            relevanceTriplet = line.split()
            file_name=relevanceTriplet[1]+"newsML.xml"
            if(relevanceTriplet[2] == '1'):
                self.relevance_dictionary[relevanceTriplet[0].lower()]['relevant'].add(file_name)
            else:
                self.relevance_dictionary[relevanceTriplet[0].lower()]['non-relevant'].add(file_name)
        qrels.close()

    def getRelevantDocsForTopic(self, topic):
        return self.relevance_dictionary[topic.lower()]['relevant']

    def getNonRelevantDocsForTopic(self, topic):
        return self.relevance_dictionary[topic.lower()]['non-relevant']

    def addDocsToIndex(self):
        for topic in self.topics:
            self.addToIndex(topic)

    def addSentences(self):
        for topic in self.topics:
            self.addSentenceToIndex(topic)

    def getSize(self):
        return sys.getsizeof(self.dictionary) + sys.getsizeof(self.relevance_dictionary)