from Index import Index
from TopicsIndex import TopicsIndex
import nltk
import sys
from xml_documents_parser import DocumentParser
import math
import re
import time
from Preprocess import Preprocess
import os
import psutil
process = psutil.Process(os.getpid())

class DocumentInvertedIndex(Index):

    def __init__(self, name,topics_filename,relevance_filename, preprocessed):
        super().__init__(name)
        self.topicIndex = TopicsIndex('Topic Index', self,topics_filename,relevance_filename, preprocessed)
        self.documentParser = DocumentParser()
        self.docLength = 0
        self.dictionary={}
        self.size = 0
        self.preProcess=Preprocess()
        self.evalDocs = self.calcEvaluatedDocs()
        self.preprocessed = preprocessed
        self.processingTime = 0 
        self.indxingTime = 0
        self.processingMemory = 0
        self.indexingMemory = 0
        self.document_lengths=dict()
    
    def addToIndex(self, filename):
        processingStartmemory = process.memory_info().rss
        parseStartTime = time.time()         
        parsedDoc = self.documentParser.parse(filename)
        words = []

        
        if(self.preprocessed):
            if(parsedDoc.get('title') != None):
                words += self.preProcess.preprocess(parsedDoc['title'])
            if(parsedDoc.get('text') != None):
                words +=  self.preProcess.preprocess(parsedDoc['text'])
            if(parsedDoc.get('byline') != None):
                words +=  self.preProcess.preprocess(parsedDoc['byline'])
            if(parsedDoc.get('dateline') != None):
                words +=  self.preProcess.preprocess(parsedDoc['dateline'])
        else:
            if(parsedDoc.get('title') != None):
                words += nltk.word_tokenize(parsedDoc['title'].lower())
            if(parsedDoc.get('text') != None):
                words +=  nltk.word_tokenize(parsedDoc['text'].lower())
            if(parsedDoc.get('byline') != None):
                words +=  nltk.word_tokenize(parsedDoc['byline'].lower())
            if(parsedDoc.get('dateline') != None):
                words +=  nltk.word_tokenize(parsedDoc['dateline'].lower())

        filenameOnlyExp = re.search('/[^/]*?newsML\.xml', filename)
        if (filenameOnlyExp):
            filenameOnly = filenameOnlyExp.group()[1:]
        
        parseEndTime = time.time()
        processingEndMemory = process.memory_info().rss

        self.processingMemory = (processingEndMemory - processingStartmemory) / (1024*1024)
        self.processingTime = parseEndTime - parseStartTime
        
        indexingStartMemory = process.memory_info().rss
        indexStartTime = time.time()
        data_analysis = nltk.FreqDist(words)
        _document_words=0
        for k, v in data_analysis.items():
            _document_words+=v
            if k in self.dictionary:
                self.dictionary[k].append((filenameOnly,v))
                self.dictionary[k][0] += v
            else:
                #[tf,idf,(document1,tf-df),(document2,tf-df)]
                self.dictionary[k] = [v,0,(filenameOnly,v)]
        indexEndTime = time.time()
        indexingEndMemory = process.memory_info().rss
        self.indexingTime = indexEndTime - indexStartTime
        self.indexingMemory = (indexingEndMemory - indexingStartMemory) / (1024*1024)
        self.docLength += len(words)
        self.document_count += 1
        self.document_lengths[filenameOnly]=_document_words
        return (self.processingTime, self.indexingTime)


    def addSententencesToIndex(self, filename):
        parsedDoc = self.documentParser.parse(filename)
        sentences = []
        if(parsedDoc.get('title') != None):
            sentences += nltk.sent_tokenize(parsedDoc['title'].lower())
        if(parsedDoc.get('text') != None):
            sentences += nltk.sent_tokenize(parsedDoc['text'].lower())
        if(parsedDoc.get('byline') != None):
            sentences += nltk.sent_tokenize(parsedDoc['byline'].lower())
        if(parsedDoc.get('dateline') != None):
            sentences += nltk.sent_tokenize(parsedDoc['dateline'].lower())
        
        data_analysis = nltk.FreqDist(sentences)
        for k, v in data_analysis.items(): 
            if k in self.dictionary:
                self.dictionary[k].append((filename,v))
                self.dictionary[k][0] += v
            else:
                #[tf,idf,(document1,tf-df),(document2,tf-df)]
                self.dictionary[k] = [v,0,(filename,v)]
        #self.docIdToFilename[self.document_count] = filename
        self.docLength += len(sentences)
        self.document_count += 1

    def print_statistics(self):
        #print(str(index["document_count"]))
        print('Number of documents indexed: '+ str(self.document_count))
        print('Number of terms indexed: '+ str(len(self.dictionary)))
        print('Total number of individual ocurrencies:')
        # podiam fazer isto com prints mais bonitos :wink:
        for k in self.dictionary.keys():
            print(str(k)+" : "+ str(self.dictionary[k][0]))

    def inverseDocFreq(self,term):
        if term not in self.dictionary:
            return 0
        idf = math.log(self.document_count/(len(self.dictionary[term])-2),10)
        return idf

    
    def calcAllIdfs(self):
        for term in self.dictionary:
            self.dictionary[term][1] = self.inverseDocFreq(term)


    def calcAllTfidfs(self):
        for term in self.dictionary:
            if (self.tf_idfs.get(term) == None):
                self.tf_idfs[term] = {}
            for postingPair in self.dictionary[term][2:]:
                self.tf_idfs[term][postingPair[0]] = self.getIdf(term) * (1 + math.log(postingPair[1], 10))


    def getTerms(self):
        return self.dictionary.keys()

    def getPosting(self, term):
        return self.dictionary.get(term)
    
    def getIdf(self, term):
        if(self.dictionary.get(term) == None):
            return 0
        else:
            return self.dictionary.get(term)[1]
        
    def clear(self):
        self.dictionary = {"document_count": 0,}

    def calcSize(self):
        self.size = 0
        for key in self.dictionary.keys():
            self.size += sys.getsizeof(self.dictionary[key][0]) + sys.getsizeof(self.dictionary[key][1])
            for posting in self.dictionary[key][2:]:
                self.size += sys.getsizeof(posting[0]) + sys.getsizeof(posting[1])

    def getSize(self):
        return self.size

    def calcEvaluatedDocs(self):
        #print(list(self.topicIndex.relevance_dictionary.values()))
        evalDocs = set()
        for i in list(self.topicIndex.relevance_dictionary.values()):
            for j in i['relevant']:
                evalDocs.add(j)

            for j in i['non-relevant']:
                evalDocs.add(j)

        return evalDocs