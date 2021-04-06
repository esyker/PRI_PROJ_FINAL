# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 15:50:04 2020

@author: diogo
"""
from xml_documents_parser import DocumentParser
from html_topics_parser import TopicsParser
from Preprocess import Preprocess
import nltk
import numpy as np
from whoosh.index import create_in
from whoosh.fields import Schema,TEXT
from whoosh import scoring
from whoosh.qparser import QueryParser, OrGroup
from whoosh.index import open_dir
import re
import os
from tqdm import tqdm


class TopicIndex():
    def __init__(self,topics_filename,dir_name):
        self.parser=TopicsParser()
        self.preProcess=Preprocess()
        self.preprocessed=True
        self.topics_parsed = self.parser.get_data(topics_filename)
        self.topics=dict()
        for topic in self.topics_parsed:
            self.topics[topic['num']]=" ".join(self.preProcess.preprocess(topic['title']
                +' '+topic['narr']+' '+topic['desc']))

class WooshDocumentIndex():
   
    def __init__(self,load,dir_name,files):
        self.preProcess=Preprocess()
        self.documentParser=DocumentParser()
        self.preprocessed=True
        if not load:
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)
            schema = Schema(id = TEXT(stored=True), content=TEXT(stored=True))
            self.ix = create_in(dir_name, schema)
            self.index(files)    
        else:
            self.ix=open_dir(dir_name)
                    
    def index(self,files):
        writer = self.ix.writer()
        #Read file.
        fl = len(files)
        for fname in tqdm(range(fl)):
            parsedDoc = self.documentParser.parse(files[fname])
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
                    
            filenameOnlyExp = re.search('/[^/]*?newsML\.xml', files[fname])
            if (filenameOnlyExp):
                filenameOnlyExp = filenameOnlyExp.group()[1:]
            writer.add_document(id=filenameOnlyExp,content=words)
        writer.commit()

    def rank(self,query_str,weighting=scoring.BM25F(),k=None):
        ''' Perform a query using the weighting scoring function and obtain the corresponding textual similarity score. '''
        ix = self.ix
    
        with ix.searcher(weighting=weighting) as searcher:
            query = QueryParser("content", ix.schema, group=OrGroup).parse(query_str)
            results = searcher.search(query,limit=k)#,scored=None,sortedby=None)
            query_res = dict()
            for i,r in enumerate(results):
                id = r['id']
                #print(i,results.score(i),r['id'],r['content'],'\n')
                query_res[id] = results.score(i)
            return query_res
        
    def generate_scores(self,query,k=None):
        '''Generate scores for a given query according to BM25, TF IDF (under a cosine similarity) and Frequency rank functions'''
        bm25 = self.rank(query,weighting=scoring.BM25F(),k=k)
        cos = self.rank(query,weighting=scoring.TF_IDF(),k=k)
        freq = self.rank(query,weighting=scoring.Frequency(),k=k)
        return bm25,cos,freq 
    
    def generate_score(self,query,measure,k=None):
        '''Generate scores for a given query according to a given measure'''
        if(measure=='bm25'):
            score = self.rank(query,weighting=scoring.BM25F(),k=k)
        elif(measure=='cos'):
            score = self.rank(query,weighting=scoring.TF_IDF(),k=k)
        elif(measure=='freq'):
            score = self.rank(query,weighting=scoring.Frequency(),k=k)
        return score 
    
    def score(self,n_docs,n_rel,bm25,cos,freq,pagerank,alpha_1=3,alpha_2=3,alpha_3=1,alpha_4=1):
        scores = dict()
        #Iterate over all documents in collection.
        for k,v in bm25.items():
            #Rank combination.
            scores[k] = alpha_1*bm25[k] + alpha_2*cos[k] + alpha_3*freq[k] #+ alpha_4 * pagerank[k]
            if(pagerank.get(k) != None):
                scores[k] += alpha_4 * pagerank[k]

        return scores






'''
from neural_network import NNClassifier

start=time.time()
input_size=np.shape(trainX[search_topics[0]][0])[0]
classifier=NNClassifier(n_in=input_size,n_h=2,n_out=2,batch_size=64,learning_rate=0.01,verbose=0)
ouputNN=classifier.evaluate(search_topics,testX,RTest,DTrain=trainX,RTrain=RTrain)
end = time.time()
totalTime=round(end-start,3)
print(totalTime)
'''