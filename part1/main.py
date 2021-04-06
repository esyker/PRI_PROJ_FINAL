# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 16:31:58 2020

@author: diogo
"""

from DocumentInvertedIndex import DocumentInvertedIndex
from os import listdir
from os.path import join
import math
import matplotlib.pyplot as plt
import time
from tqdm import tqdm


import pickle
import re
from pprint import pprint
from statistics import stdev
import os
import psutil
process = psutil.Process(os.getpid())

def getAllFiles(path):
    subdirs = listdir(path)
    docs = []
    for subdir in subdirs:
        subdirFiles = listdir(join(path, subdir))
        docs += list(map(lambda x : path + "/" + subdir+ "/" +x, subdirFiles))
    return docs

def getEvaledDocs(filename):
    f = open(filename, 'r')
    relDocs = set()
    for line in f:
        triplet = line.split()
        relDocs.add(triplet[1]+'newsML.xml')
    return relDocs


def indexing(D,args = {'topicsFile': 'topics.txt', 'relevanceFile': 'qrels.test', 'preprocessing': False, 'dictFile': 'no-preprocessed-dictionary.comp'}):
    parsingTime = 0
    indexingTime = 0
    memoryAtStart = process.memory_info().rss
    if (args.get('dictFile') != None and args.get('dictFile') in listdir('.')):
        pickleFile = open(args.get('dictFile'), 'rb')
        iIndex = pickle.load(pickleFile)
        pickleFile.close()
        return (iIndex, (0, round(iIndex.getSize()/(1024*1024),2)))

    if(args.get('topicsFile') == None or args.get('relevanceFile') == None or args.get('preprocessing') == None):
        raise ValueError("Missing arguments!\nArguments dictionary must contain 'topicFile', 'relevanceFile' and 'tokenizationType' fields!")

    evaledDocs = getEvaledDocs(args.get('relevanceFile')) 
    filteredD = list(set(filter(lambda x: re.search('/[^/]*?newsML\.xml', x).group()[1:] in evaledDocs ,D)))

    start = time.time()
    iIndex = DocumentInvertedIndex('Train',args.get('topicsFile'), args.get('relevanceFile'), args.get('preprocessing'))
    

    sizeD = len(filteredD)
    for i in tqdm(range(0, sizeD)):
        times = iIndex.addToIndex(filteredD[i])
        parsingTime += times[0]
        indexingTime += times[1]
    
        
    iIndex.calcAllIdfs()
    end = time.time()
    iIndex.calcSize()
    size=round(iIndex.getSize()/(1024*1024),2)#size rounded to 2 decimal places
    totalTime=round(end-start,3)

    if(args.get('dictFile') != None):
        with open(args.get('dictFile'), 'wb') as dictFile:
            pickle.dump(iIndex, dictFile)
    memoryAtEnd = process.memory_info().rss
    print("Parsing Time: {0}s".format(round(parsingTime, 2)))
    print("Indexing Time: {0}s".format(round(indexingTime, 2)))

    return (iIndex, (totalTime, (memoryAtEnd - memoryAtStart) / (1024*1024) ))#returns tuple with (iIndex,(time(s),sizeOccupied(MB)))
        

def extract_topic_query(q, I, k,args = {'metric': 'TF-IDF'}):
    topic = None
    q = q.lower()

    if(args.get('metric') == None):
        raise ValueError("You must choose a metric! 'TF-IDF' or 'TF'")

    try:
        topic = I.topicIndex.dictionary[q]
    except KeyError:
        print("Query Topic Not Found")
        return None

    topic = topic[2:]
    
    topicTerms = list(map(lambda x: x[0], topic))
    dictTerms = list(filter(lambda x: x in topicTerms, I.getTerms()))

    if(args.get('metric') == 'TF'):
        # Sum of term frequencies TF 
        paramterizedTerms = [(t, I.getPosting(t)[0]) for t in dictTerms]
        paramterizedTerms = sorted(paramterizedTerms, reverse = True, key = (lambda x: x[1]))[0:k]

    elif(args.get('metric') == 'TF-IDF'):
        #TF*IFDF   TF is in I.getPosting(t)[0] and IDF is in I.getPosting(t)[1] 
        paramterizedTerms = [(t, I.getPosting(t)[0]*I.getPosting(t)[1]) for t in dictTerms]
        paramterizedTerms = sorted(paramterizedTerms, reverse = True, key = (lambda x: x[1]))[0:k]
    else:
        raise ValueError("Invalid metric '{0}' ! Valid Options: 'TF-IDF' or 'TF'".format(args.get('metric')))
    
    return list(map(lambda x: x[0], paramterizedTerms))


def boolean_query(q,I,k,args = {'metric': 'TF-IDF'}):
    query=extract_topic_query(q,I,k, args)
    documents=dict()
    for term in query:
        for doc_id,tf in I.dictionary[term][2:]:#term[0] is the term and term[1] is the tf
            if doc_id not in documents:
                documents[doc_id]=1
            else:
                documents[doc_id]+=1

    documents = [(k, v) for k,v in documents.items()]
    min_term_frequency = math.floor(0.8*len(query))
    documents = sorted(documents, key = lambda x : x[1], reverse = True)
    documents=[doc_id for doc_id,tf in documents if tf >= min_term_frequency]
    return list(filter(lambda x:x in I.topicIndex.getRelevantDocsForTopic(q.lower()) 
                       or x in I.topicIndex.getNonRelevantDocsForTopic(q.lower()),documents))
    
def ranking(q,p,I,args = {'metric': 'TF-IDF'}):
    if(args.get('metric') == 'TF-IDF'):
        similarities = rankingTFIDF(q, p, I)
    elif(args.get('metric') == 'BM25'):
        similarities = rankingBM25(q, p, I)
    elif(args.get('metric') == 'RRF'):
        similarities1 = rankingTFIDF(q, p, I)
        similarities2 = rankingBM25(q, p, I)
        similarities = RRF([similarities1, similarities2])
    else:
        raise ValueError("You must choose a ranking function: 'TF-IDF' or 'BM25' or 'RRF or 'cosSim'")
    similarities = [(k, v) for k,v in similarities.items()]
    similarities = [s for s in similarities if (s[0] in I.topicIndex.getRelevantDocsForTopic(q) or s[0] in I.topicIndex.getNonRelevantDocsForTopic(q))]   
    similarities = sorted(similarities, key= lambda x: x[1], reverse=True)
    similarities=list(filter(lambda x:x[0] in I.topicIndex.getRelevantDocsForTopic(q.lower()) 
                       or x[0] in I.topicIndex.getNonRelevantDocsForTopic(q.lower()),similarities))
    return similarities[0:p]
    
def rankingTFIDF(q,p,I):
    similarities = {}
    for term in I.topicIndex.dictionary[q.lower()]:#iterate over all the terms in the doc query
        tfidf_topic = term[1] #get the topic tf-idf from the dictionary of the topic
        try:
            iIndex=I.dictionary[term[0]] #get the postings from the term in the document Inverted Index
        except KeyError:
            continue
        idf=iIndex[1] #get the idf from the term posting (in the docment Inverted Index)
        for pair in iIndex[2:]:
            docId = pair[0]
            TF_df = pair[1]
            if docId not in similarities.keys():
                similarities[docId] = 0
            similarities[docId] += math.log(1+TF_df,10)* idf * math.log(1+tfidf_topic,10) * idf
    return similarities

def rankingBM25(q, p, I):
    k1 = 1.2
    b = 0.75
    avgDocSum = 0
    for i in I.document_lengths.keys():
        avgDocSum += I.document_lengths[i]
    avgDocLen =  avgDocSum / len(I.document_lengths.keys())
    similarities = { } #idf, #doc_tf
    for term in I.topicIndex.dictionary[q.lower()]:
        postings = I.getPosting(term[0])
        if (postings == None):
            continue
        idfTerm = postings[1]
        for posting in postings[2:]:
            docId = posting[0]
            tf = posting[1]

            if (similarities.get(docId) == None):
                similarities[docId] = 0
            similarities[docId] += idfTerm * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (I.document_lengths[docId]/ avgDocLen))))
    return similarities

def RRF(rankingsList):
    rrf = { }
    for doc in rankingsList[0].keys():
        if (rrf.get(doc) == None):
            rrf[doc] = 0
        for ranking in rankingsList:
            rrf[doc] += (1 / 50 + ranking[doc])
    return rrf


def calcPrecision(retrieved, relevant, nonRelevant):
    truePositives = len(set(retrieved).intersection(relevant))
    falsePositives = len(set(retrieved).intersection(nonRelevant)) 
    if( truePositives + falsePositives == 0):
        return 0
    return truePositives / (truePositives + falsePositives)

def calcRecall(retrieved,relevant,nonRelevant):
    truePositives = len(set(retrieved).intersection(relevant))
    falseNegatives=len((set(relevant)|set(nonRelevant)-set(retrieved)).intersection(relevant))
    if( truePositives + falseNegatives == 0):
        return 0
    return truePositives / (truePositives + falseNegatives)

def calcPrecisionAtRecall(recall,retrieved,relevant,nonRelevant):
    if (recall==0):
        return 1;
    return calcPrecision(retrieved[0:recall],relevant,nonRelevant)

def plotPrecisionRecallCurve(recall,precision):
    if not precision:
        return None
    fig, ax = plt.subplots()
    for i in range(len(recall)-1):
        ax.plot((recall[i],recall[i]),(precision[i],precision[i+1]),'k-',color='red') #vertical\n",
        ax.plot((recall[i],recall[i+1]),(precision[i+1],precision[i+1]),'k-',color='red') #horizontal\n"
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    plt.axis([0.0,1.1,0.0,1.1])
    plt.show()


def BPREF(queryResult, relevant, nonRelevant):
    _sum = 0
    relevantSize = len(relevant)
    nonRelevantSize = len(nonRelevant.intersection(set(queryResult)))
    dividingFactor = min(relevantSize,nonRelevantSize)

    judgedAndRetreieved = set(queryResult).intersection(relevant)
    judgedAndRetrievedSize = len(judgedAndRetreieved)
    if(nonRelevantSize == 0):
        return 1
    for i in range(1,judgedAndRetrievedSize+1):
        judgedAndRelevantAndAk = (nonRelevant).intersection(set(queryResult[:i]))
        _sum += float(1 - (len(judgedAndRelevantAndAk)) / dividingFactor)
    return float(float(_sum) / relevantSize)

def MAP(retrieved, relevantFiles, nonRelevantFiles):
    precisionSum = 0
    k = len(retrieved)
    if(k == 0):
        return 0
    for i in range(1, k+1):
        if (retrieved[i-1] in relevantFiles):
            precisionSum += float(calcPrecision(retrieved[:i], relevantFiles, nonRelevantFiles))
    return float(precisionSum) / float(k)

    
def precisionRecallCurve(query_result,relevantFiles,nonRelevantFiles):
    if not query_result:
        return [],[]
    recallValues=[]
    precisionAtRecall = []

    for recall_value in range(len(query_result)+1):#0.1 intervals
        result=calcPrecisionAtRecall(recall_value,query_result,relevantFiles,nonRelevantFiles)
        precisionAtRecall.append(result)
        if(len(query_result) == 0):
            recallValues.append(0)
            continue
        recallValues.append(recall_value/len(query_result))
    return recallValues,precisionAtRecall

def evaluation(Q_test,R_test,D_test,args = {'k':10, 'rankingMeasure': 'TF-IDF','booleanMetric': 'TF-IDF', 'queryType': 'ranked'}):
    iIndex=D_test
    queryStartTime = 0
    queryEndTime = 0
    timeSum = 0
    queriesDone = 0
    statsObject = {}
    k=args.get('k')
    if k == None or (not isinstance(k,int)):
        raise ValueError("Invalid k '{0}' !".format(args.get('k')))
    for topic in Q_test:
        statsObject[topic] = {}
        query_result=None
        if (args.get('queryType') != None and args.get('queryType') == 'boolean'):
                queryStartTime = time.time()
                query_result=boolean_query(topic,iIndex,k, {'metric': args.get('booleanMetric')})
                queryEndTime = time.time()
                queriesDone += 1

        elif (args.get('queryType') != None and args.get('queryType') == 'ranked'):
                queryStartTime = time.time()
                query_result = list(map(lambda x: x[0], ranking(topic,k,iIndex, {'metric': args.get('rankingMeasure')})))
                queryEndTime = time.time()
                queriesDone += 1
        else:
            raise ValueError("Invalid query type '{0}' !".format(args.get('queryType')))

        timeSum += queryEndTime - queryStartTime

        if (query_result != None):
            relevantFiles=R_test.getRelevantDocsForTopic(topic)
            nonRelevantFiles = R_test.getNonRelevantDocsForTopic(topic)
            
            recallValues,precisionAtRecall=precisionRecallCurve(query_result, relevantFiles,nonRelevantFiles)
            plotPrecisionRecallCurve(recallValues,precisionAtRecall)
            statsObject[topic]['precision-at-recall'] = precisionAtRecall
            statsObject[topic]['MAP'] = MAP(query_result, relevantFiles, nonRelevantFiles)
            statsObject[topic]['BPREF'] = BPREF(query_result, relevantFiles, nonRelevantFiles)
            statsObject[topic]['Precision'] = calcPrecision(query_result, relevantFiles, nonRelevantFiles)
            statsObject[topic]['Recall'] = calcRecall(query_result, relevantFiles, nonRelevantFiles)
            statsObject[topic]['efficiency'] = queryEndTime - queryStartTime
    return statsObject
