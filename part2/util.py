from os import listdir
from os.path import join
import os
import re

import matplotlib.pyplot as plt
import math

from xml_documents_parser import DocumentParser
from html_topics_parser import TopicsParser
from Preprocess import Preprocess
from sklearn.feature_extraction.text import TfidfVectorizer

def parseDoc(doc):
        tokenizer=Preprocess()
        parser=DocumentParser()
        doc = doc.replace('d_', 'D_')
        doc = doc.replace('newsml', 'newsML')
        parsedDoc = parser.parse(doc)
        words = []
        if(parsedDoc.get('title') != None):
            words += tokenizer.preprocess(parsedDoc['title'])
        if(parsedDoc.get('text') != None):
            words +=  tokenizer.preprocess(parsedDoc['text'])
        if(parsedDoc.get('byline') != None):
            words +=  tokenizer.preprocess(parsedDoc['byline'])
        if(parsedDoc.get('dateline') != None):
            words +=  tokenizer.preprocess(parsedDoc['dateline'])
        return words

def noPreprocessDocs(docs):
    parser=DocumentParser()
    docList = []
    for doc in docs:
        doc = doc.replace('d_', 'D_')
        doc = doc.replace('newsml', 'newsML')
        parsedDoc = parser.parse(doc)
        words = ""
        if(parsedDoc.get('title') != None):
            words += parsedDoc['title']
        if(parsedDoc.get('text') != None):
            words +=  parsedDoc['text']
        if(parsedDoc.get('byline') != None):
            words +=  parsedDoc['byline']
        if(parsedDoc.get('dateline') != None):
            words +=  parsedDoc['dateline']
        docList.append(words)
    return docList

def noPreprocessTopics(path):
    parser = TopicsParser()
    splittedTopics = parser.get_data(path)
    topics = []
    for i in splittedTopics:
        topicText = ''
        topicText += i['title'] + ' '
        topicText += i['desc'] + ' '
        topicText += i['narr']
        topics.append(topicText)
    return topics

def pointDistance(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[0]-p2[0])**2)

def getAllFiles(path):
    subdirs = listdir(path)
    docs = []
    for subdir in subdirs:
        subdirFiles = listdir(join(path, subdir))
        docs += list(map(lambda x : path + "/" + subdir+ "/" +x, subdirFiles))
    return docs

def getEvaledDocs(filename):
    if (filename == 'qrels.train'):
        allFiles = getAllFiles('./rcv1/D_train')
    else:
        allFiles = getAllFiles('./rcv1/D_test')
    f = open(filename, 'r')
    relDocsIds = set()
    for line in f:
        triplet = line.split()
        relDocsIds.add(triplet[1])
    f.close()
    #relDocs = set(filter(lambda x: extractFileId(x) in relDocsIds , allFiles))
    relDocs = list(filter(lambda x: extractFileId(x) in relDocsIds , allFiles))

    return relDocs

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def extractFileId(path):
    filenameOnlyExp = re.search('/[^/]*?newsML\.xml', path)
    filenameOnly = ''
    if (filenameOnlyExp):
        filenameOnly = filenameOnlyExp.group()[1:-10]
        return filenameOnly

def getEvaledDocsForTopic(filename, topic, col): #col 'test' or 'train'
    if(col == 'test'):
        allFiles = getAllFiles('./rcv1/D_test')
    if(col == 'train'):
        allFiles = getAllFiles('./rcv1/D_train')
    
    f = open(filename, 'r')
    relDocsIds = set()
    for line in f:
        triplet = line.split()
        if(triplet[0] == topic):
            relDocsIds.add(triplet[1])
    f.close()

    relDocs = list(filter(lambda x: extractFileId(x) in relDocsIds , allFiles))
    return relDocs



def getRelevanceForTopic(filename, topic):
    f = open(filename, 'r')
    relevance = []
    for line in f:
        triplet = line.split()
        if(triplet[0] == topic):
            relevance.append(int(triplet[2]))
    return relevance

def VectorizerParseDoc(doc):
        tokenizer=Preprocess()
        parser=DocumentParser()
        parsedDoc = parser.parse(doc.replace('d_train', 'D_train').replace('d_test', 'D_test').replace('newsml.xml', 'newsML.xml'))
        words = []
        if(parsedDoc.get('title') != None):
            words += tokenizer.preprocess(parsedDoc['title'])
        if(parsedDoc.get('text') != None):
            words +=  tokenizer.preprocess(parsedDoc['text'])
        if(parsedDoc.get('byline') != None):
            words +=  tokenizer.preprocess(parsedDoc['byline'])
        if(parsedDoc.get('dateline') != None):
            words +=  tokenizer.preprocess(parsedDoc['dateline'])
        return words
    
def getDocsForTopics(filename, topics, allFiles): #col 'test' or 'train'    
    f = open(filename, 'r')
    DTrain=dict()
    RTrain=dict()
    docs_ids=set()
    for line in f:
        triplet = line.split()
        if(triplet[0] in topics):
            docs_ids.add(triplet[1])
            try:
                DTrain[triplet[0]].append(triplet[1])
                RTrain[triplet[0]].append(int(triplet[2]))
            except KeyError:
                DTrain[triplet[0]]=[triplet[1]]
                RTrain[triplet[0]]=[int(triplet[2])]        
    f.close()
    return DTrain,RTrain,docs_ids

def getCollectionVector(topics):
    #'Train'
    TrainFiles = getAllFiles('./rcv1/D_train')
    DTrain,RTrain,train_docs_ids=getDocsForTopics('qrels.train',topics,TrainFiles)
    #'Test'
    TestFiles = getAllFiles('./rcv1/D_test')
    DTest,RTest,test_docs_ids=getDocsForTopics('qrels.test',topics,TestFiles)
    fileRepresentation=dict()
    
    allFiles=TrainFiles+TestFiles
    docs_ids=train_docs_ids.union(test_docs_ids)
    allFiles=[file for file in allFiles if extractFileId(file) in docs_ids]
    
    vectorizer = TfidfVectorizer(tokenizer=VectorizerParseDoc,use_idf = False)
    vectorspace = vectorizer.fit_transform(allFiles)
    for i in range(len(allFiles)):
        fileRepresentation[extractFileId(allFiles[i])]=vectorspace[i].toarray()[0]
        #print(type(vectorspace[i]),np.shape(vectorspace[i]),' ',np.shape(vectorspace[i].toarray()[0]))     
        #fileRepresentation[file_id]=vectorspace.toarray()[0]
    for topic in topics:
        for i in range(len(DTrain[topic])):
            DTrain[topic][i]=fileRepresentation[DTrain[topic][i]]
        for i in range(len(DTest[topic])):
            DTest[topic][i]=fileRepresentation[DTest[topic][i]]
    return DTrain,RTrain,DTest,RTest

def getCollection(topics):
    #'Train'
    TrainFiles = getAllFiles('./rcv1/D_train')
    DTrain,RTrain,train_docs_ids=getDocsForTopics('qrels.train',topics,TrainFiles)
    #'Test'
    TestFiles = getAllFiles('./rcv1/D_test')
    DTest,RTest,test_docs_ids=getDocsForTopics('qrels.test',topics,TestFiles)    
    
    #files=TrainFiles+TestFiles
    #docs_ids=train_docs_ids.union(test_docs_ids)
    TrainFiles=[file for file in TrainFiles if extractFileId(file) in train_docs_ids]
    TestFiles=[file for file in TestFiles if extractFileId(file) in test_docs_ids]
    return TrainFiles,TestFiles,DTrain,RTrain,DTest,RTest


def computeSum(vector):
    sum1 = 0
    for k,v in vector.items():
        sum1 += v
    return sum1

def fillVector(allEntries, vector):
    for entry in allEntries:
        if (vector.get(entry) == None):
            vector[entry] = 0

def normalizeDict(d):
    vecSum = computeSum(d)
    for k,v in d.items():
        d[k] = v/vecSum

        
def getRelevantNonRelevant(topic):
    evaledDocs = getEvaledDocsForTopic('qrels.test', topic, 'test')
    relevance = getRelevanceForTopic('qrels.test', topic)
    relevant = [extractFileId(evaledDocs[i]) for i in range(len(evaledDocs)) if relevance[i] == 1]
    nonRelevant = [extractFileId(evaledDocs[i]) for i in range(len(evaledDocs)) if relevance[i] == 0]
    return relevant,nonRelevant

def setupEvalManyFeatures(search_topics, DTrain, DTest,trainDocsIndex,testDocsIndex,topicsIndex):
    topics=topicsIndex.topics

    train_bm25=dict()
    train_cos=dict()
    train_freq=dict()
    test_bm25=dict()
    test_cos=dict()
    test_freq=dict()

    for topic in search_topics:
        train_bm25[topic],train_cos[topic],train_freq[topic] =trainDocsIndex.generate_scores(topics[topic.lower()], k=None)
        test_bm25[topic],test_cos[topic],test_freq[topic]=testDocsIndex.generate_scores(topics[topic.lower()], k=None)

    trainX=dict()
    testX=dict()
    
    for topic in search_topics:
        trainX[topic]=list()
        for fileID in DTrain[topic.upper()]:
            fileName=fileID+"newsML.xml"
            try :
                value=[train_bm25[topic][fileName],train_cos[topic][fileName],
                                     train_freq[topic][fileName]]
                trainX[topic].append(value)
            except:
                trainX[topic].append([0,0,0])
                
        testX[topic]=list()
        for fileID in DTest[topic.upper()]:
            fileName=fileID+"newsML.xml"
            try :
                value=[test_bm25[topic][fileName],test_cos[topic][fileName],
                                    test_freq[topic][fileName]]
                testX[topic].append(value)
            except:
                testX[topic].append([0,0,0])
    return trainX, testX

def setupEvalOneFeature(search_topics, DTrain, DTest,trainDocsIndex,testDocsIndex,topicsIndex):
    topics=topicsIndex.topics

    train_bm25=dict()
    test_bm25=dict()

    for topic in search_topics:
        train_bm25[topic] =trainDocsIndex.generate_score(topics[topic.lower()],measure='bm25', k=None)
        test_bm25[topic]=testDocsIndex.generate_score(topics[topic.lower()],measure='bm25',k=None)

    trainX=dict()
    testX=dict()
    
    for topic in search_topics:
        trainX[topic]=list()
        for fileID in DTrain[topic.upper()]:
            fileName=fileID+"newsML.xml"
            try :
                value=[train_bm25[topic][fileName]]
                trainX[topic].append(value)
            except:
                trainX[topic].append([0])
                
        testX[topic]=list()
        for fileID in DTest[topic.upper()]:
            fileName=fileID+"newsML.xml"
            try :
                value=[test_bm25[topic][fileName]]
                testX[topic].append(value)
            except:
                testX[topic].append([0])
    return trainX, testX

def RRF(rankingsList):
    rrf = 0
    #for score in rankingsList:
    #        rrf += (1 / (50 + score))
    i=3
    for score in rankingsList:
        rrf+=i*score
        i-=1
    return rrf

def fscoreFunc(precision, recall):
    fscore=0
    if(precision == 0 or recall == 0):
        return fscore
    else:
        fscore= 2 / ( (1/recall) + (1/precision))
        return fscore
    
def BPREF(queryResult, relevant, nonRelevant):
    _sum = 0
    relevant=set(relevant)
    nonRelevant=set(nonRelevant)
    relevantSize = len(relevant)
    nonRelevantSize = len(nonRelevant.intersection(set(queryResult)))
    dividingFactor = min(relevantSize,nonRelevantSize)

    judgedAndRetrieved = set(queryResult).intersection(relevant)
    judgedAndRetrievedSize = len(judgedAndRetrieved)
    if(nonRelevantSize == 0):
        return 1
    for i in range(1,judgedAndRetrievedSize+1):
        judgedAndRelevantAndAk = (nonRelevant).intersection(set(queryResult[:i]))
        _sum += float(1 - (len(judgedAndRelevantAndAk)) / dividingFactor)
    return float(float(_sum) / relevantSize)

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

def MAP(retrieved, relevantFiles, nonRelevantFiles):
    precisionSum = 0
    k = len(retrieved)
    if(k == 0):
        return 0
    for i in range(1, k+1):
        if (retrieved[i-1] in relevantFiles):
            precisionSum += float(calcPrecision(retrieved[:i], relevantFiles, nonRelevantFiles))
    return float(precisionSum) / float(k)

def fscore(precision, recall):
    if(precision == 0 or recall == 0):
        return 0
    return 2 / ( (1/recall) + (1/precision))

def averagePrecision(precisionRecallCurve):
    total=0
    count=0
    #get the precision values from the precision recall curve
    for precision in precisionRecallCurve[1][1:]:
        total+=precision
        count+=1
    return (precision/count)

def calcPrecisionAtRecall(recall,retrieved,relevant,nonRelevant):
    if (recall==0):
        return 1;
    return calcPrecision(retrieved[0:recall],relevant,nonRelevant)

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

def plotPrecisionRecallCurve(recall,precision):
    if not precision:
        print("Weren't retrieved any documents with this type of ranking!")
        return None
    fig, ax = plt.subplots()
    for i in range(len(recall)-1):
        ax.plot((recall[i],recall[i]),(precision[i],precision[i+1]),'k-',color='red') #vertical\n",
        ax.plot((recall[i],recall[i+1]),(precision[i+1],precision[i+1]),'k-',color='red') #horizontal\n"
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    plt.axis([0.0,1.1,0.0,1.1])
    plt.show()

def savePrecisionRecallCurve(recall,precision,path):
    if not precision:
        print("Weren't retrieved any documents with this type of ranking!")
        return None
    fig, ax = plt.subplots()
    for i in range(len(recall)-1):
        ax.plot((recall[i],recall[i]),(precision[i],precision[i+1]),'k-',color='red') #vertical\n",
        ax.plot((recall[i],recall[i+1]),(precision[i+1],precision[i+1]),'k-',color='red') #horizontal\n"
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    plt.axis([0.0,1.1,0.0,1.1])
    plt.savefig(path)

def compute_metrics(docs_names, relevant, nonRelevant,k):
    precision = calcPrecision(docs_names[:k], relevant, nonRelevant)
    recall = calcRecall(docs_names[:k], relevant, nonRelevant)
    fscoreVal = fscoreFunc(precision,recall)
    precision_recall_curve=precisionRecallCurve(docs_names[:k],relevant,nonRelevant)
    bpref=BPREF(docs_names[:k], relevant, nonRelevant)
    #avg_precision=averagePrecision(precision_recall_curve)
    _MAP=MAP(docs_names[:k],relevant,nonRelevant)
    return precision,recall,fscoreVal,precision_recall_curve,bpref,_MAP

def get_worst_best_performance(IRmetrics):
    worst_precision=1
    worst_topic=None
    best_precision=-1
    best_topic=None
    for topic in IRmetrics:
        if topic!='MAP' and topic != 'Mean BPREF':
            precision=IRmetrics[topic][0]
            if precision<worst_precision:
                worst_topic=topic
                worst_precision=precision
            if precision>best_precision:
                best_topic=topic
                best_precision=precision
    output=[best_topic, best_precision,
            worst_topic, worst_precision]
    return output

def show_graphics_metrics_IRmodels(IRmodels,IRmodels_names,search_topics):
    #print(IRmodels)
    for topic in search_topics:
        for i in range(len(IRmodels)):
                print(IRmodels_names[i]," TOPIC ",topic)
                plotPrecisionRecallCurve(IRmodels[i][topic][3][0],IRmodels[i][topic][3][1])
    
    print("----------IR Models Summary----------\n")
    for i in range(len(IRmodels)): 
        best_worst=get_worst_best_performance(IRmodels[i])
        print(IRmodels_names[i],'\n')
        print('MAP:',IRmodels[i]['MAP'], ' Mean BPREF:',IRmodels[i]['Mean BPREF'],
              ' Best precision ',best_worst[1],' for ',best_worst[0],' Worst precision ',
              best_worst[3],' for ',best_worst[2],'\n')
    print('------------------------------------')
        
def show_metrics_classifiers(classifiers,classifiers_names,search_topics): 
    for topic in search_topics:
        for i in range(len(classifiers)):
                print(classifiers_names[i]," TOPIC ",topic)
                precision,recall,fscore,avg_prec_score=classifiers[i][topic]
                print("Precision:",precision," Recall:",recall," F-Score",fscore," AP:",avg_prec_score,'\n')
    
    print("---------- Classifiers Summary----------\n")
    for i in range(len(classifiers)): 
        print(classifiers_names[i],' ','MAP:',classifiers[i]['MAP'],'\n')
    print('------------------------------------')