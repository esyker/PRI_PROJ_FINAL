from util import getEvaledDocsForTopic,getAllFiles,noPreprocessTopics,getCollection,getEvaledDocs
from util import setupEvalManyFeatures,setupEvalOneFeature
from util import plotPrecisionRecallCurve
from cluster import clustering,interpret,cohesion,separation,evaluation
from woosh_index import WooshDocumentIndex, TopicIndex
#from cluster import *
from PageRank import *
from knn import KNNClassifier
from logistic import LogisticClassifier
from xGBoost import XGBOOSTClassifier
from neural_network import NNClassifier
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
import time
from util import RRF

def testClustering1():
    D = []
    for i in ['R101' , 'R121', 'R150', 'R170', 'R180']:
        D += getEvaledDocsForTopic('qrels.test', i, 'test')
    description, clusters, vectorspace, labels, distances = clustering(D, 'docs', number_clusters = 11)
    
    medoid, mean = interpret(clusters[0], D, distances, 'docs')
    print("Medoid: ", medoid)
    print("Mean: ", mean)
    sil_score, avgCoheision, avgSeparation = evaluation(D, vectorspace, labels, distances)
    print("Silhouette: ", sil_score)
    print("AvgCoheision: ", avgCoheision)
    print("AvgSeparation: ", avgSeparation)
    
    cohesionValues = cohesion(labels, distances)
    separationValues = separation(labels, distances)

    print(cohesionValues)
    print(separationValues)

    plt.title('Cohesions values distribuition for document collection')
    plt.xlabel('Cohesion')
    plt.ylabel('Count')
    plt.hist(list(cohesionValues.values()), bins = 11)
    plt.show()

    plt.title('Separation values distribuition for document collection')
    plt.xlabel('Separation')
    plt.ylabel('Count')
    plt.hist(list(separationValues.values()), bins = 11)
    plt.show()

def testClustering2():
    print("Getting docs...")
    #D = list(getEvaledDocs('qrels.train'))
    D = getAllFiles('./rcv1/D_train/')
    description, clusters, vectorspace, labels, distances = clustering(D, 'docs', number_clusters = 100)
    medoid, mean = interpret(clusters[0], D, distances, 'docs')
    print("Medoid: ", medoid)
    print("Mean: ", mean)
    sil_score, avgCoheision, avgSeparation = evaluation(D, vectorspace, labels, distances)
    print("Silhouette: ", sil_score)
    print("AvgCoheision: ", avgCoheision)
    print("AvgSeparation: ", avgSeparation)
    cohesionValues = cohesion(labels, distances)
    separationValues = separation(labels, distances)


    plt.title('Cohesions values distribuition for Dtrain')
    plt.xlabel('Cohesion')
    plt.ylabel('Count')
    plt.hist(list(cohesionValues.values()), bins = 38)
    plt.show()

    plt.title('Separation values distribuition for Dtrain')
    plt.xlabel('Separation')
    plt.ylabel('Count')
    plt.hist(list(separationValues.values()), bins = 38)
    plt.show()

def testClusteringTopics():
    D = noPreprocessTopics('topics.txt')
    print(len(D))
    description, clusters, vectorspace, labels, distances = clustering(D, 'topics', number_clusters = 38)
    clusterToInterpret = list(filter(lambda x: 11 in x, clusters))
    print(clusterToInterpret)
    medoid, mean = interpret(np.array(clusterToInterpret), D, distances, 'topics')
    print("Medoid: ", medoid)
    print("Mean: ", mean)
    sil_score, avgCoheision, avgSeparation = evaluation(D, vectorspace, labels, distances)
    print("Silhouette: ", sil_score)
    print("AvgCoheision: ", avgCoheision)
    print("AvgSeparation: ", avgSeparation)
    cohesionValues = cohesion(labels, distances)
    separationValues = separation(labels, distances)

    plt.title('Cohesions values distribuition for topics')
    plt.xlabel('Cohesion')
    plt.ylabel('Count')
    plt.hist(list(cohesionValues.values()), bins = 38)
    plt.show()

    plt.title('Separation values distribuition for tpoics')
    plt.xlabel('Separation')
    plt.ylabel('Count')
    plt.hist(list(separationValues.values()), bins = 38)
    plt.show()


def testClassification(search_topics,k,load=False,features='many'):
    trainfiles,testfiles,DTrain,RTrain,DTest,RTest=getCollection(search_topics)
    
    start=time.time()
    trainDocsIndex=WooshDocumentIndex(load,"trainIndex",trainfiles)
    testDocsIndex=WooshDocumentIndex(load,"testIndex",testfiles)
    #testDocsIndex.index(testfiles)
    topicIndex=TopicIndex("topics.txt","topicsindexdir")
    end = time.time()
    totalTime=round(end-start,3)
    print(totalTime)
    
    if features=='many':
        trainX, testX = setupEvalManyFeatures(search_topics, DTrain, DTest,
                                  trainDocsIndex,testDocsIndex,topicIndex)
    elif features=='one':
        trainX, testX = setupEvalOneFeature(search_topics, DTrain, DTest,
                                      trainDocsIndex,testDocsIndex,topicIndex)
    #topic='R143'
    #print("TOPIC",topic)
    #toprint=sorted(zip(testX[topic],DTest[topic]),key=lambda pair: RRF(pair[0]),reverse=True)
    #print(toprint)
    #print("KEYS")
    #print(test_bm25.keys())
    #for doc_id in DTest[topic]:
    #    doc_name=doc_id+"newsML.xml"
    #    print(doc_name)
    #    print(doc_name," BM25",test_bm25[topic][doc_name]," COS "
    #          ,test_cos[topic][doc_name]," FREQ ",test_freq[topic][doc_name])
    
    start=time.time()
    #classifier=KNNClassifier(5)
    classifier1=LogisticClassifier()
    classifier2=XGBOOSTClassifier()
    #input_size=np.shape(trainX[search_topics[0]][0])[0]
    #classifier=NNClassifier(n_in=input_size,n_out=2,batch_size=64,learning_rate=0.5,verbose=False)
    log_classifier_metrics, log_nonAidedIROutput,log_aidedRanked,log_aidedNonRanked=classifier1.evaluate(search_topics,DTest,RTest,RTrain=RTrain
                               ,k=k,trainX=trainX,testX=testX,ranking_type='proba')
    
    x_classifier_metrics,x_nonAidedIROutput,x_aidedRanked,x_aidedNonRanked=classifier2.evaluate(search_topics,DTest,RTest,RTrain=RTrain
                               ,k=k,trainX=trainX,testX=testX,ranking_type='proba')

    #outputKNN=classifier.evaluate(search_topics,testX,RTest,DTrain=trainX,RTrain=RTrain)
    return log_classifier_metrics, log_nonAidedIROutput,log_aidedRanked,log_aidedNonRanked,x_classifier_metrics, x_nonAidedIROutput,x_aidedRanked,x_aidedNonRanked
    
def testPageRank():
    #D_train = getAllFiles('./rcv1/D_train')[:5000]
    D_train = getEvaledDocs('qrels.train')
    #graph = build_graph(D_train, 'tfidf', 0.2)
    #rankedDocs = undirected_page_rank('R130', D_train, 10, graph, variant='priors')
    #print(rankedDocs)
    #evaluatePageRank1(D_train, 10, 'tfidf', 0.6, variant='standard', iterations=20)
    #evaluatePageRank1(D_train, 10, 'tfidf', 0.2, variant='priors', iterations=20, d=0.6)
    #evaluatePageRank2(D_train, 10, 'tfidf', iterations = 20)
    evaluatePageRank3(D_train, 10, 'tfidf', 0.6, iterations = 20)




#testPageRank()

#classifier_metrics, nonAidedIROutput,aidedIROutput=testClassification2(load=False)
#testPageRank()
#search_topics=['R'+str(i) for i in range(181,201,1)]
search_topics=['R136','R137','R175']

log_classifier_metrics, log_nonAidedIROutput,log_aidedRanked,log_aidedNonRanked,x_classifier_metrics, x_nonAidedIROutput,x_aidedRanked,x_aidedNonRanked=testClassification(search_topics
                                            ,k=10,load=True,features='many')




print("----------------------\n",
      'LOGISTIC CLASSIFIER\n',
      '-----------------------\n\n')
for topic in search_topics:
    print("LOGISTIC MANY FEATURES TOPIC ",topic," NON-AIDED")
    plotPrecisionRecallCurve(log_nonAidedIROutput[topic][3][0],log_nonAidedIROutput[topic][3][1])
    print("LOGISTIC MANY FEATURES TOPIC ",topic," AIDED NON RANKED")
    plotPrecisionRecallCurve(log_aidedNonRanked[topic][3][0],log_aidedNonRanked[topic][3][1])
    print("LOGISTIC MANY FEATURES TOPIC ",topic," AIDED RANKED")
    plotPrecisionRecallCurve(log_aidedRanked[topic][3][0],log_aidedRanked[topic][3][1])

print('STATISTICS:')
print('CLASSIFIER')
print(log_classifier_metrics)
print('NON AIDED')
print(log_nonAidedIROutput)
print('AIDED NOT RANKED')
print(log_aidedNonRanked)
print('AIDED RANKED')
print(log_aidedRanked)

print("----------------------\n",
      'XGBOOST CLASSIFIER\n',
      '-----------------------\n\n')

for topic in search_topics:
    print("XGBOOST MANY FEATURES TOPIC ",topic," NON-AIDED")
    plotPrecisionRecallCurve(x_nonAidedIROutput[topic][3][0],x_nonAidedIROutput[topic][3][1])
    print("XGBOOST MANY FEATURES TOPIC ",topic," AIDED NON RANKED")
    plotPrecisionRecallCurve(x_aidedNonRanked[topic][3][0],x_aidedNonRanked[topic][3][1])
    print("XGBOOST MANY FEATURES TOPIC ",topic," AIDED RANKED")
    plotPrecisionRecallCurve(x_aidedRanked[topic][3][0],x_aidedRanked[topic][3][1])
    
print('STATISTICS:')
print('CLASSIFIER')
print(x_classifier_metrics)
print('NON AIDED')
print(x_nonAidedIROutput)
print('AIDED NOT RANKED')
print(x_aidedNonRanked)
print('AIDED RANKED')
print(x_aidedRanked)


feat_classifier_metrics,feat_nonAidedIROutput,feat_aidedRanked,feat_aidedNonRanked=testClassification(search_topics
                                            ,k=10,load=True,features='one')
for topic in search_topics:
    print("ONE FEATURE TOPIC ",topic," NON-AIDED")
    plotPrecisionRecallCurve(feat_nonAidedIROutput[topic][3][0],feat_nonAidedIROutput[topic][3][1])
    print("ONE FEATURE TOPIC ",topic," AIDED NON RANKED")
    plotPrecisionRecallCurve(feat_aidedNonRanked[topic][3][0],feat_aidedNonRanked[topic][3][1])
    print("ONE FEATURE TOPIC ",topic," AIDED RANKED")
    plotPrecisionRecallCurve(feat_aidedRanked[topic][3][0],feat_aidedRanked[topic][3][1])