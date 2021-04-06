from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from Preprocess import Preprocess
from xml_documents_parser import DocumentParser
from tqdm import tqdm
from scipy import sparse
from scipy.sparse import * 
from util import *
import matplotlib.pyplot as plt
import math
import functools
import sys
#from yellowbrick.cluster import KElbowVisualizer

def file_to_string(file):
    ptr=open(file)
    _str=ptr.read()        
    return _str 

def calcOptimNclusters(vectorspace, krange=(2,30)):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=krange, metric='silhouette', timings= True)
    visualizer.fit(vectorspace)        # Fit the data to the visualizer
    visualizer.show()
    exit()
                
def clustering(D, col, number_clusters, minDocFreq = 2, maxDocFreq = 0.9):
    if(col == 'docs'):
        D = noPreprocessDocs(D)
 
    vectorizer = TfidfVectorizer(use_idf = False, min_df= minDocFreq, max_df=maxDocFreq, stop_words='english')
    vectorspace = vectorizer.fit_transform(D)
    
    #print("Calculating silhouette...")
    #calcOptimNclusters(vectorspace, krange=(2,100))
    #calcOptimNclusters(vectorspace)

    kmeans = KMeans(n_clusters=number_clusters)
    labels = kmeans.fit_predict(vectorspace)
    centroids = kmeans.cluster_centers_
    distances = kmeans.fit_transform(vectorspace.toarray())

    pca = PCA(2)
    distances = pca.fit_transform(distances)

    uniqueLabels = np.unique(labels)
    clustersDescription = []
    clusters = []
    for i in uniqueLabels:
        clustersDescription.append(((centroids[:,0][i], centroids[:,0][i]), list(map(lambda x: D[x], list(np.where(labels == i)[0])))))
        clusters.append(np.where(labels == i)[0])
    return clustersDescription, clusters, vectorspace, labels, distances

def interpret(cluster, D, distances, col):
    nDistances = len(distances)
    distanceMatrix = {}

    for i in  range(nDistances):
        if i not in cluster: 
            continue
        pointDistances = []
        for j in range(nDistances):
            if j not in cluster: 
                continue
            pointDistances.append(pointDistance(distances[i], distances[j]))
        distanceMatrix[i] = pointDistances
    min = sys.maxsize
    medoid = 0
    for i in distanceMatrix.keys():
        j = float(functools.reduce(lambda x, y: x + y, distanceMatrix[i]))
        if j < min:
            min = j
            medoid = i
    mean = np.mean(list(distanceMatrix.values()))
    if(col == 'docs'):
        medoid = D[medoid]
    elif(col == 'topics'):
        medoid = 'R' + str(101+medoid)
    return medoid, mean ### DONT KNOW IF MEAN IS CORRECT ###
    


def cohesion(labels, distances):
    sse = 0
    sseDict = {}
    uniqueLabels = np.unique(labels)
    for label in uniqueLabels:
        sse = 0
        clusterDocs = np.where(labels == label)[0]
        for doc1 in clusterDocs:
            for doc2 in clusterDocs:
                sse += (pointDistance(distances[doc1], distances[doc2])/2) ** 2
        sseDict[label] = sse
    return sseDict

def separation(labels, distances):
    ssb = 0
    ssbDict = {}
    uniqueLabels = np.unique(labels)
    for label in uniqueLabels:
        ssb = 0
        clusterDocs = np.where(labels == label)[0]
        for doc1 in clusterDocs:
            nonClsuterDocs = np.where(labels != label)[0]
            for doc2 in nonClsuterDocs:
                ssb += (pointDistance(distances[doc1], distances[doc2])/2) ** 2
        ssbDict[label] = ssb
    return ssbDict





def evaluation(D, vectorspace, labels, distances):
    '''
    internal measure
    '''
    '''
    vectorspace = kwargs.get('vectorspace')
    model = kwargs.get('model')
    labels = kwargs.get('labels')
    if not vectorspace or not model or not labels:
        raise ValueError("vectorspace, model and labels must be defined as arguments")
    '''
    sil_score = silhouette_score(vectorspace, labels, metric = 'cosine')
    #print('silhouette score = {}'.format(sil_score))
    '''
    external measure
    '''
    '''
    ar_score = adjusted_rand_score(model.labels_, labels)
    print('adjusted rand score score = {}'.format(ar_score))
    '''
    avgCoheision = np.average(list(cohesion(labels, distances).values()))
    avgSeparation = np.average(list(separation(labels, distances).values()))
    return sil_score, avgCoheision, avgSeparation
