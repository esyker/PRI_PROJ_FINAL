# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 19:41:33 2020

@author: diogo
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_recall_fscore_support,average_precision_score
from util import compute_metrics, RRF

class KNNClassifier():
    def __init__(self):
        self.clf = KNeighborsClassifier(n_neighbors=5)
    
    def only_one_label_exception(self,RTrain):
        number_0s=0
        number_1s=0
        for label in RTrain:
            if label==0:
                number_0s+=1
            else:
                number_1s+=1
        if number_0s==0 or number_1s==0:
            raise ValueError
            
    def train(self,topic,DTrain,RTrain,**kwargs):
        self.only_one_label_exception(RTrain[topic])
        DTrain=DTrain[topic]
        RTrain=RTrain[topic]
        self.clf.fit(DTrain,RTrain)
        
    
    def classify(self,doc,topic,**kwargs):
        return self.clf.predict_proba([doc])[0]
                    
    def evaluate(self,topic,DTest,RTest,**kwargs):
        k = kwargs.get('k')
        testX=kwargs.get('testX')
        ranking_type=kwargs.get('ranking_type')
        relevant=kwargs.get('relevant')
        nonRelevant=kwargs.get('nonRelevant')
        '''
        Evaluating the classifier
        '''
        docs=testX[topic]
        feedback=RTest[topic]
        y_scores=self.clf.predict_proba(docs)
        y_pred=self.clf.predict(docs)
        precision,recall,fscore,true_sum = precision_recall_fscore_support(feedback, y_pred
                                                , average='macro',zero_division=1)
        avg_prec_score = average_precision_score(feedback,y_scores[:,1])
        classifier_metrics=[precision,recall,fscore,avg_prec_score]
        '''
        Binary retrieval
        '''
        scores_names=zip(y_scores,DTest[topic]) 
        positive_class_predicted=[doc for doc in scores_names if doc[0][1]>0.5]
        aided_non_ranked_docs_names=[doc[1] for doc in positive_class_predicted]
        '''
        Evaluating binary retrieval
        '''
        precision,recall,fscoreVal,precision_recall_curve,bpref,avg_prec=compute_metrics(aided_non_ranked_docs_names
                                                                          ,relevant, nonRelevant,k)
        aidedNonRanked=[precision,recall,fscoreVal,precision_recall_curve,bpref,avg_prec]
        '''
        Extension towards ranking
        '''
        if ranking_type=='proba':#sorts according to probabilities
            aided_ranked_docs_names=[x for _, x in sorted(zip(y_scores,DTest[topic]), 
                                                    key=lambda pair: pair[0][1],reverse=True)]
        else:#sort according to the score of the docs classified as positive
            aided_ranked_docs_names=[doc[1] for doc in sorted(positive_class_predicted,
                                           key=lambda x:RRF(x[0]),reverse=True)[:k]]
        '''
        Evaluating Aided IR
        '''
        precision,recall,fscoreVal,precision_recall_curve,bpref,avg_prec=compute_metrics(aided_ranked_docs_names
                                                                          ,relevant, nonRelevant,k)
        aidedRanked=[precision,recall,fscoreVal,precision_recall_curve,bpref,avg_prec]
        
        return aidedRanked,aidedNonRanked,classifier_metrics