# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 12:06:18 2020

@author: diogo
"""

from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import precision_recall_fscore_support,average_precision_score
from util import compute_metrics, RRF

class LogisticClassifier():
    def __init__(self,**args):
        hyper_parameters=args.get('hyper_parameters')
        if(hyper_parameters!=None):
            solver=hyper_parameters['solver']
            C=hyper_parameters['C']
            penalty=hyper_parameters['penalty']
            self.clf = LogisticRegression(
                    random_state=1, solver=solver,C=C,penalty=penalty,multi_class='multinomial')
        else:
            self.clf=LogisticRegression(random_state=1,solver='lbfgs',multi_class='multinomial')
        
    def train(self,topic,DTrain,RTrain,**kwargs):
        DTrain=DTrain[topic]
        RTrain=RTrain[topic]
        self.clf.fit(DTrain,RTrain)
        
    
    def classify(self,doc,topic,**kwargs):
        return (self.clf.predict_proba([doc])[0])
                    
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