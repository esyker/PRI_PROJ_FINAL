# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 16:29:28 2020

@author: diogo
"""

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression  
from scipy.stats import loguniform

def hyper_parameter_search(classifier_type,trainX,RTrain):
    hyper_params=dict()
    space = dict()
    space['solver'] = ['lbfgs']
    space['penalty'] = ['none', 'l2']
    space['C'] = loguniform(1e-5, 100)
    model = LogisticRegression(multi_class='multinomial')
    #perform topic-conditional hyper-parameter search
    for topic in trainX:
        if topic != 'R175':
            if classifier_type=='logistic':
                search = RandomizedSearchCV(model, space, n_iter=100, scoring='accuracy', n_jobs=-1, random_state=1)
                # execute search
                result = search.fit(trainX[topic], RTrain[topic])
                # summarize result
                print('Best Score: %s' % result.best_score_)
                print('Best Hyperparameters: %s' % result.best_params_)
                hyper_params[topic]=result.best_params_
    return hyper_params
            