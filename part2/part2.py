from logistic import LogisticClassifier
from xGBoost import XGBOOSTClassifier
from MLP import MLPerceptronClassifier
from knn import KNNClassifier
from util import getCollection,setupEvalManyFeatures,setupEvalOneFeature
from util import getRelevantNonRelevant,compute_metrics,RRF
from util import show_graphics_metrics_IRmodels,show_metrics_classifiers
from statistics import mean
from woosh_index import WooshDocumentIndex,TopicIndex
from hyper_parameter_tuning import hyper_parameter_search

def training(q,DTrain,RTrain,**args):
    '''
    @input topic document q ∈ Q, training collection Dtrain, judgments Rtrain, and
    optional arguments on the classification process
    @behavior learns a classification model to predict the relevance of documents on the
    topic q using Dtrain and Rtrain, where the training process is subjected to
    proper preprocessing, classifier’s selection and hyperparameterization
    @output q-conditional classification model
    '''
    classifier_type = args.get('classifier_type')
    if classifier_type=='logistic':
        hyper_parameters=args.get('hyper_paremeters')
        classifier=LogisticClassifier(hyper_parameters=hyper_parameters)
    elif classifier_type=='XGBOOST':
        classifier=XGBOOSTClassifier()
    elif classifier_type=='MLP':
        classifier=MLPerceptronClassifier()
    elif classifier_type=='KNN':
        classifier=KNNClassifier()
    classifier.train(q,DTrain,RTrain)
    return classifier

def classify(d,q,M,**args):
    '''
    @input document d ∈ Dtest, topic q ∈ Q, and classification model M
    @behavior classifies the probability of document d to be relevant for topic q given M
    @output probabilistic classification output on the relevance of document d to the
    topic t
    '''
    classifier=M[q]
    return classifier.classify(d,q)
    
def evaluate(Qtest,DTest,**args):
    '''
    @input subset of topics Qtest ⊆ Q, testing document collection Dtest, judgments
    Rtest, and arguments for the classification and retrieval modules
    @behavior evaluates the behavior of the IR system in the presence and absence of
    relevance feedback. In the presence of relevance feedback, training and
    testing functions are called for each topic in Qtest for a more comprehensive assessment
    @output performance statistics regarding the underlying classification system and
    the behavior of the aided IR system
    '''
    RTrain=args.get('RTrain')
    RTest=args.get('RTest')
    trainX=args.get('trainX')
    testX=args.get('testX')
    classifier_type=args.get('classifier_type')
    hyper_parameters=args.get('hyper_parameters')
    Model=dict()
    #metrics
    aidedRanked=dict()
    aidedNonRanked=dict()
    nonAided = dict()
    classifier_metrics=dict()
    #k used for retrieval
    k=args.get('k')
    ranking_type=args.get('ranking_type')
    for topic in Qtest:
        relevant,nonRelevant=getRelevantNonRelevant(topic)
        '''
        Non-Aided IR
        '''
        ranked_docs_names=[name for score, name in sorted(zip(testX[topic],DTest[topic]), 
                            key=lambda pair: RRF(pair[0]),reverse=True)]
        precision,recall,fscoreVal,precision_recall_curve,bpref,avg_prec=compute_metrics(ranked_docs_names
                                                                          ,relevant, nonRelevant,k)
        nonAided[topic] = [precision, recall, fscoreVal,precision_recall_curve,bpref,avg_prec]
        '''
        Train the model
        '''
        try:
            if(hyper_parameters):
                Model[topic]=training(topic,trainX,RTrain,classifier_type=classifier_type,
                     hyper_paremeters=hyper_parameters[topic])
            else:
                print(hyper_parameters)
                Model[topic]=training(topic,trainX,RTrain,classifier_type=classifier_type,
                     hyper_paremeters=None)
            aidedRanked[topic],aidedNonRanked[topic],classifier_metrics[topic]=Model[topic].evaluate(topic,
                          DTest,RTest,k = k,testX=testX,ranking_type=ranking_type,
                          relevant=relevant,nonRelevant=nonRelevant)
        except ValueError:
            print("For topic ", topic
                  ,"the classifier needs samples of at least 2 classes in the data"
                  , "but the data contains only one class: 1")
            aidedNonRanked[topic]=nonAided[topic]
            aidedRanked[topic]=nonAided[topic]
            #values from Non Aided
            classifier_metrics[topic]=[precision,recall,fscoreVal,avg_prec]
        
    '''
    Calculate Average values for the metrics
    '''
    nonAided['MAP'] = mean([nonAided[topic][5] for topic in nonAided])
    nonAided['Mean BPREF'] = mean([nonAided[topic][4] for topic in nonAided if topic != 'MAP'])
    
    aidedRanked['MAP']= mean([aidedRanked[topic][5] for topic in aidedRanked])
    aidedRanked['Mean BPREF']= mean([aidedRanked[topic][4] for topic in aidedRanked if topic != 'MAP'])
    
    aidedNonRanked['MAP']= mean([aidedNonRanked[topic][5] for topic in aidedNonRanked])
    aidedNonRanked['Mean BPREF']= mean([aidedNonRanked[topic][4] for topic in aidedNonRanked if topic != 'MAP'])

    classifier_metrics['MAP']=mean([classifier_metrics[topic][3] for topic in classifier_metrics])    
    return aidedRanked,aidedNonRanked,nonAided,classifier_metrics



'''
Setup
'''
#Qtest=['R'+str(i) for i in range(101,201,1)]
Qtest=['R103','R102','R142','R145','R131','R145','R198']#,'R145']#,'R106','R107','R195','R175']

trainfiles,testfiles,DTrain,RTrain,DTest,RTest=getCollection(Qtest)
trainDocsIndex=WooshDocumentIndex(load=True,dir_name="trainIndex",files=trainfiles)
testDocsIndex=WooshDocumentIndex(load=True,dir_name="testIndex",files=testfiles)
topicIndex=TopicIndex("topics.txt","topicsindexdir")

features='many'#features='one' for only one feature in the vector (bm25)
if features=='many':
    trainX, testX = setupEvalManyFeatures(Qtest, DTrain, DTest,
                              trainDocsIndex,testDocsIndex,topicIndex)
elif features=='one':
    trainX, testX = setupEvalOneFeature(Qtest, DTrain, DTest,
                                  trainDocsIndex,testDocsIndex,topicIndex)
 
'''
Training
'''
'''
classifierModel=dict()
classifierModel['R103']=training('R103',trainX,RTrain,classifier_type='logistic')
classifierModel['R102']=training('R102',trainX,RTrain,classifier_type='logistic')
'''
'''
Classify
'''
'''
document1_name=DTrain['R103'][0]#choose first document in the collection
document1_features=trainX['R103'][0]#pick the feature representation
probabilities1=classify(document1_features,'R103',classifierModel)
print('DOCUMENT WITH ID ',document1_name, ' with probabilities ', probabilities1,'for topic R103')
document2_name=DTrain['R103'][1]
document2_features=trainX['R103'][1]
probabilities2=classify(document2_features,'R103',classifierModel)
print('DOCUMENT WITH ID ',document2_name, ' with probabilities ', probabilities2,'for topic R103')
'''
'''
Evaluate
'''
k=10
IRmodels=[]
IRmodels_names=[]
classifiers=[]
classifiers_names=[]
'''
Ranking by Probability
'''
hyper_parameters=None

l_aidedRanked,l_aidedNonRanked,l_nonAided,l_classifier_metrics=evaluate(Qtest,DTest,
    trainX=trainX,testX=testX,RTrain=RTrain,RTest=RTest,k=k,classifier_type='logistic',ranking_type='proba',
    hyper_parameters=hyper_parameters)

IRmodels+=l_nonAided,l_aidedNonRanked,l_aidedRanked
IRmodels_names+='Not Aided',' Logistic Aided Not Ranked without Hyper Paremeter Tuning','Logistic Aided Ranked By Probability without Hyper Parameter Tuning'
classifiers.append(l_classifier_metrics)
classifiers_names.append('Logistic Classifier without Hyper Parametrization')
'''
Hyper parameter Tuning
'''
hyper_parameters=hyper_parameter_search('logistic',trainX,RTrain)

l_aidedRanked,l_aidedNonRanked,l_nonAided,l_classifier_metrics=evaluate(Qtest,DTest,
    trainX=trainX,testX=testX,RTrain=RTrain,RTest=RTest,k=k,classifier_type='logistic',ranking_type='proba',
    hyper_parameters=hyper_parameters)

IRmodels+=l_aidedNonRanked,l_aidedRanked
IRmodels_names+='Logistic Aided Not Ranked with Hyper Paremeter Tuning','Logistic Aided Ranked By Probability with Hyper Parameter Tuning'
classifiers.append(l_classifier_metrics)
classifiers_names.append('Logistic Classifier with Hyper Parametrization')

'''
Ranked by Score
'''
x_aidedRanked,x_aidedNonRanked,x_nonAided,x_classifier_metrics=evaluate(Qtest,DTest,
    trainX=trainX,testX=testX,RTrain=RTrain,RTest=RTest,k=k,classifier_type='XGBOOST',ranking_type='score')

IRmodels+=x_aidedNonRanked,x_aidedRanked
IRmodels_names+='XGBoost Aided Not Ranked', 'XGBoost Aided Ranked By Probability'
classifiers.append(x_classifier_metrics)
classifiers_names.append('XGBOOST Classifier')

show_graphics_metrics_IRmodels(IRmodels,IRmodels_names,Qtest)

show_metrics_classifiers(classifiers,classifiers_names,Qtest)