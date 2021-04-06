from util import *
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from woosh_index import *

from statistics import mean
from woosh_index import *
from whoosh import scoring
from util import *
import matplotlib.pyplot as plt
import pprint


def build_graph(D, sim, teta, minDocFreq=2, maxDocFreq=0.9):
	vectorizer = None
	docs = noPreprocessDocs(D)
	if (sim == 'tfidf'):
		vectorizer = TfidfVectorizer(min_df= minDocFreq, max_df=maxDocFreq, stop_words='english')
	tfidf = vectorizer.fit_transform(docs)
	
	pairWiseDocSim = tfidf * tfidf.T
	pairWiseDocSim = pairWiseDocSim.toarray()

	graph = {
		'outlinks': {},
		'inlinks': {},
		'weights': {}
	}
	# the graph is already indirected
	for i in range(len(pairWiseDocSim)):
		for j in range(len(pairWiseDocSim[i])):
			if(i == j):
				continue
			if(pairWiseDocSim[i][j] > teta):

				if(graph.get('outlinks').get(D[i]) == None):
					graph['outlinks'][D[i]] = [] 
				graph['outlinks'][D[i]] += [D[j]]

				if(graph.get('inlinks').get(D[j]) == None):
					graph['inlinks'][D[j]] = []
				graph['inlinks'][D[j]] += [D[i]]

				if(graph.get('weights').get(D[i]) == None):
					graph['weights'][D[i]] = {}

				graph['weights'][D[i]][D[j]] = pairWiseDocSim[i][j]

	return graph



def undirected_page_rank(q, D, p, graph, variant='standard', priorWeighting = scoring.TF_IDF(), iterations = 10, d = 0.15):

	outDegrees = {}
	N = len(graph['outlinks'].keys())
	prestigeVector = {}


	topicIndex=TopicIndex("topics.txt","topicsindexdir")
	topics=topicIndex.topics

	trainIndex = WooshDocumentIndex(True, "trainIndex", D)
	Dids = list(map(lambda x: extractFileId(x)+"newsML.xml", D))


	for u in graph['outlinks'].keys():
		outDegrees[u] = len(graph['outlinks'][u])
		prestigeVector[u] = float(1/N)

	if (variant == 'standard'):
		for i in range(iterations):
			tempPrestigeVector = prestigeVector.copy() 
			for v in prestigeVector.keys():
				newValue = sum([prestigeVector[u] / outDegrees[u] for u in graph['inlinks'][v]])
				tempPrestigeVector[v] = d/N + (1 - d) * newValue
			prestigeVector = tempPrestigeVector.copy()


	elif (variant == 'priors'):

		priors = trainIndex.rank(topics[q.lower()], weighting=priorWeighting, k=None)
		fillVector(Dids, priors)
		normalizeDict(priors)
		for i in range(iterations):
			tempPrestigeVector = prestigeVector.copy() 
			for v in prestigeVector.keys():
				# For all inlinks of v
				newValue = 0
				for inlinkV in graph['inlinks'][v]:
					weightSum = sum([graph['weights'][inlinkV][ininlinkV] for ininlinkV in graph['inlinks'][inlinkV]])
					newValue += (prestigeVector[inlinkV] * graph['weights'][inlinkV][v] / weightSum)
				tempPrestigeVector[v] = d * priors[extractFileId(v)+"newsML.xml"] + (1 - d) * newValue

			prestigeVector = tempPrestigeVector.copy()
			
	orderedPrestigeVector = sorted(prestigeVector.items(), key=lambda x: x[1], reverse=True)
	orderedPrestigeVector = list(map(lambda x: (extractFileId(x[0]) + "newsML.xml", x[1]), orderedPrestigeVector))
	return orderedPrestigeVector[:p]
	
def evaluatePageRank1(D, p, sim, teta, variant='standard', minDocFreq=2, maxDocFreq=0.9, iterations = 10, d = 0.15):

	allTopics = ['R'+str(i) for i in range(101, 200+1)]
	#allTopics = ['R101']
	topicCount = len(allTopics)
	graph = build_graph(D, sim, teta, minDocFreq, maxDocFreq)


	topicIndex=TopicIndex("topics.txt","topicsindexdir")
	topicQueries=topicIndex.topics
	trainIndex = WooshDocumentIndex(True, "trainIndex", D)
	Dids = list(map(lambda x: extractFileId(x)+"newsML.xml", D))

	precisionSum1 = 0
	recallSum1 = 0
	fScoreSum1 = 0
	mapSum1 = 0
	bprefSum1 = 0

	precisionSum2 = 0
	recallSum2 = 0
	fScoreSum2 = 0
	mapSum2 = 0
	bprefSum2 = 0

	if (variant == 'standard'):
		pageRankResult = undirected_page_rank(None, D, p, graph, variant=variant, iterations=iterations)

	for topic in allTopics:
		normalRetrievalResult = trainIndex.rank(topicQueries[topic.lower()], weighting=scoring.TF_IDF(), k=None).items()
		normalRetrievalResult = sorted(normalRetrievalResult, key= lambda x: x[1], reverse=True)[:p]

		precisionScore, recallScore, fScore, mapScore, bprefScore, recallValues, precisionRecallCurve = calcMetrics(normalRetrievalResult, topic)
		savePrecisionRecallCurve(recallValues, precisionRecallCurve,'report/pagerank/q4/'+topic+'_1.png')


		precisionSum1 += precisionScore
		recallSum1 += recallScore
		fScoreSum1 += fScore
		mapSum1 += mapScore
		bprefSum1 += bprefScore

		if (variant == 'priors'):
			pageRankResult = undirected_page_rank(topic, D, p, graph, variant=variant)
		precisionScore, recallScore, fScore, mapScore, bprefScore, recallValues, precisionRecallCurve = calcMetrics(pageRankResult, topic)
		savePrecisionRecallCurve(recallValues,precisionRecallCurve,'report/pagerank/q4/'+topic+'_2.png')



		precisionSum2 += precisionScore
		recallSum2 += recallScore
		fScoreSum2 += fScore
		mapSum2 += mapScore
		bprefSum2 += bprefScore
		
	
	print("Normal")
	print("Average Precision: ", precisionSum1/topicCount)
	print("Average Recall: ", recallSum1/topicCount)
	print("Average FScore: ", fScoreSum1/topicCount)
	print("Average MAP: ", mapSum1/topicCount)
	print("Average BPREF: ", bprefSum1/topicCount)

	print("With Page Ranking")
	print("Average Precision: ", precisionSum2/topicCount)
	print("Average Recall: ", recallSum2/topicCount)
	print("Average FScore: ", fScoreSum2/topicCount)
	print("Average MAP: ", mapSum2/topicCount)
	print("Average BPREF: ", bprefSum2/topicCount)



def calcMetrics(result, topic):
	resultDocs = [i[0] for i in result]
	evaledDocs = getEvaledDocsForTopic('qrels.train', topic, 'train')
	relevance = getRelevanceForTopic('qrels.train', topic)
	relevant = [evaledDocs[i] for i in range(len(evaledDocs)) if relevance[i] == 1]
	nonRelevant = [evaledDocs[i] for i in range(len(evaledDocs)) if relevance[i] == 0]

	relevant = list(map(lambda x: extractFileId(x) + "newsML.xml", relevant))
	nonRelevant = list(map(lambda x: extractFileId(x) + "newsML.xml", nonRelevant))

	precision = calcPrecision(resultDocs, relevant, nonRelevant)
	recall = calcRecall(resultDocs, relevant, nonRelevant)
	fScoreValue = fscore(precision, recall)
	mapValue = MAP(resultDocs, relevant, nonRelevant)
	bpref = BPREF(resultDocs, relevant, nonRelevant)
	recallValues, precisionAtRecall = precisionRecallCurve(resultDocs, relevant, nonRelevant)
	return precision, recall, fScoreValue, mapValue, bpref, recallValues, precisionAtRecall

def evaluatePageRank2(D, p, sim, minDocFreq=2, maxDocFreq=0.9, iterations = 10, d=0.15):
	allTopics = ['R'+str(i) for i in range(101, 200+1)]
	#allTopics = ['R101']
	topicCount = len(allTopics)
	

	trainIndex = WooshDocumentIndex(True, "trainIndex", D)
	topicIndex = TopicIndex("topics.txt","topicsindexdir")

	precisions = []
	recalls = []
	fScores = []
	maps = []
	bprefs = []
	tetas = []

	teta = 0.1
	while (round(teta,2) <= 0.95):
		graph = build_graph(D, sim, teta, minDocFreq, maxDocFreq)
		precisionSum = 0
		recallSum = 0
		fScoreSum = 0
		mapSum = 0
		bprefSum = 0

		for topic in allTopics:
			pageRankResult = undirected_page_rank(None, D, p, graph, variant='standard')
			precisionScore, recallScore, fScore, mapScore, bprefScore, recallValues, precisionAtRecall = calcMetrics(pageRankResult, topic)

			precisionSum += precisionScore
			recallSum += recallScore
			fScoreSum += fScore
			mapSum += mapScore
			bprefSum += bprefScore

		averagePreccision = precisionSum/topicCount
		averageRecall = recallSum/topicCount
		averageFscore = fScoreSum/topicCount
		averageMap = mapSum/topicCount
		averageBref = bprefSum/topicCount

		precisions.append(averagePreccision)
		recalls.append(averageRecall)
		fScores.append(averageFscore)
		maps.append(averageMap)
		bprefs.append(averageBref)
		tetas.append(round(teta, 2))

		print("Teta: ", round(teta, 2))
		print("Average Precision: ", averagePreccision)
		print("Average Recall: ", averageRecall)
		print("Average FScore: ", averageFscore)
		print("Average MAP: ", averageMap)
		print("Average BPREF: ", averageBref)
		print()

		teta += 0.05

	plt.title("Variation of measures for different values of teta")
	plt.plot(tetas, precisions, label = "precision")
	plt.plot(tetas, recalls, label = "recall")
	plt.plot(tetas, fScores, label = "FScore")
	plt.plot(tetas, maps, label = "MAP")
	plt.plot(tetas, bprefs, label = "BPREF")

	plt.legend()
	plt.xlabel('teta')
	plt.ylabel('measures')
	plt.show()

def evaluatePageRank3(D, p, sim, teta, minDocFreq=2, maxDocFreq=0.9, iterations = 10, d = 0.15):
	allTopics = ['R'+str(i) for i in range(101, 200+1)]
	topicCount = len(allTopics)
	graph = build_graph(D, sim, teta, minDocFreq, maxDocFreq)
	pageRankResult = undirected_page_rank(None, D, p, graph, variant='standard')
	print(pageRankResult)

