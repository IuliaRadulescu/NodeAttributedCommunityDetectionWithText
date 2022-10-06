import pymongo
from datetime import datetime
import numpy as np
import re
import string
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from stop_words import get_stop_words
import contractions
import umap

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()

'''
text preprocessing pipeline - for a single unit of text corpus (a single document)
'''
class TextPreprocessor:

    @staticmethod
    def removeLinks(textDocument):
        return re.sub(r'(https?://[^\s]+)', '', textDocument)

    @staticmethod
    def removeEmojis(textDocument):
        emoj = re.compile("["
                          u"\U0001F600-\U0001F64F"  # emoticons
                          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                          u"\U0001F680-\U0001F6FF"  # transport & map symbols
                          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                          u"\U00002500-\U00002BEF"  # chinese char
                          u"\U00002702-\U000027B0"
                          u"\U00002702-\U000027B0"
                          u"\U000024C2-\U0001F251"
                          u"\U0001f926-\U0001f937"
                          u"\U00010000-\U0010ffff"
                          u"\u2640-\u2642"
                          u"\u2600-\u2B55"
                          u"\u200d"
                          u"\u23cf"
                          u"\u23e9"
                          u"\u231a"
                          u"\ufe0f"  # dingbats
                          u"\u3030"
                          "]+", re.UNICODE)
        return re.sub(emoj, '', textDocument)

    @staticmethod
    def removeRedditReferences(textDocument):
        return re.sub(r'(/r/[^\s]+)', '', textDocument)

    @staticmethod
    def removeCodeAndNonASCII(textDocument):
        # remove code snippets
        textDocument = re.sub(r'```.+?```', '', textDocument)
        textDocument = re.sub(r'``.+?``', '', textDocument)
        textDocument = re.sub(r'`.+?`', '', textDocument)

        # remove xmls
        textDocument = re.sub(r'<.+?>', '', textDocument)
        return re.sub('[^a-zA-Z0-9 \'?!,.]', '', textDocument)

    @staticmethod
    def stopWordRemoval(tokenizedDocument):
        finalStop = list(get_stop_words('english'))  # About 900 stopwords
        nltkWords = stopwords.words('english')  # About 150 stopwords
        finalStop.extend(nltkWords)
        finalStop = list(set(finalStop))

        # filter stop words and one-letter words/chars except i
        return list(filter(lambda token: (token not in finalStop), tokenizedDocument))

    @staticmethod
    def doLemmatization(tokenizedDocument):
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token) for token in tokenizedDocument]

    @staticmethod
    def doCleaning(textDocument):
        # make lower
        textDocument = textDocument.lower()
        # reddit specific preprocessing
        textDocument = TextPreprocessor.removeLinks(textDocument)
        textDocument = TextPreprocessor.removeEmojis(textDocument)
        textDocument = TextPreprocessor.removeRedditReferences(textDocument)
        textDocument = TextPreprocessor.removeCodeAndNonASCII(textDocument)
        # decontract
        textDocument = contractions.fix(textDocument)
        # remove remaining '
        textDocument = re.sub('[^a-zA-Z0-9 ?!,.]', '', textDocument)

        # tokenize
        tokenized = word_tokenize(textDocument)

        # filter empty
        tokenized = list(filter(lambda x: x.strip() != '', tokenized))

        if (len(tokenized) == 0):
            return False

        return ' '.join(tokenized)

    @staticmethod
    def doProcessing(textDocument):
        # remove everything that is not letter, number, or space
        textDocument = re.sub('[^a-zA-Z0-9 ]', '', textDocument)

        # tokenize
        tokenized = word_tokenize(textDocument)

        # remove stop words
        tokenizedNoStop = TextPreprocessor.stopWordRemoval(tokenized)

        finalTokens = [lemmatizer.lemmatize(token) for token in tokenizedNoStop]

        return finalTokens

class Enhancer:
    def __init__(self, collectionName, preprocessedDataset, redditIds, documentVectors, docs2Tags):
        self.collectionName = collectionName
        self.preprocessedDataset = preprocessedDataset
        self.redditIds = redditIds
        self.documentVectors = documentVectors
        self.docs2Tags = docs2Tags
        self.dimReducer = umap.UMAP(n_components=3, random_state=42)

    def computeDoc2VecEmbeddings(self):
        X = [self.documentVectors.get_vector(self.docs2Tags[self.collectionName + '_' + str(documentNr)], norm=True) for documentNr in range(len(self.preprocessedDataset))]
        return preprocessing.normalize(self.dimReducer.fit_transform(X))

    def updateNodesWithDoc2Vec(self):
        X = self.computeDoc2VecEmbeddings()

        redditId2X = {}

        for counter in range(len(self.redditIds)):
            redditId = self.redditIds[counter]
            redditId2X[redditId] = X[counter].tolist()

        MongoDBClient.getInstance().updateCommentsWithDoc2Vec(self.redditIds, redditId2X, self.collectionName)


class Clusterer:

    def __init__(self, collectionName, vectorEmbeddings, redditIds):
        self.collectionName = collectionName
        self.vectorEmbeddings = vectorEmbeddings
        self.redditIds = redditIds

    def computeCentroid(self, cluster):
        if not isinstance(cluster, np.ndarray):
            cluster = np.array(cluster)
        length, dim = cluster.shape
        centroid = np.array([np.sum(cluster[:, i]) / length for i in range(dim)])

        # return normalized centroid
        return (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid))

    def getClusteringLabels(self, noClusters, commentEmbeddings):
        kMeansClusterer = KMeans(n_clusters=noClusters)
        kMeansClusterer.fit(commentEmbeddings)

        len_ = np.sqrt(np.square(kMeansClusterer.cluster_centers_).sum(axis=1)[:, None])
        centers = kMeansClusterer.cluster_centers_ / len_

        return (kMeansClusterer.labels_, centers)

    def doClustering(self):

        allCommentsLen = len(self.vectorEmbeddings)

        # if just one comment, no need to perform clustering
        if (allCommentsLen == 1):
            return [{0: [0]}, {0: self.computeCentroid(self.vectorEmbeddings)}]

        # worst case values
        maxSilhouette = -2
        maxNoClusters = 1
        bestLabels = [0] * allCommentsLen
        bestCenters = []

        for noClusters in range(min(2, (allCommentsLen - 1)), min(6, allCommentsLen)):

            (labels, centers) = self.getClusteringLabels(noClusters, self.vectorEmbeddings)

            if (len(list(set(labels))) <= 1):
                continue

            sscore = silhouette_score(X=self.vectorEmbeddings, labels=labels, metric='cosine')

            if (sscore > maxSilhouette):
                maxSilhouette = sscore
                maxNoClusters = noClusters
                bestLabels = labels
                bestCenters = centers

        print('Best noClusters is', maxNoClusters, 'with score', sscore)

        clusterIds2Centroids = {}

        for clusterId in range(max(bestLabels) + 1):
            clusterIds2Centroids[clusterId] = bestCenters[clusterId]

        return bestLabels, clusterIds2Centroids

    def updateClusters(self, labels, centroids):

        clusters2RedditIds = {}

        for counter in range(0, len(self.redditIds)):

            label = labels[counter]
            redditId = self.redditIds[counter]

            if label not in clusters2RedditIds:
                clusters2RedditIds[label] = []

            clusters2RedditIds[label].append(redditId)

        for clusterId in clusters2RedditIds:
            cluster = centroids[clusterId]
            MongoDBClient.getInstance().updateCommentsWithClusters(clusters2RedditIds[clusterId], int(clusterId), cluster.tolist(), self.collectionName)

class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance != None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance == None:
            MongoDBClient()

        return MongoDBClient.__instance

    def updateCommentsWithDoc2Vec(self, redditIds, redditId2X, collectionName):
        self.dbClient = pymongo.MongoClient('localhost', 27017)

        db = self.dbClient.communityDetectionWimbledon

        for redditId in redditIds:
            db[collectionName].update_many(
                {
                    'redditId': redditId
                }, {
                    '$set': {
                        'doc2vec': redditId2X[redditId]
                    }
                })

        self.dbClient.close()

    def updateCommentsWithClusters(self, redditIds, clusterId, centroid, collectionName):

        self.dbClient = pymongo.MongoClient('localhost', 27017)

        db = self.dbClient.communityDetectionWimbledon

        db[collectionName].update_many(
            {
            'redditId': {
                '$in': redditIds
                }
            },{
                '$set': {
                    'clusterIdKMeans': clusterId,
                    'centroid': centroid
                }
            })

        self.dbClient.close()

def getAllCollections(db, prefix='twelveHours'):

    allCollections = db.list_collection_names()

    allCollections = list(filter(lambda x: prefix in x, allCollections))

    return sorted(allCollections)

def cleanDataset():
    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionWimbledon
    allCollections = getAllCollections(db)

    for collectionName in allCollections:
        print('Started cleaning', collectionName)
        allRecords = list(db[collectionName].find())
        comments = [x['body'] for x in allRecords]
        redditIds = [x['redditId'] for x in allRecords]
        cleanDataset = [TextPreprocessor.doCleaning(document) for document in comments]

        redditId2Clean = dict(zip(redditIds, cleanDataset))

        for redditId in redditIds:
            db[collectionName].update_many(
                {
                    'redditId': redditId
                }, {
                    '$set': {
                        'cleanBody': redditId2Clean[redditId]
                    }
                })
        print('Finished cleaning', collectionName)

    dbClient.close()

def preprocessDataset():
    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionWimbledon
    allCollections = getAllCollections(db)

    for collectionName in allCollections:
        print('Started preprocessing', collectionName)
        allRecords = list(db[collectionName].find())
        cleanComments = [x['cleanBody'] for x in allRecords]
        redditIds = [x['redditId'] for x in allRecords]
        preprocessedBody = [TextPreprocessor.doProcessing(document) if document != False else False for document in cleanComments]

        redditId2Preprocessed = dict(zip(redditIds, preprocessedBody))

        for redditId in redditIds:
            db[collectionName].update_many(
                {
                    'redditId': redditId
                }, {
                    '$set': {
                        'preprocessedBody': redditId2Preprocessed[redditId]
                    }
                })
        print('Finished preprocessing', collectionName)

    dbClient.close()

def getDoc2VecModel(computeModel = False):

    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionWimbledon

    allCollections = getAllCollections(db)

    # compute doc2vec model

    # create collections dictionaries
    collections2Documents = {}
    collections2RedditIds = {}
    allDocuments = []

    for collectionName in allCollections:
        allRecords = list(db[collectionName].find())
        preprocessedBodies = [x['preprocessedBody'] for x in allRecords]
        redditIds = [x['redditId'] if bool(x['preprocessedBody']) != False else False for x in allRecords]

        # remove false values
        preprocessedBodies = list(filter(bool, preprocessedBodies))
        redditIds = list(filter(bool, redditIds))

        allDocuments += preprocessedBodies

        collections2Documents[collectionName] = preprocessedBodies
        collections2RedditIds[collectionName] = redditIds

    dbClient.close()

    docs2Tags = {}

    documentsIterator = 0
    for collectionName in collections2Documents:
        for documentNr in range(len(collections2Documents[collectionName])):
            docs2Tags[collectionName + '_' + str(documentNr)] = str(documentsIterator)
            documentsIterator += 1

    # compute doc2vec model
    if (computeModel == True):
        print('Started computing model...')
        # 16 neurons (vector size) and 3 words window - because we have small documents
        doc2vecModel = computeDoc2VecModel(16, 3, allDocuments)
        # save model to file
        doc2vecModel.save('doc2VecTrainingWimbledon')
        print('Saved model computing model')

    else:
        doc2vecModel = Doc2Vec.load('doc2VecTrainingWimbledon')

    return (collections2Documents, collections2RedditIds, doc2vecModel, docs2Tags)

def enahnce(collections2Documents, collections2RedditIds, doc2vecModel, docs2Tags):
    for collectionName in collections2Documents:
        preprocessedDataset = collections2Documents[collectionName]
        redditIds = collections2RedditIds[collectionName]

        print('Enhancing collection', collectionName, 'with', len(preprocessedDataset), 'comments')

        doc2vecEnhancer = Enhancer(collectionName, preprocessedDataset, redditIds, doc2vecModel.dv, docs2Tags)

        doc2vecEnhancer.updateNodesWithDoc2Vec()

        print('Enhancing collection', collectionName, 'END ==')

def cluster():
    dbClient = pymongo.MongoClient('localhost', 27017)
    db = dbClient.communityDetectionWimbledon

    allCollections = getAllCollections(db)

    for collectionName in allCollections:
        print('Clustering collection', collectionName)

        allRecords = list(db[collectionName].find())
        redditIds = [x['redditId'] if bool(x['preprocessedBody']) != False else False for x in allRecords]
        vectorEmbeddings = [x['doc2vec'] if bool(x['preprocessedBody']) != False else False for x in allRecords]

        # remove false values
        redditIds = list(filter(bool, redditIds))
        vectorEmbeddings = list(filter(bool, vectorEmbeddings))

        clusterer = Clusterer(collectionName, vectorEmbeddings, redditIds)
        (labels, centroids) = clusterer.doClustering()
        clusterer.updateClusters(labels, centroids)

        print('Clustering collection', collectionName, 'END ==')

    dbClient.close()

'''
preprocessedDocuments = a list of lists of tokens; example = [ ['Lilly', 'is', 'beautiful', 'cat'], ['Milly', 'is', 'wonderful' 'cat'] ]
https://radimrehurek.com/gensim/models/doc2vec.html
'''
def computeDoc2VecModel(vectorSize, windowSize, allDocuments):
    documents = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(allDocuments)]
    return Doc2Vec(documents, vector_size=vectorSize, window=windowSize, epochs=50, dm=1, workers=20)

# cleanDataset()
# preprocessDataset()
# (collections2Documents, collections2RedditIds, doc2vecModel, docs2Tags) = getDoc2VecModel()
# enahnce(collections2Documents, collections2RedditIds, doc2vecModel, docs2Tags)
cluster()

