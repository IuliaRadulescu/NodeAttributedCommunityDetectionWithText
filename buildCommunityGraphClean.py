from iLouvain import ILouvain
import pymongo
import numpy as np
from igraph import Graph, VertexClustering
from igraph import plot
from abc import ABC, abstractmethod
from sklearn.neighbors import kneighbors_graph

class CommunityGraphBuilder(ABC):

    def __init__(self, dataset):
        self.dataset = dataset
        self.g = Graph()

    @abstractmethod
    def getEdges(self):
        return

    @abstractmethod
    def getNodes(self):
        return

    def buildCommunityGraph(self, justNodesWithEdges = False):

        nodesList = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))

        # print(edgesList)

        self.g.add_vertices(nodesList)
        self.g.add_edges(edgesList)

        # remove nodes without edges
        if (justNodesWithEdges):
            nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
            self.g.delete_vertices(nodesToRemove)


class BuildAuthorsCommunityGraph(CommunityGraphBuilder):

    def __init__(self, dataset):
        super(BuildAuthorsCommunityGraph, self).__init__(dataset)
        
        # use a generic name for comments with no authors
        self.genericAuthorName = 'JhonDoe25122020'

    def buildCommunityGraph(self, kNNEnhance = False, justNodesWithEdges = False):

        (attributesDict, nodesList) = self.getNodes()

        print('Initial N nodes: ', len(nodesList))

        edgesList = self.getEdges()
        edgesList = list(filter(lambda x: x != None, edgesList))
        
        finalEdgesList = []

        for edge in edgesList:
            if ((edge[0], edge[1]) in finalEdgesList) or ((edge[1], edge[0]) in finalEdgesList) or edge[0] == edge[1]:
                continue
            finalEdgesList.append(edge)

        self.g.add_vertices(nodesList)

        for attributeName in attributesDict:
            self.g.vs[attributeName] = attributesDict[attributeName]

        if (kNNEnhance == True):
            kNNGraph = kneighbors_graph(self.g.vs['doc2vec'], 3, mode='connectivity', include_self=True)
            kNNGraph = kNNGraph.toarray()
            for i in range(len(kNNGraph[0]) - 1):
                for j in range(i+1, len(kNNGraph[0])):
                    if (kNNGraph[i][j] > 0):
                        authori = self.g.vs[i]['author'].tolist()
                        authorj = self.g.vs[j]['author'].tolist()

                        if ((authori, authorj) in finalEdgesList) or ((authorj, authori) in finalEdgesList):
                            continue
                        finalEdgesList.append((authori, authorj))

        self.g.add_edges(finalEdgesList)

        if justNodesWithEdges:
            nodesToRemove = [v.index for v in self.g.vs if v.degree() == 0]
            self.g.delete_vertices(nodesToRemove)

    def getEdges(self):

        def getEdgesFromRecord(record, authors):

            if (('parentAuthorId' not in record) or 
            (record['parentAuthorId'] not in authors) or
            ('author' not in record) or
            ('author' == False)):
                return None

            parentId = record['parentAuthorId']

            return (record['author'], parentId)

        authors = list(set(map(lambda x: x['author'], self.dataset)))

        return list(set(map(lambda x: getEdgesFromRecord(x, authors), self.dataset)))

    def getNodes(self):

        nodesList = list(map(lambda x: x['author'], self.dataset))

        nodesList = list(set(nodesList))
        nodesList = list(map(lambda x: self.genericAuthorName if x == False else x, nodesList))

        attributes = {'doc2vec': list(map(lambda x: np.array(x['doc2vec']), self.dataset)), 'author': list(map(lambda x: np.array(x['author']), self.dataset))}

        return (attributes, nodesList)

    def getGraph(self):
        return self.g

    def plotGraph(self, attributeField = 'clusterIdSmooth'):

        attributeDict = dict(zip([x['author'] for x in self.dataset], [x[attributeField] for x in self.dataset]))

        author2Indexes = dict(zip([node['name'] for node in self.g.vs], [node.index for node in self.g.vs]))

        for author in attributeDict:
            if author in author2Indexes:
                self.g.vs[author2Indexes[author]][attributeField] = attributeDict[author]
                
        clusters = VertexClustering(self.g, self.g.vs[attributeField])

        print('MODULARITY', clusters.modularity)

        plot(clusters)


class MongoDBClient:

    __instance = None

    def __init__(self):

        if MongoDBClient.__instance is not None:
            raise Exception('The MongoDBClient is a singleton')
        else:
            MongoDBClient.__instance = self

    @staticmethod
    def getInstance():
        
        if MongoDBClient.__instance is None:
            MongoDBClient()

        return MongoDBClient.__instance

    def updateAuthors(self, authors, clusterId, collectionName, attrName = 'clusterIdSimple'):

        self.dbClient = pymongo.MongoClient('localhost', 27017)

        db = self.dbClient.communityDetectionWimbledon

        db[collectionName].update_many(
            {
            'author': {
                '$in': authors
                }
            },{
                '$set': {
                    attrName: clusterId
                }
            })
            
        self.dbClient.close()

def applyLouvainModule(collectionName, g):

    # and also show modularity
    clusters = g.community_multilevel()
    modularity_score = g.modularity(clusters.membership)

    # plot(clusters)

    print('The modularity is ', modularity_score)

    # updateClusters(clusters, collectionName, 'clusterIdKnnEnhance')

def updateClusters(clusters, collectionName, attrName = 'clusterIdSimple'):

    subgraphs = clusters.subgraphs()

    clusterId = 0

    for subgraph in subgraphs:
        authors = []
        for vertex in subgraph.vs:
            authors.append(vertex['name'])
        MongoDBClient.getInstance().updateAuthors(authors, clusterId, collectionName, attrName)
        clusterId += 1

dbClient = pymongo.MongoClient('localhost', 27017)
db = dbClient.communityDetectionWimbledon

def getCommentsCommunity(collectionName, kNNEnhance = False, justNodesWithEdges = False):

    allComments = list(db[collectionName].find())

    # filter empty comments (just emojis or links)
    allComments = [comment if bool(comment['preprocessedBody']) != False else False for comment in allComments]
    allComments = list(filter(bool, allComments))

    commentsCommunity = BuildAuthorsCommunityGraph(allComments)
    commentsCommunity.buildCommunityGraph(kNNEnhance, justNodesWithEdges)

    return commentsCommunity

def getAllCollections(prefix, startWithCollection = False):

    def filterCollections(c, prefix, startWithCollection):
        startWithPrefix = prefix in c

        if (startWithCollection == False):
            return startWithPrefix
        
        return startWithPrefix and (int(c.split('_')[1]) >= int(startWithCollection.split('_')[1])) and (int(c.split('_')[2]) >= int(startWithCollection.split('_')[2])) and (int(c.split('_')[3]) >= int(startWithCollection.split('_')[3]))

    allCollections = db.list_collection_names()

    prefix = 'twelveHours'
    allCollections = list(filter(lambda c: filterCollections(c, prefix, startWithCollection), allCollections))

    return sorted(allCollections)

def applySimpleLouvain():

    allCollections = getAllCollections('twelveHours')

    for collectionName in allCollections:

        print('===> Running Louvain on ', collectionName)

        community = getCommentsCommunity(collectionName)

        applyLouvainModule(collectionName, community.getGraph())

def applyInertiaLouvain():
    
    allCollections = getAllCollections('twelveHours')

    for collectionName in allCollections:

        print('===> Running Inertia Louvain on ', collectionName)

        community = getCommentsCommunity(collectionName)
        communityGraph = community.getGraph()

        iLouvain = ILouvain()
        nodes2communities = dict(zip(range(len(communityGraph.vs)), range(len(communityGraph.vs))))
        nodeId2Doc2Vec = dict(zip(range(len(communityGraph.vs)), communityGraph.vs['doc2vec']))

        iLouvain.louvain(np.array(list(communityGraph.get_adjacency())), nodes2communities, nodeId2Doc2Vec)

        partition = VertexClustering(communityGraph, list(nodes2communities.values()))

        print('Real modularity:', partition.modularity)

        # updateClusters(partition, collectionName, 'clusterIdILouvain')

def applyKnnEnhance():
    allCollections = getAllCollections('twelveHours')

    for collectionName in allCollections:
        print('===> Running Louvain on ', collectionName)

        community = getCommentsCommunity(collectionName, True)

        applyLouvainModule(collectionName, community.getGraph())

def plotCollection(collectionName, attributeField):

    community = getCommentsCommunity(collectionName, False)
    community.plotGraph(attributeField)

def getSharedAuthorsStats():

    allCollections = getAllCollections('twelveHours')
    commonAuthorsNr = []

    for collectionId1 in range(len(allCollections)):
        for collectionId2 in range(len(allCollections)):
            if (collectionId1 < collectionId2):
                authors1 = db[allCollections[collectionId1]].distinct('author')
                authors2 = db[allCollections[collectionId2]].distinct('author')
                commonAuthorsNr.append(len(list(set(authors1).intersection(set(authors2)))))

    return (min(commonAuthorsNr), sum(commonAuthorsNr)/len(commonAuthorsNr), max(commonAuthorsNr))

applyKnnEnhance()
# applyInertiaLouvain()
# applySimpleLouvain()