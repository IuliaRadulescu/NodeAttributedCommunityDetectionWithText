import numpy as np
import time
from random import shuffle
import collections
import copy

'''
NOTE: This is a personal implmentation of the I-Louvain algorithm by Combe et al.

All credit goes to:

Combe, David, et al. "I-louvain: An attributed graph clustering method." International Symposium on Intelligent Data Analysis. Springer, Cham, 2015.
'''

class ILouvain():

    def initialize(self, node2community):
        community2nodes = {}
        for k, v in node2community.items():
            community2nodes[v] = community2nodes.get(v, []) + [k]

        return (node2community, community2nodes)

    '''
    Get the number of edges between node _node_ and community _communityId_
    '''
    def getNode2CommunityNumberOfEdges(self, node, community, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[community]
        onlyCommunityNodesWithEdgesToNode = list(filter(lambda x: graphAdjMatrix[node][x] > 0, communityNodes))
        return len(onlyCommunityNodesWithEdgesToNode)

    '''
    Get the sum of all nodes' degrees in community _communityId_
    '''
    def getCommunityAllNodeDegreesSum(self, community, graphAdjMatrix, community2nodes):
        communityNodes = community2nodes[community]
        return sum([self.getNodeDegree(communityNode, graphAdjMatrix) for communityNode in communityNodes])

    def getNodeDegree(self, node, graphAdjMatrix):
        nodeEdges = graphAdjMatrix[node]
        return np.sum(nodeEdges[nodeEdges > 0])

    '''
    Get all nodes interconnected with a node
    '''
    def getNodeNeighs(self, node, graphAdjMatrix, allNodes):
        return list(filter(lambda x: graphAdjMatrix[node][x] > 0, allNodes))

    def computeModularityGain(self, node, communityId, graphAdjMatrix, community2nodes):
        m = self.computeM(graphAdjMatrix)

        if (m == 0):
            return 0

        l_n_neighCommunity = self.getNode2CommunityNumberOfEdges(node, communityId, graphAdjMatrix, community2nodes)
        sum_neighCommunity = self.getCommunityAllNodeDegreesSum(communityId, graphAdjMatrix, community2nodes)
        k_n = self.getNodeDegree(node, graphAdjMatrix)

        return 1/m * (l_n_neighCommunity - (sum_neighCommunity * k_n) / (2 * m))

    def computeModularityGainInertia(self, node, community, community2nodes, distanceMatrix, node2Inertia, fullInertia):

        communitySum = 0

        for neigh in community2nodes[community]:
            if (node < neigh):
                distance = distanceMatrix[node][neigh]
            else:
                distance = distanceMatrix[neigh][node]

            communitySum += (node2Inertia[node] * node2Inertia[neigh]) / (2 * len(distanceMatrix) * fullInertia) - distance

        return (1 / (len(distanceMatrix) * fullInertia)) * communitySum

    def computeModularityInertiaUtils(self, nodeId2Doc2Vec, graphAdjMatrix):

        modularityMatrix = np.zeros((len(nodeId2Doc2Vec.keys()), len(nodeId2Doc2Vec.keys())))
        distanceMatrix = np.zeros((len(nodeId2Doc2Vec.keys()), len(nodeId2Doc2Vec.keys())))

        vectors = list(nodeId2Doc2Vec.values())

        fullInertia = self.getInertia(vectors)

        if fullInertia == float(0):
            return 0

        # cache inertias
        node2inertia = {}

        for nodeId in range(len(graphAdjMatrix[0])):
            node2inertia[nodeId] = self.getInertia(vectors, nodeId2Doc2Vec[nodeId])

        N = len(nodeId2Doc2Vec.keys())

        for i in range(len(graphAdjMatrix[0])):
            for j in range(len(graphAdjMatrix[0])):

                ijDistance = pow(np.linalg.norm(nodeId2Doc2Vec[i] - nodeId2Doc2Vec[j]), 2)
                distanceMatrix[i][j] = ijDistance

                a = (node2inertia[i] * node2inertia[j]) / (2 * N * fullInertia)
                b = ijDistance

                modularityMatrix[i][j] = a - b

        return (modularityMatrix, distanceMatrix, node2inertia, fullInertia)

    def moveNodeToCommunity(self, node, oldCommunity, newCommunity, community2nodes, node2community):
        node2community[node] = newCommunity
        community2nodes[oldCommunity].remove(node)
        community2nodes[newCommunity].append(node)
        return (node2community, community2nodes)

    '''
    Total number of edges in graph
    '''
    def computeM(self, graphAdjMatrix):
        m = 0

        for k in range(len(graphAdjMatrix[0])):
            m += np.sum(graphAdjMatrix[k, k:len(graphAdjMatrix[0])])

        return m/2

    def computeModularity(self, graphAdjMatrix, node2community):

        m = self.computeM(graphAdjMatrix)

        if (m == 0):
            return 0

        partialSums = []

        for i in range(len(graphAdjMatrix[0]) - 1):
            for j in range(i + 1, len(graphAdjMatrix[0])):
                if (node2community[i] == node2community[j]):
                    partialSums.append(graphAdjMatrix[i][j] - (self.getNodeDegree(i, graphAdjMatrix) * self.getNodeDegree(j, graphAdjMatrix)) / (2 * m))

        return sum(partialSums)/(2*m)

    def computeModularityInertia(self, graphAdjMatrix, modularityMatrixInertia, node2community, fullInertia):

        inertiaSum = 0

        for i in range(len(graphAdjMatrix[0]) - 1):
            for j in range(i + 1, len(graphAdjMatrix[0])):
                if (node2community[i] == node2community[j]):
                    inertiaSum += modularityMatrixInertia[i][j]

        inertiaSum = inertiaSum / (2 * len(graphAdjMatrix[0]) * fullInertia)

        print('INERTIA MODULARITY', inertiaSum)

        return inertiaSum

    def getCentroid(self, vectors):

        if (len(vectors) == 0):
            return np.array([])

        vectors = np.array(vectors)

        return vectors.mean(axis=0)

    def getInertia(self, vectors, g=None):

        # if reference (center of gravity) not provided, use centroid
        if (g is None):
            g = self.getCentroid(vectors)

        inertia = 0

        for vector in vectors:
            distance = pow(np.linalg.norm(vector - g), 2)
            inertia += distance

        return inertia

    def computeNewNode2Doc2Vec(self, community2nodes, nodeId2Doc2Vec):

        communities = list(filter(lambda x: len(community2nodes[x]) > 0, community2nodes.keys()))

        nodeId2Doc2VecTemp = {}

        for communityId in range(len(communities)):
            community = communities[communityId]
            vectors = []
            for node in community2nodes[community]:
                vectors.append(nodeId2Doc2Vec[node])
            nodeId2Doc2VecTemp[communityId] = self.getCentroid(vectors)

        return nodeId2Doc2VecTemp

    '''
    new2oldCommunities = contains mappings between current and prev step
    '''
    def computeNewAdjMatrix(self, community2nodes, new2oldCommunities, graphAdjMatrix):
        
        communities = list(community2nodes.keys())

        temporaryAdjMatrix = np.zeros((len(communities), len(communities)))

        for community1Id in range(len(communities)):
            for community2Id in range(len(communities)):
                community1 = communities[community1Id]
                community2 = communities[community2Id]
                temporaryAdjMatrix[community1Id][community2Id] = sum(self.interCommunitiesNodeWeights(community1, community2, graphAdjMatrix, community2nodes))

        newCommunityIterator = 0

        for community in community2nodes:
            # otherwise, replace it
            new2oldCommunities[newCommunityIterator] = community
            newCommunityIterator += 1
        
        return (temporaryAdjMatrix, new2oldCommunities)

    def interCommunitiesNodeWeights(self, community1, community2, graphAdjMatrix, community2nodes):

        interCommunitiesNodeWeights = []

        for i in community2nodes[community1]:
            for j in community2nodes[community2]:
                # don't parse same edge twice
                if (i >= j):
                    continue
                if (graphAdjMatrix[i][j] != 0):
                    interCommunitiesNodeWeights.append(graphAdjMatrix[i][j])

        return interCommunitiesNodeWeights

    def expandSuperNode(self, superNode, community2nodesFull):
        return community2nodesFull[superNode]

    def decompressSupergraph(self, community2nodes, community2nodesFull, new2oldCommunities):
        community2expandedNodes = {}
        node2communityOrdered = {}

        # take each super community and expand its super nodes (the super nodes are actually communities of nodes at the previous step)
        for superCommunity in community2nodes:
            oldCommunity = new2oldCommunities[superCommunity]
            expandedNodes = [self.expandSuperNode(new2oldCommunities[superNode], community2nodesFull) for superNode in community2nodes[superCommunity]]
            # flatten expanded nodes
            expandedNodes = [item for sublist in expandedNodes for item in sublist]
            community2expandedNodes[oldCommunity] = expandedNodes
            for node in community2expandedNodes[oldCommunity]:
                node2communityOrdered[node] = oldCommunity

        node2communityOrdered = collections.OrderedDict(sorted(node2communityOrdered.items()))

        return (node2communityOrdered, community2expandedNodes)

    def louvain(self, graphAdjMatrix, node2community, nodeId2Doc2Vec):

        start_time = time.time()

        theta = 0.0001
        alpha = 0.9

        isFirstPass = True

        while True:

            if isFirstPass:
                (node2community, community2nodes) = self.initialize(node2community)
                graphModularity = self.computeModularity(graphAdjMatrix, node2community)

                (modularityMatrixInertia, distanceMatrix, node2Inertia, fullInertia) = self.computeModularityInertiaUtils(nodeId2Doc2Vec, graphAdjMatrix)
                graphModularity += self.computeModularityInertia(graphAdjMatrix, modularityMatrixInertia, node2community, fullInertia)

            print('Started Louvain first phase')

            modularityFirstPhase = graphModularity

            while True:

                nodes = list(node2community.keys())

                shuffle(nodes)

                for node in nodes:

                    neighs = self.getNodeNeighs(node, graphAdjMatrix, nodes)

                    for neigh in neighs:

                        nodeCommunity = node2community[node]
                        neighCommunity = node2community[neigh]

                        if (neighCommunity == nodeCommunity):
                            continue

                        # try to move node
                        (node2communityTemp, community2nodesTemp) = self.moveNodeToCommunity(node, nodeCommunity, neighCommunity, \
                                                            copy.deepcopy(community2nodes), copy.deepcopy(node2community))

                        fullModularityGain = self.computeModularityGain(node, neighCommunity, graphAdjMatrix, community2nodesTemp) - \
                                             self.computeModularityGain(node, nodeCommunity, graphAdjMatrix, community2nodesTemp)

                        # computeModularityGainInertia(self, node, community, distanceMatrix, node2Inertia, fullInertia)
                        modularityGainWithInertia = self.computeModularityGainInertia(node, neighCommunity, community2nodesTemp, distanceMatrix, node2Inertia, fullInertia) - \
                                                    self.computeModularityGainInertia(node, nodeCommunity, community2nodesTemp, distanceMatrix, node2Inertia, fullInertia)

                        fullModularityGain = alpha * fullModularityGain + (1 - alpha) * modularityGainWithInertia

                        # if modularity gain is positive, perform move
                        if (fullModularityGain > 0):
                            (node2community, community2nodes) = (node2communityTemp, community2nodesTemp)

                newModularity = self.computeModularity(graphAdjMatrix, node2community) + self.computeModularityInertia(graphAdjMatrix, modularityMatrixInertia, node2community, fullInertia)

                print(modularityFirstPhase, newModularity)

                if (newModularity - modularityFirstPhase <= theta):
                    break
                
                modularityFirstPhase = newModularity

            print('Finished Louvain first phase')

            print('Start Louvain second phase')

            # filter communities with no nodes
            community2nodes = {i: j for i, j in community2nodes.items() if j != []}

            if isFirstPass:
                community2nodesFull = community2nodes
                node2communityFull = node2community
                # cache previous step configuration in case modularity decreases instead of increasing
                new2oldCommunities = dict(zip(community2nodes.keys(), community2nodes.keys()))
            else:
                # cache previous step configuration in case modularity decreases instead of increasing
                (node2communityFull, community2nodesFull) = self.decompressSupergraph(community2nodes, community2nodesFull, new2oldCommunities)

            # if modularity of the first phase is smaller than previous modularity, break
            if (modularityFirstPhase <= graphModularity):
                break
            
            graphModularity = modularityFirstPhase

            nodeId2Doc2Vec = self.computeNewNode2Doc2Vec(community2nodes, nodeId2Doc2Vec)
            (graphAdjMatrix, new2oldCommunities) = self.computeNewAdjMatrix(community2nodes, new2oldCommunities, graphAdjMatrix)
            nodes2communities = dict(zip(range(len(graphAdjMatrix[0])), range(len(graphAdjMatrix[0]))))
            (modularityMatrixInertia, distanceMatrix, node2Inertia, fullInertia) = self.computeModularityInertiaUtils(nodeId2Doc2Vec, graphAdjMatrix)

            (node2community, community2nodes) = self.initialize(nodes2communities)

            isFirstPass = False

        print("--- %s execution time in seconds ---" % (time.time() - start_time))

        return node2communityFull