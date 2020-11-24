#!/bin/python3
#Implementation of Kmeans Unsupervised Learning
""" Import neccessary libraries """
import numpy as np
import math
import random
import copy



class kmeans():
    """ Create an instance of the objects """
    def __init__(self, dataset, numOfCentroids, label = False, seed_value = 0):
        random.seed(seed_value)         # Seed Random Vlue
        self._numK = numOfCentroids     # Number of Centroids
        self.centroids = {}             # Initialize Empty Centroids
        self._numOfDataset = len(dataset) # Number of data in the dataset             
        self._data = dataset
        if label:                        # If dataset is labelled
            self.raw = dataset           # Save dataset with label
            self._data = dataset[:, :dataset.shape[1]-1] # Remove label for operation
        self._chooseCentroids()          # Call Method for choosing Centrods Randomly
    
    def _chooseCentroids(self):
        """ Choose Random Centroids from DataSet"""
        for i in range(self._numK):
            self.centroids[i] = random.choice(self._data)

        return self.centroids

    def _calculateDistance(self, data = None):
        """ calculate the distance of centroids from every
            data in the DataSet"""
        distances = []
        for i in range(self._numOfDataset):
            new = []
            for j in range(self._numK):
                summation = 0
                for k in range(self._data.shape[1]):
                    """ Calculate distance using euclidean Distance """
                    summation += math.sqrt(((self._data[i, k] - self.centroids[j][k]))**2)
                new.append(summation)
            new.append(i)
            distances.append(new)

        return distances

    def _selectCloseData(self, distances):
        """ Select data that is closer to each centroids """
        centroids = {}
        for data in distances:
            index = data.index(min(data[0:self._numK]))
    
            if index not in centroids:
                centroids[index] = [data[-1]]
            else:
                centroids[index].append(data[-1])
            
        return centroids


    def _findNewCentroids(self, centroids):
        """ Find new Centroids by calculating the mean of data closer to each centroids"""
        for key in centroids:
            self.centroids[key] = np.mean(self._data[centroids[key]], axis = 0)
            
            

    def _decisiondistance(self, data, label = False):
        """ helper method for calculating centroid closer to query data """
        if label:
            data = data[:-1]
        least = float("inf")
        label = None
        for key in self.centroids:
            summation = 0
            for k in range(len(data)):
                summation += math.sqrt(((data[k] - key[1][k]))**2)
            if summation < least:
                least = summation
                label = key[0]

        return label

        

    def performOperation(self, iteration = 1):
        """ Method for performing the Operations and return new centroid
            1. Calculate Distance of each data to each centroids
            2. Choose closest data to the cencroids
            3. Find new Centroid by calling the Method
        """
        for _ in range(iteration):
            distances = self._calculateDistance()
            centroids = self._selectCloseData(distances)
            self._findNewCentroids(centroids)
        return centroids


    def createCluster(self, Label = True):
        """ Create Cluster and return assigned Label 
            if data is labeled, else return centroids and vectors
        """
        centroids = copy.deepcopy(self.centroids)
        distances = self.performOperation()
        while not self.equal(self.centroids, centroids):
            centroids = copy.deepcopy(self.centroids)
            distances = self.performOperation()

        if not Label:
            return self
        return self.assignLabel(distances)

    def equal(self, first, second):
        """ helper method to check if centroids has converged
            i.e there is no more changes to centroid assignment
        """
        for key in first:
            if not np.array_equal(first[key], second[key]):
                return False
        return True


    def assignLabel(self, distances):
        """ Assign Label by using the majority in each centroids
        """
        classifyer = []
        for key in distances.keys():
            new = self.raw[distances[key]]
            (values, counts) = np.unique(new[:,-1],return_counts=True)
            index = np.argmax(counts)
            grouping = self.centroids.pop(key)
            classifyer.append((values[index], grouping))

        self.centroids = classifyer
        return self


    def makeDecision(self, data, label = False):
        return self._decisiondistance(data, label)




