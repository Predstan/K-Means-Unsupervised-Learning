#!/usr/bin/env python3

import sys 
import numpy as np
from cluster import kmeans


if len(sys.argv) != 5:
    print()
    print("Usage: %s [Seed Value] [Number of Centroids] [training Data] [Test Data] " %(sys.argv[0]))
    print()
    sys.exit(1)


train, test = np.loadtxt(sys.argv[3]), np.loadtxt(sys.argv[4])
model = kmeans(train, int(sys.argv[2]) , label = True, seed_value= int(sys.argv[1]))
model.createCluster()
correct = 0
if test.ndim < 2:
    classification = model.makeDecision(test, label = True)
    print(classification, test[-1])
    if test[-1] == classification:
        correct+=1
else:
    for data in test:
        classification = model.makeDecision(data, label = True)
        print(classification, data[-1])
        if data[-1] == classification:
            correct+=1
        
print(correct)


