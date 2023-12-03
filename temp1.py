import csv
import sys
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
from tqdm import tqdm
import py_stringmatching as sm
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
csv.field_size_limit(sys.maxsize)
import random
#'''
with open('qqpSimilarity.pkl', 'rb') as file:
    readList = pickle.load(file)

def randomSample(size, pool):
    randomSample = random.choices(pool, k=size)
    posCounter = 0
    for s in randomSample:
        if s[3] == 1:
            posCounter += 1
    return posCounter / size

def importanceSample(size, pool, weight,sumWeight,weightIdx):
    sample = random.choices(pool, weights=weight,k=size)
    mean = 0
    for s in sample:
        if s[3] == 1:
            mean += sumWeight / (len(pool) * s[weightIdx])
    return mean/size

numTrail = 20
totalSizeList = [1000000,10000000]

cosWeight = [r[0] for r in readList]
cosWeightSum = sum(cosWeight)
tfIdfWeight = [r[1] for r in readList]
tfIdfWeightSum = sum(tfIdfWeight)
encoderWeight = [r[2] for r in readList]
encoderWeightSum = sum(encoderWeight)
resultList = []
for totalSize in totalSizeList:
    randomMean = []
    cosMean = []
    tfIdfMean = []
    encoderMean = []
    for _ in range(numTrail):
        randomResult = len(readList) * randomSample(size=totalSize,pool=readList)
        cosResult = len(readList) * importanceSample(size = totalSize, pool = readList,weight=cosWeight,sumWeight=cosWeightSum,weightIdx=0)
        tfIdfResult = len(readList) * importanceSample(size=totalSize,pool=readList,weight=tfIdfWeight,sumWeight=tfIdfWeightSum,weightIdx=1)
        encoderResult = len(readList) * importanceSample(size=totalSize,pool=readList,weight=encoderWeight,sumWeight=encoderWeightSum,weightIdx=2)

        randomMean.append(randomResult)
        cosMean.append(cosResult)
        tfIdfMean.append(tfIdfResult)
        encoderMean.append(encoderResult)
    print(np.mean(randomMean))
    print(np.mean(cosMean))
    print(np.mean(tfIdfMean))
    print(np.mean(encoderMean))
    result = (np.var(randomMean),np.var(cosMean),np.var(tfIdfMean),np.var(encoderMean))
    print(result)
    resultList.append(result)

with open('qqpvarianceResult.pkl', 'wb') as file:
      pickle.dump(resultList, file)
#'''
'''
with open('qqpSimilarity.pkl', 'rb') as file:
    readList = pickle.load(file)

BlockingWeightFirst = [(r[2],r[3]) for r in readList]
BlockingWeightFirst.sort(reverse=True)

totalPos = 0
for b in BlockingWeightFirst:
    if b[1] == 1:
        totalPos += 1

threasholdSizeList = [5000,10000,50000,100000,500000,1000000,10000000]
recall = []
fp = []
fn = []
for thresholdSize in threasholdSizeList:
    upperPos = 0
    for b in BlockingWeightFirst[:thresholdSize]:
        if b[1] == 1:
            upperPos += 1
    recall.append(upperPos/totalPos)
    fp.append((thresholdSize-upperPos)/thresholdSize)
    fn.append(upperPos/(len(BlockingWeightFirst) - thresholdSize))


qqpgraph = [recall,fp,fn]


with open('qqpGraph.pkl', 'wb') as file:
      pickle.dump(qqpgraph, file)
'''