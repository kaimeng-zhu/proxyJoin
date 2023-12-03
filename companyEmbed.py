import csv
import sys
import os
import math
from tqdm import tqdm
import py_stringmatching as sm
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
csv.field_size_limit(sys.maxsize)
file_path = 'companySimilarity.pkl'

testSentencesLeft = []
testSentencesRight = []
testPosPairs = set()
with open('companyVal/data/table0.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        testSentencesLeft.append(row[2])
with open('companyVal/data/table1.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        testSentencesRight.append(row[2])

with open('companyVal/oracle_labels/01.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        testPosPairs.add((int(row[0]),int(row[1])))

with open('companyEmbedLeft.pkl', 'rb') as file:
   leftEmbedding = np.array(pickle.load(file))

with open('companyEmbedRight.pkl', 'rb') as file:
   rightEmbedding = np.array(pickle.load(file))

def save(similarityResult,file_path):
    with open(file_path, 'wb') as file:
      pickle.dump(similarityResult, file)

maxSequenceLimit = 256
al_tok = sm.AlphabeticTokenizer()
cos = sm.Cosine()
similarityResult = []
tokenizedLeft = [al_tok.tokenize(l)[:maxSequenceLimit] for l in testSentencesLeft]
tokenizedRight = [al_tok.tokenize(r)[:maxSequenceLimit] for r in testSentencesRight]
tfIdf = sm.TfIdf(tokenizedLeft+tokenizedRight)
for lIdx in tqdm(range(len(testSentencesLeft))):
    for rIdx in range(len(testSentencesRight)):
        curLabel = 0
        if (lIdx,rIdx) in testPosPairs:
            curLabel = 1
        a = leftEmbedding[lIdx]
        b = rightEmbedding[rIdx]
        similarityResult.append((cos.get_raw_score(tokenizedLeft[lIdx],tokenizedRight[rIdx]),
                            tfIdf.get_raw_score(tokenizedLeft[lIdx],tokenizedRight[rIdx]),
                            float(dot(a, b)/(norm(a)*norm(b)))
                            ,curLabel))
    if lIdx % 1000 == 0:
        save(similarityResult=similarityResult,file_path=file_path)
        print("saving")
save(similarityResult=similarityResult,file_path=file_path)

