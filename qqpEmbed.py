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
file_path = 'qqpSimilarity.pkl'

allData = []
evalData = set()
idToText = {}
# Open the CSV file for reading

with open('quora_duplicate_questions.tsv', 'r', newline='') as file:
    # Create a CSV reader
    reader = csv.reader(file)

    next(reader)
    # Iterate over each data row in the CSV file
    for row in reader:
        allData.append(row[0].split('\t'))
with open('quora/data/table0.csv', 'r', newline='') as file:
    # Create a CSV reader
    reader = csv.reader(file)

    for row in reader:
        idToText[int(row[0])] = row[1]

with open('quoraVal/data/remap.csv', 'r', newline='') as file:
    reader = csv.reader(file)

    for row in reader:
        evalData.add(int(row[0]))

with open('qqpEmbedLeft.pkl', 'rb') as file:
   leftEmbedding = np.array(pickle.load(file))

with open('qqpEmbedRight.pkl', 'rb') as file:
   rightEmbedding = np.array(pickle.load(file))


negScore = 0
posScore = 1

testSentencesLeft = []
testSentencesRight = []
testPosPairs = set() #(leftIdx,rightIdx)


for idx, d in enumerate(allData):
   if len(d) != 6:
        continue
   try:
      curLabel = int(d[-1])
      curLeftIdx = int(d[1])
      curRightIdx = int(d[2])
      if curLeftIdx in evalData:
         testSentencesLeft.append(idToText[curLeftIdx])
      if curRightIdx in evalData:
         testSentencesRight.append(idToText[curRightIdx])
      if curLeftIdx in evalData and curRightIdx in evalData and curLabel == 1:
         testPosPairs.add((len(testSentencesLeft)-1,len(testSentencesRight)-1))
   except BaseException as e:
      continue
   
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
    if lIdx % 3000 == 0:
      save(similarityResult=similarityResult,file_path=file_path)
      print("saving")
save(similarityResult=similarityResult,file_path=file_path)

