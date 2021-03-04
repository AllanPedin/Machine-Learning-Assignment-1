import constants
from classes import Setosa, Versicolor, Virginica
import numpy as np
import math
import csv

CLASS_POSITION = constants.CLASS_POSITION

def getArrayData(): #returns data in numpy array from dataset
    data = []
    with open(constants.DATA_FILE,"r") as file:
        reader = csv.reader(file)
        for row in reader:
            classification = row[CLASS_POSITION]
            if(classification == Setosa.STRING_REP):
                row[CLASS_POSITION]= 1
            elif(classification == Versicolor.STRING_REP):
                row[CLASS_POSITION]= 2
            elif(classification == Virginica.STRING_REP):
                row[CLASS_POSITION]= 3
            
            for i in range(len(row)) :
                row[i] = float(row[i])
            data.append(row)
    return np.asarray(data)

def randomizeData(data):
    np.random.seed(constants.RANDOM_SEED)
    np.random.shuffle(data)
    return data

def splitTestTraining(data):
    numTestDataPoints = math.floor( len(data) *constants.TEST_DATA_PERCENT)
    return np.split(data, [numTestDataPoints])

def kFoldSplit(data, k):
    return np.array_split(data, k)

class data:
    allData = randomizeData(getArrayData())
    testData, trainingData = splitTestTraining(allData)
    foldedData = kFoldSplit(trainingData, constants.NUM_FOLDS)

