from constants import CLASS_POSITION
from classes import Setosa, Versicolor, Virginica
from dataReader import data
from distance import euclidean, mahalanobis, cosine
import hyperParameters

import numpy as np
from copy import deepcopy


def getKNN(k, trainingSet, point, distanceFunction):
    neighborhood = deepcopy(trainingSet[:k])
    for i in range(k, len(trainingSet)):
        for j in range(len(neighborhood)):
            if distanceFunction == mahalanobis:
                distanceI = distanceFunction( trainingSet[i][:CLASS_POSITION] , point[:CLASS_POSITION], trainingSet  )
                distanceJ = distanceFunction( neighborhood[j][:CLASS_POSITION], point[:CLASS_POSITION], trainingSet  )
                if distanceI < distanceJ:
                    neighborhood[j] = deepcopy(trainingSet[i])
                    break
            else:
                distanceI = distanceFunction( trainingSet[i][:CLASS_POSITION] , point[:CLASS_POSITION]  )
                distanceJ = distanceFunction( neighborhood[j][:CLASS_POSITION], point[:CLASS_POSITION]  )
                if distanceI < distanceJ:
                    neighborhood[j] = deepcopy(trainingSet[i])
                    break
    return neighborhood

def vote(knn):
    votes = [0,0,0]
    for neighbor in knn:
        if neighbor[CLASS_POSITION] == Setosa.INT_REP:
            votes[0] = votes[0] +1
        elif neighbor[CLASS_POSITION] == Versicolor.INT_REP:
            votes[1] = votes[1] +1
        elif neighbor[CLASS_POSITION] == Virginica.INT_REP:
            votes[2] = votes[2] +1
    if votes[0] >= votes[1] and votes[0] >= votes[2]:
        return Setosa.INT_REP
    elif votes[1] >= votes[0] and votes[1] >= votes[2]:
        return Versicolor.INT_REP
    else:
        return Virginica.INT_REP



def evaluate(k,trainingSet, testSet, distanceFunction):
    setosa = Setosa()
    versicolor = Versicolor()
    virginica = Virginica()
    for point in testSet:
        knn = getKNN(k, trainingSet, point, distanceFunction)
        classInt = vote(knn)
        if classInt == point[CLASS_POSITION]: #properly classified
            if classInt == setosa.INT_REP:
                setosa.truePositives +=1
                versicolor.trueNegatives +=1
                virginica.trueNegatives+=1
            elif classInt == versicolor.INT_REP:
                setosa.trueNegatives +=1
                versicolor.truePositives +=1
                virginica.trueNegatives+=1
            else:
                setosa.trueNegatives +=1
                versicolor.trueNegatives +=1
                virginica.truePositives+=1
        else: #improperly classified
            if classInt == setosa.INT_REP:
                setosa.falsePositives +=1
                if point[CLASS_POSITION] == versicolor.INT_REP:
                    versicolor.falseNegatives+=1
                    virginica.trueNegatives+=1
                else:
                    versicolor.trueNegatives+=1
                    virginica.falseNegatives+=1     
            elif classInt == versicolor.INT_REP:
                versicolor.falsePositives +=1
                if point[CLASS_POSITION] == setosa.INT_REP:
                    setosa.falseNegatives+=1
                    virginica.trueNegatives+=1
                else:
                    setosa.trueNegatives+=1
                    virginica.falseNegatives+=1
            else:
                virginica.falsePositives +=1
                if point[CLASS_POSITION] == setosa.INT_REP:
                    setosa.falseNegatives+=1
                    versicolor.trueNegatives+=1
                else:
                    setosa.trueNegatives+=1
                    versicolor.falseNegatives+=1   
    setosa.printPerformance()
    versicolor.printPerformance()
    virginica.printPerformance()
    printMacroAverages(setosa, versicolor, virginica)
    return (setosa, versicolor, virginica)

def crossValidation(k, distanceFunction):
    folds = data.foldedData
    for i in range(len(folds)):
        print("------------- Start Fold", i, "-------------")
        testFold = deepcopy(folds[i])
        foldsCopy = deepcopy(folds)
        folds1Removed  = np.delete(foldsCopy, i, 0)
        trainingData = np.concatenate(folds1Removed)
        evaluate(k, trainingData, testFold, distanceFunction)
        print("------------- End Fold", i, "-------------")

def printMacroAverages(setosa, versicolor, virginica):
        accuracy = (setosa.calcAccuracy() + versicolor.calcAccuracy() + virginica.calcAccuracy())/3
        Precision = (setosa.calcPrecision() + versicolor.calcPrecision() + virginica.calcPrecision())/3
        Recall = (setosa.calcRecall() + versicolor.calcRecall() + virginica.calcRecall())/3
        F1score = (setosa.calcF1score() + versicolor.calcF1score() + virginica.calcF1score())/3

        print("Macro Averaged Accuracy:",  accuracy)
        print("Macro Averaged Precision:", Precision)
        print("Macro Averaged Recall:",    Recall)
        print("Macro Averaged F1score:",   F1score)