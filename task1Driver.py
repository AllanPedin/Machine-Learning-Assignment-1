from knn import evaluate
from distance import euclidean
from constants import RANDOM_SEED

import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt


np.random.seed(RANDOM_SEED)
#1A
print("A")
mean = [1,1]
covariance = [[.3,.2],[.2,.2]]
samples = 500
distA = np.random.multivariate_normal(mean, covariance, samples)
total = [0,0]
for i in distA:
    total = total + i
average = total/len(distA)
print("average of generated distribution:", average)

#1B
print("B")
mean = [3,4]
covariance = [[.3,0],[0,.2]]
samples = 500
distB = np.random.multivariate_normal(mean, covariance, samples)
total = [0,0]
for i in distB:
    total = total + i
average = total/len(distB)
print("average of generated distribution:", average)

#1C
print("C")
mean = [2,3]
covariance = [[.3,0],[0,.2]]
samples = 300
distC = np.random.multivariate_normal(mean, covariance, samples)

total = [0,0]
for i in distC:
    total = total + i
average = total/len(distC)
print("average of generated distribution:", average)


#2
aClass = []
for i in range(len(distA)):
    aClass.append(np.append(distA[i],[0]))
aClass = np.asarray(aClass)
bClass = []
for i in range(len(distB)):
    bClass.append(np.append(distB[i],[1]))
bClass = np.asarray(bClass)



# classify
CLASS_POSITION = 2
def getKNN(k, trainingSet, point, distanceFunction):
    neighborhood = deepcopy(trainingSet[:k])
    for i in range(k, len(trainingSet)):
        for j in range(len(neighborhood)):
            distanceI = distanceFunction( trainingSet[i][:CLASS_POSITION] , point  )
            distanceJ = distanceFunction( neighborhood[j][:CLASS_POSITION], point  )
            if distanceI < distanceJ:
                neighborhood[j] = deepcopy(trainingSet[i])
                break
    return neighborhood

def vote(knn):
    votes = [0,0]
    for neighbor in knn:
        if neighbor[CLASS_POSITION] == 0:
            votes[0] = votes[0] +1
        elif neighbor[CLASS_POSITION] == 1:
            votes[1] = votes[1] +1
    if votes[0] > votes[1]:
        return 0
    else:
        return 1
def classify(k, trainingSet, dataTobeClassified, distanceFunction):
    classifiedData = []
    for i in range(len(dataTobeClassified)):
        dataPoint = dataTobeClassified[i]
        knn = getKNN(k,trainingSet, dataPoint, euclidean )
        classForPoint = vote(knn)
        classifiedData.append(np.append(dataPoint, classForPoint))
    return np.asarray(classifiedData)

k1 = 1
trainingSet = np.append(aClass,bClass, axis=0)

print("Classifying...")
cClass1 = classify(k1,trainingSet, distC, euclidean )
# print("Classified data:", cClass1)

k30 = 30
print("Classifying...")
cClass30 = classify(k30,trainingSet, distC, euclidean )
# print("Classified data:", cClass1)

#3

def makePlot(class1,class2, classifiedData, k):
    predicted1_samples = []
    predicted2_samples = []
    for point in classifiedData:
        if point[2] == 0:
            predicted1_samples.append(point)
        else:
            predicted2_samples.append(point)
    predicted1_samples = np.asarray(predicted1_samples)
    predicted2_samples = np.asarray(predicted2_samples)
    print("Length1", len(predicted1_samples))
    print("Length2", len(predicted2_samples))


    fig_output = "k = "+str(k)
    fig_title = "KNN ("+fig_output+") Euclidean Distance Classification"

    fig = plt.figure()
    plt.plot(class1[:, 0], class1[:, 1], 'b.', label='given class a')
    plt.plot(class2[:, 0], class2[:, 1], 'r.', label='given class b')
    plt.plot(predicted1_samples[:, 0], predicted1_samples[:, 1], 'g*', label='predicted class a')
    plt.plot(predicted2_samples[:, 0], predicted2_samples[:, 1], '*', color='orange', label='predicated class b')
    plt.xlabel('X-axis')
    plt.ylabel('Y-Axis')
    plt.title(fig_title)
    plt.tight_layout()
    plt.grid(True, lw=0.5)
    plt.legend()
    fig.savefig(fig_output)

makePlot(aClass,bClass,cClass1,k1)
makePlot(aClass,bClass,cClass30,k30)



