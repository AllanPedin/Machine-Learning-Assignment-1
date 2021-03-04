import hyperParameters
from dataReader import data
from knn import crossValidation, evaluate


k = hyperParameters.k
distanceFunction = hyperParameters.distanceFunction

trainingData = data.trainingData
testData = data.testData

print("Running with paramaters:::: k =", k, "distancefunction =", distanceFunction.__name__)
print("------------- Cross Validation run ------------- \n")
crossValidation(k,distanceFunction)
print("\n\n\n")
print("------------- Final Test -------------\n")
evaluate(k, trainingData, testData, distanceFunction)