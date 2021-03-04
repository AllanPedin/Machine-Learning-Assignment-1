from scipy.spatial import distance
from constants import CLASS_POSITION
import numpy as np
def euclidean(point1,point2):
    return distance.euclidean(point1,point2)
def mahalanobis(point1,point2, trainingSet):
    covMatrix = np.cov(trainingSet[:CLASS_POSITION], bias=True)
    VI = np.linalg.inv(covMatrix) 
    return distance.mahalanobis(point1,point2, VI)
def cosine(point1,point2):
    return distance.cosine(point1, point2)