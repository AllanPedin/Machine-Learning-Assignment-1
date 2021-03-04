from performance import accuracy, precision, recall, f1score
class classifiable:
    def __init__(self):
        self.truePositives = 0
        self.trueNegatives = 0
        self.falsePositives = 0
        self.falseNegatives = 0
    def getTotal(self):
        return self.truePositives + self.trueNegatives + self.falsePositives + self.falseNegatives
    def printData(self):
        print(self)
        print("TruePositives:",self.truePositives)
        print("TrueNegatives:",self.trueNegatives)
        print("FalsePositives:",self.falsePositives)
        print("FalseNegatives:",self.falseNegatives)
    def calcAccuracy(self):
        return accuracy(self.truePositives, self.trueNegatives, self.getTotal())
    def calcPrecision(self):
        return precision(self.truePositives, self.falsePositives )
    def calcRecall(self):
        return recall(self.truePositives, self.falseNegatives)
    def calcF1score(self):
        return f1score(self.truePositives, self.falsePositives, self.falseNegatives )
    def printPerformance(self):
        print("Class::::::",self.STRING_REP,"::::::")
        print("Accuracy:",  self.calcAccuracy())
        print("Precision:", self.calcPrecision())
        print("Recall:",    self.calcRecall())
        print("F1score:",   self.calcF1score())
    STRING_REP = "This shouldn't ever show up"
    INT_REP = -1
    


class Setosa(classifiable):
    STRING_REP = "Iris-setosa"
    INT_REP = 1
class Versicolor(classifiable):
    STRING_REP = "Iris-versicolor"
    INT_REP = 2
class Virginica(classifiable):
    STRING_REP = "Iris-virginica"
    INT_REP = 3