def accuracy(truePostives, trueNegatives, total):
    return (truePostives+trueNegatives)/(total)
def precision(truePostives, falsePositives):
    return truePostives/(truePostives+falsePositives)
def recall(truePostives, falseNegatives):
    return truePostives/(truePostives+falseNegatives)
def f1score(truePostives, falsePositives, falseNegatives):
    p = precision(truePostives,falsePositives)
    r = recall(truePostives,falseNegatives)
    return 2*((p*r)/(p+r))