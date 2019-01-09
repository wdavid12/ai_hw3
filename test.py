import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron

from classifier import *
import validation
from hw3_utils import *

def knn_validation():
    for i in [1,3,5,7,13]:
        knn = knn_factory(i)
        acc, err = validation.evaluate(knn, 2)
        print(i,acc,err, sep=',')

def sklearn_validation():
    clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
    factory = sklearn_factory(clf)
    acc, err = validation.evaluate(factory, 2)
    print(1,acc, err, sep=',')

    clf = Perceptron()
    factory = sklearn_factory(clf)
    acc, err = validation.evaluate(factory, 2)
    print(2,acc, err, sep=',')

def main():
    # knn_validation()
    sklearn_validation()

if __name__ == '__main__':
    main()
