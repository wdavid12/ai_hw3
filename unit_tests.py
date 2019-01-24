import numpy as np
from classifier import  *
from validation import *
from hw3_utils import *

def test1():
    a = np.array([7,4])
    b = np.array([4,8])
    print(euclidean_distance(a,b))

def test2():
    data, classes, _ = load_data()
    to_classify = data[10]
    factory = knn_factory()
    print(type(factory))
    clf = factory.train(data,classes)
    print(type(clf))
    print(clf.classify(to_classify))
    print(classes[10])

def test3(k):
    print('-------------',k,'------------------')
    data, classes, _ = load_data()
    split_crosscheck_groups(data,classes,k)
    for i in range(1,k+1):
        data,labels = load_k_fold_data(i)
        print(i,data.shape,labels.shape)

def test4():
    data, classes, _ = load_data()
    split_crosscheck_groups(data, classes, 2)
    knn3 = knn_factory(3)
    accuracy, error = evaluate(knn3, 2)
    print(accuracy, error)


test1()
test2()
test3(2)
test3(3)
test3(4)
test3(5)
test4()
