import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import tree

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

def competition():
    data, classes, _ = load_data()
    classes  = np.array(classes, dtype=bool)
    clfs = {
        'knn1': KNeighborsClassifier(n_neighbors=1),
        'knn3': KNeighborsClassifier(n_neighbors=3),
        'knn5': KNeighborsClassifier(),
        'tree1': DecisionTreeClassifier(criterion='entropy'),
        'tree_gini': DecisionTreeClassifier(),
        'tree_trim': DecisionTreeClassifier(min_samples_split=10),
        'svm': svm.SVC(),
    }
    print('name','mean','std',sep=',')

    for k, clf in clfs.items():
        scores = cross_val_score(clf, data, classes, cv=5)
        print(k, scores.mean(), scores.std(), sep=',')

def dot():
    data, classes, _ = load_data()
    classes  = np.array(classes, dtype=bool)
    clf = DecisionTreeClassifier(criterion='gini', min_samples_split=25, max_depth=6)
    scores = cross_val_score(clf, data, classes, cv=3)
    print(scores.mean(), scores.std(), sep=',')
    clf.fit(data, classes)
    dot_data = tree.export_graphviz(clf, out_file=None)
    print(dot_data)

def feature_selection_tree():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel, VarianceThreshold
    X, y, _ = load_data()
    y  = np.array(y, dtype=bool)

    # clf = RandomForestClassifier(n_estimators=100)
    # clf = ExtraTreesClassifier(n_estimators=50)
    clf = DecisionTreeClassifier(criterion='gini', min_samples_split=25, max_depth=6)
    # clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=25)
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True, max_features=55, threshold=-np.inf)
    X_new = model.transform(X)
    clf = DecisionTreeClassifier(criterion='gini', min_samples_split=25, max_depth=6)
    clf.fit(X_new, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    print(dot_data)


def feature_selection():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.feature_selection import SelectFromModel, VarianceThreshold

    knn_L1_K1 = knn_factory(k=1)
    knn_L1_K3 = knn_factory(k=3)
    knn1 = KNeighborsClassifier(n_neighbors=1, weights = 'distance')
    knn3 = KNeighborsClassifier(n_neighbors=3, weights = 'distance')

    knn5 = KNeighborsClassifier(n_neighbors=5, weights = 'distance')
    knn7 = KNeighborsClassifier(n_neighbors=7, weights = 'distance')

    X, y, _ = load_data()
    y  = np.array(y, dtype=bool)

    # clf = RandomForestClassifier(n_estimators=100)
    # clf = ExtraTreesClassifier(n_estimators=50)
    clf = DecisionTreeClassifier(criterion='gini', min_samples_split=25, max_depth=6)
    # clf = DecisionTreeClassifier(criterion='entropy', min_samples_split=25)
    print("BEFORE:")
    print(X.shape)
    acc, err = validation.evaluate_matrix(knn_L1_K1,X,y, 2)
    print("KNN_L1_K1",acc, err, sep=',')
    acc, err = validation.evaluate_matrix(knn_L1_K3,X,y ,2)
    print("KNN_L1_K3",acc, err, sep=',')
    print("KNN=1", cross_val_score(knn1, X, y, cv=3).mean())
    print("KNN=3", cross_val_score(knn3, X, y, cv=3).mean())
    print("KNN=5", cross_val_score(knn5, X, y, cv=3).mean())
    print("KNN=7", cross_val_score(knn7, X, y, cv=3).mean())
    print("Tree", cross_val_score(clf, X, y, cv=3).mean())

    # clf = clf.fit(X, y)
    # results_K1 = []
    # results_K3 = []
    # results_K3_sk = []
    # for i in range(15,186,5):
        # model = SelectFromModel(clf, prefit=True, max_features=i, threshold=-np.inf)
        # X_new = model.transform(X)
        # print(X_new.shape)

        # acc, err = validation.evaluate_matrix(knn_L1_K1,X_new,y, 2)
        # print("KNN_L1_K1",acc, err, sep=',')
        # results_K1.append(acc)
        # acc, err = validation.evaluate_matrix(knn_L1_K3,X_new,y, 2)
        # print("KNN_L1_K3",acc, err, sep=',')
        # results_K3.append(acc)
        # results_K3_sk.append(cross_val_score(knn3, X_new,y, cv=3).mean())
        # # print("KNN=1", cross_val_score(knn1, X_new, y, cv=3).mean())
        # # print("KNN=3", cross_val_score(knn3, X_new, y, cv=3).mean())
        # # print("KNN=5", cross_val_score(knn5, X_new, y, cv=3).mean())
        # # print("KNN=7", cross_val_score(knn7, X_new, y, cv=3).mean())
        # # print("Tree", cross_val_score(clf, X_new, y, cv=3).mean())
    # m1, m3, m3_sk =max(results_K1), max(results_K3), max(results_K3_sk)
    # print("K1:", m1, "K3:", m3, "K3_sk",m3_sk)
    # for i in range(len(results_K1)):
        # if results_K1[i] == m1:
            # print("K1 idx:", i*5)
        # if results_K3[i] == m3:
            # print("K3 idx:", i*5)
        # if results_K3_sk[i] == m3_sk:
            # print("K1 idx:", i*5)

def fft_test():
    from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    from sklearn.preprocessing import normalize
    X, y, _ = load_data()
    y  = np.array(y, dtype=bool)
    fft = np.fft.fft(X, axis=1)
    fft = np.abs(fft)**2
    fft = normalize(np.argsort(fft, axis=1 )[:,165:187], axis=0)
    X_new = np.concatenate([X,fft], axis=1)
    # X_new = fft
    knn1 = KNeighborsClassifier(n_neighbors=1)
    knn3 = KNeighborsClassifier(n_neighbors=3)
    knn5 = KNeighborsClassifier(n_neighbors=5)
    clf = RandomForestClassifier(n_estimators=100)
    print("KNN=1", cross_val_score(knn1, X_new, y, cv=3).mean())
    print("KNN=3", cross_val_score(knn3, X_new, y, cv=3).mean())
    print("KNN=5", cross_val_score(knn5, X_new, y, cv=3).mean())
    print("Tree", cross_val_score(clf, X_new, y, cv=3).mean())


def main():
    # knn_validation()
    # sklearn_validation()
    # competition()
    # dot()
    # feature_selection()
    # feature_selection_tree()
    fft_test()

if __name__ == '__main__':
    main()
